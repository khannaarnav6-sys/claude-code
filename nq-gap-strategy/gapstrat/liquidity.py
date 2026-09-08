"""The liquidity map: every level price is drawn toward, built before the open.

ICT's premise is that intraday price moves between pools of resting orders --
the highs and lows where stops sit -- rather than wandering. A setup is only
"A+" when you can name the pool it just took and the pool it is heading for.
This module builds that map for one session, from bars that had all printed
before 09:30.

Sessions, in New York time, are the conventional ICT windows:

    Asia      20:00 (prior evening) -- 00:00
    London    02:00 -- 05:00
    NY open   09:30, with the killzone running 08:30 -- 11:00

The weekly levels matter for a different reason. Futures close Friday at 17:00
and reopen Sunday at 18:00, and the space between those two prices -- the new
week opening gap -- is both a magnet and a support shelf that price returns to
for days afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

ET = "America/New_York"


@dataclass(frozen=True)
class Level:
    """One pool of liquidity, named so a trade can say what it is aiming at."""

    name: str
    price: float
    kind: str  # "high" or "low"


@dataclass
class LiquidityMap:
    day: object
    session_open: float | None = None
    asia_high: float | None = None
    asia_low: float | None = None
    london_high: float | None = None
    london_low: float | None = None
    overnight_high: float | None = None
    overnight_low: float | None = None
    prior_day_high: float | None = None
    prior_day_low: float | None = None
    prior_day_close: float | None = None
    prior_week_high: float | None = None
    prior_week_low: float | None = None
    nwog_high: float | None = None  # new week opening gap: Friday close -> Sunday open
    nwog_low: float | None = None
    weekly_fvgs: list = field(default_factory=list)  # (low, high, direction)
    daily_fvgs: list = field(default_factory=list)

    def highs(self) -> list[Level]:
        """Pools above, as targets for a long or liquidity for a short to take."""
        named = [
            ("asia_high", self.asia_high),
            ("london_high", self.london_high),
            ("overnight_high", self.overnight_high),
            ("prior_day_high", self.prior_day_high),
            ("prior_week_high", self.prior_week_high),
        ]
        return [Level(n, p, "high") for n, p in named if p is not None]

    def lows(self) -> list[Level]:
        named = [
            ("asia_low", self.asia_low),
            ("london_low", self.london_low),
            ("overnight_low", self.overnight_low),
            ("prior_day_low", self.prior_day_low),
            ("prior_week_low", self.prior_week_low),
        ]
        return [Level(n, p, "low") for n, p in named if p is not None]

    def draw(self, direction: int, above: float, named: str = "asia") -> Level | None:
        """The pool this trade is aiming at.

        `named` picks the family -- "asia" is the session the user watches, but
        the same machinery serves prior-day or prior-week targets, and
        "nearest" takes whichever pool is closest beyond the entry, which is
        the most conservative reading of a draw on liquidity.
        """
        pools = self.highs() if direction > 0 else self.lows()
        beyond = [
            lv for lv in pools if (lv.price > above if direction > 0 else lv.price < above)
        ]
        if not beyond:
            return None
        if named == "nearest":
            return min(beyond, key=lambda lv: abs(lv.price - above))
        if named == "furthest":
            return max(beyond, key=lambda lv: abs(lv.price - above))
        wanted = [lv for lv in beyond if lv.name.startswith(named)]
        return wanted[0] if wanted else None


def _window(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return bars[(bars.index >= start) & (bars.index < end)]


def _hi_lo(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    if frame.empty:
        return None, None
    return float(frame["high"].max()), float(frame["low"].min())


def new_week_opening_gap(bars: pd.DataFrame, day, tz) -> tuple[float | None, float | None]:
    """The space between Friday's 17:00 close and Sunday's 18:00 reopen.

    Returned low-first regardless of gap direction, since it is used as a zone
    rather than a signed move.
    """
    open_ts = pd.Timestamp(day.year, day.month, day.day, 9, 30, tz=tz)
    # Walk back to the most recent Sunday reopen at or before this session.
    days_since_sunday = (open_ts.weekday() + 1) % 7
    sunday = (open_ts - pd.Timedelta(days=days_since_sunday)).normalize().tz_localize(None)
    sunday = pd.Timestamp(sunday, tz=tz)

    sunday_open_window = _window(
        bars, sunday + pd.Timedelta(hours=18), sunday + pd.Timedelta(hours=20)
    )
    friday = sunday - pd.Timedelta(days=2)
    friday_close_window = _window(
        bars, friday + pd.Timedelta(hours=16, minutes=30), friday + pd.Timedelta(hours=17)
    )
    if sunday_open_window.empty or friday_close_window.empty:
        return None, None
    sunday_open = float(sunday_open_window["open"].iloc[0])
    friday_close = float(friday_close_window["close"].iloc[-1])
    return min(sunday_open, friday_close), max(sunday_open, friday_close)


def timeframe_fvgs(bars: pd.DataFrame, rule: str, lookback: int = 8) -> list[tuple[float, float, int]]:
    """Fair value gaps on a higher timeframe, newest last.

    Weekly and daily imbalances are context rather than triggers: they say
    which direction the larger move is unbalanced in, and they act as shelves
    an intraday move can stall at.
    """
    from .data import resample
    from .gaps import find_gaps

    higher = resample(bars, rule)
    if len(higher) < 3:
        return []
    return [
        (g.low, g.high, g.direction) for g in find_gaps(higher.iloc[-(lookback + 3) :])
    ]


def build(bars: pd.DataFrame, day, prior_day=None, prior_week_days: list | None = None) -> LiquidityMap:
    """Assemble the map for one session using only pre-open information."""
    tz = bars.index.tz
    open_ts = pd.Timestamp(day.year, day.month, day.day, 9, 30, tz=tz)
    midnight = open_ts.normalize()

    asia = _window(bars, midnight - pd.Timedelta(hours=4), midnight)  # 20:00 -> 00:00
    london = _window(bars, midnight + pd.Timedelta(hours=2), midnight + pd.Timedelta(hours=5))
    overnight = _window(bars, open_ts - pd.Timedelta(hours=15, minutes=30), open_ts)
    session = _window(bars, open_ts, open_ts + pd.Timedelta(hours=7))

    asia_high, asia_low = _hi_lo(asia)
    london_high, london_low = _hi_lo(london)
    overnight_high, overnight_low = _hi_lo(overnight)

    prior_high = prior_low = prior_close = None
    if prior_day is not None:
        prior_open = pd.Timestamp(prior_day.year, prior_day.month, prior_day.day, 9, 30, tz=tz)
        prior_rth = _window(bars, prior_open, prior_open + pd.Timedelta(hours=6, minutes=30))
        if not prior_rth.empty:
            prior_high = float(prior_rth["high"].max())
            prior_low = float(prior_rth["low"].min())
            prior_close = float(prior_rth["close"].iloc[-1])

    week_high = week_low = None
    if prior_week_days:
        frames = []
        for d in prior_week_days:
            o = pd.Timestamp(d.year, d.month, d.day, 9, 30, tz=tz)
            frames.append(_window(bars, o, o + pd.Timedelta(hours=6, minutes=30)))
        frames = [f for f in frames if not f.empty]
        if frames:
            week = pd.concat(frames)
            week_high, week_low = _hi_lo(week)

    nwog_low, nwog_high = new_week_opening_gap(bars, day, tz)

    return LiquidityMap(
        day=day,
        session_open=float(session["open"].iloc[0]) if not session.empty else None,
        asia_high=asia_high,
        asia_low=asia_low,
        london_high=london_high,
        london_low=london_low,
        overnight_high=overnight_high,
        overnight_low=overnight_low,
        prior_day_high=prior_high,
        prior_day_low=prior_low,
        prior_day_close=prior_close,
        prior_week_high=week_high,
        prior_week_low=week_low,
        nwog_high=nwog_high,
        nwog_low=nwog_low,
        weekly_fvgs=timeframe_fvgs(bars[bars.index < open_ts], "1W"),
        daily_fvgs=timeframe_fvgs(bars[bars.index < open_ts], "1D"),
    )
