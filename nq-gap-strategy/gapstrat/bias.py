"""Deciding which way the session is leaning before taking a gap.

The base strategy takes the first gap and trades whichever way it points. That
throws away the question every discretionary trader asks first: *which side am I
looking for today?* This module answers it, and the strategy can then skip gaps
that argue against the session's own bias.

Every method here is evaluated with bars up to and including the signal bar and
never past it, so a bias is only ever formed from what was on the screen when
the order would have gone in.

  overnight_break  price trading outside the overnight range picks the side.
  opening_range    the most recent break of the first 15 minutes' range.
  sweep_reversal   price takes out a prior low and closes back above it -- the
                   stop-run-then-reverse pattern -- which turns the bias up.
  prior_day        today's open above or below yesterday's regular-hours close.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

METHODS = ("none", "overnight_break", "opening_range", "sweep_reversal", "prior_day")


@dataclass(frozen=True)
class SessionLevels:
    """Reference prices known before the New York open."""

    prior_high: float | None
    prior_low: float | None
    prior_close: float | None
    overnight_high: float | None
    overnight_low: float | None
    session_open: float | None

    @property
    def complete(self) -> bool:
        return None not in (self.overnight_high, self.overnight_low, self.session_open)


def session_levels(bars: pd.DataFrame, day, prior_day=None) -> SessionLevels:
    """Build the pre-open reference levels for one session.

    The overnight window runs from 18:00 the previous evening -- the CME
    reopen -- through 09:29, which is the liquidity the New York open trades
    against.
    """
    tz = bars.index.tz
    open_ts = pd.Timestamp(day.year, day.month, day.day, 9, 30, tz=tz)
    overnight_start = open_ts - pd.Timedelta(hours=15, minutes=30)  # 18:00 prior evening

    overnight = bars[(bars.index >= overnight_start) & (bars.index < open_ts)]
    session = bars[bars.index >= open_ts]
    session = session[session.index < open_ts + pd.Timedelta(hours=7)]

    prior_high = prior_low = prior_close = None
    if prior_day is not None:
        prior_open = pd.Timestamp(prior_day.year, prior_day.month, prior_day.day, 9, 30, tz=tz)
        prior_rth = bars[(bars.index >= prior_open) & (bars.index < prior_open + pd.Timedelta(hours=6, minutes=30))]
        if not prior_rth.empty:
            prior_high = float(prior_rth["high"].max())
            prior_low = float(prior_rth["low"].min())
            prior_close = float(prior_rth["close"].iloc[-1])

    return SessionLevels(
        prior_high=prior_high,
        prior_low=prior_low,
        prior_close=prior_close,
        overnight_high=float(overnight["high"].max()) if not overnight.empty else None,
        overnight_low=float(overnight["low"].min()) if not overnight.empty else None,
        session_open=float(session["open"].iloc[0]) if not session.empty else None,
    )


def _overnight_break(seen: pd.DataFrame, levels: SessionLevels) -> int:
    if levels.overnight_high is None or levels.overnight_low is None:
        return 0
    last = float(seen["close"].iloc[-1])
    if last > levels.overnight_high:
        return 1
    if last < levels.overnight_low:
        return -1
    return 0  # still inside the range: no side to take


def _opening_range(seen: pd.DataFrame, levels: SessionLevels, bars_in_range: int = 3) -> int:
    """Most recent close beyond the first 15 minutes' range wins."""
    if len(seen) <= bars_in_range:
        return 0
    opening = seen.iloc[:bars_in_range]
    high, low = float(opening["high"].max()), float(opening["low"].min())
    side = 0
    for close in seen["close"].iloc[bars_in_range:]:
        if close > high:
            side = 1
        elif close < low:
            side = -1
    return side


def _sweep_reversal(seen: pd.DataFrame, levels: SessionLevels) -> int:
    """A run on prior liquidity that fails, which is the pattern on the chart.

    Price trades below a reference low and then closes back above it: the
    sellers who broke it are trapped, and the bias flips up. Mirrored for a
    failed run at the highs. The most recent sweep is the one that counts.
    """
    lows = [lv for lv in (levels.overnight_low, levels.prior_low) if lv is not None]
    highs = [lv for lv in (levels.overnight_high, levels.prior_high) if lv is not None]
    if not lows and not highs:
        return 0

    side = 0
    for _, bar in seen.iterrows():
        for low in lows:
            if bar["low"] < low <= bar["close"]:
                side = 1
        for high in highs:
            if bar["high"] > high >= bar["close"]:
                side = -1
    return side


def _prior_day(seen: pd.DataFrame, levels: SessionLevels) -> int:
    if levels.prior_close is None or levels.session_open is None:
        return 0
    if levels.session_open > levels.prior_close:
        return 1
    if levels.session_open < levels.prior_close:
        return -1
    return 0


_DISPATCH = {
    "overnight_break": _overnight_break,
    "opening_range": _opening_range,
    "sweep_reversal": _sweep_reversal,
    "prior_day": _prior_day,
}


def determine(method: str, seen: pd.DataFrame, levels: SessionLevels) -> int:
    """Session bias from bars `seen` so far: +1 long, -1 short, 0 undecided."""
    if method == "none":
        return 0
    if method not in _DISPATCH:
        raise ValueError(f"unknown bias method {method!r}")
    if seen.empty:
        return 0
    return _DISPATCH[method](seen, levels)
