"""Loading and shaping NQ futures intraday bars.

Bars come from the Yahoo Finance chart endpoint, which is the only free source
reachable from this environment. Its intraday history is capped: 1-minute bars
go back ~30 days and must be pulled in <=8 day windows, 5-minute bars go back
60 days. Everything is cached to CSV so a backtest run does not re-download.

Timestamps are bar-open times, converted to America/New_York, which is the
timezone every session rule in this project is written in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0"}
ET = "America/New_York"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# How far back each interval is actually served, and the largest window the
# endpoint accepts in a single request.
INTERVAL_LIMITS = {
    "1m": (29, 7),
    "2m": (59, 59),
    "5m": (59, 59),
    "15m": (59, 59),
}


@dataclass(frozen=True)
class Contract:
    """Futures contract specs used for position sizing and P&L."""

    symbol: str
    tick_size: float
    point_value: float
    commission_round_turn: float

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.point_value


NQ = Contract("NQ=F", tick_size=0.25, point_value=20.0, commission_round_turn=4.50)
MNQ = Contract("NQ=F", tick_size=0.25, point_value=2.0, commission_round_turn=1.00)


def _fetch(symbol: str, interval: str, **params) -> pd.DataFrame:
    resp = requests.get(
        CHART_URL.format(symbol=requests.utils.quote(symbol, safe="")),
        params={"interval": interval, **params},
        headers=HEADERS,
        timeout=45,
    )
    resp.raise_for_status()
    chart = resp.json()["chart"]
    if chart.get("error"):
        raise RuntimeError(f"Yahoo error for {interval} {params}: {chart['error']}")
    result = chart["result"][0]
    stamps = result.get("timestamp")
    if not stamps:
        return pd.DataFrame()
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        },
        index=pd.to_datetime(stamps, unit="s", utc=True),
    )
    return frame


def download(symbol: str, interval: str, lookback_days: int | None = None) -> pd.DataFrame:
    """Pull as much history as the endpoint will serve for `interval`."""
    if interval not in INTERVAL_LIMITS:
        raise ValueError(f"unsupported interval {interval!r}")
    max_lookback, chunk = INTERVAL_LIMITS[interval]
    lookback = min(lookback_days or max_lookback, max_lookback)

    frames = []
    if chunk >= lookback:
        # A single `range` request reaches further back than an equivalent
        # period1/period2 window does -- Yahoo counts range in trading days.
        frames.append(_fetch(symbol, interval, range=f"{lookback}d"))
    else:
        now = int(time.time())
        cursor = lookback
        while cursor > 0:
            window_end = max(cursor - chunk, 0)
            frames.append(
                _fetch(symbol, interval, period1=now - cursor * 86400, period2=now - window_end * 86400)
            )
            cursor = window_end
            time.sleep(0.4)

    if not frames:
        return pd.DataFrame()
    bars = pd.concat([f for f in frames if not f.empty])
    bars = bars[~bars.index.duplicated(keep="first")].sort_index()
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars.index = bars.index.tz_convert(ET)
    bars.index.name = "timestamp"
    return bars


def cache_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace("=", "").replace("/", "")
    return DATA_DIR / f"{safe}_{interval}.csv"


def load(symbol: str, interval: str, refresh: bool = False) -> pd.DataFrame:
    """Return cached bars, downloading them first if needed."""
    path = cache_path(symbol, interval)
    if refresh or not path.exists():
        bars = download(symbol, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        bars.to_csv(path)
        return bars
    bars = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    bars.index = bars.index.tz_convert(ET)
    return bars


def resample(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate to a coarser bar size, anchored so buckets align with :00.

    Anchoring matters: a 5-minute signal bar has to start on 9:30, 9:35, ...
    the way a chart would draw it, not on whatever the first row happens to be.
    """
    agg = bars.resample(rule, label="left", closed="left", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return agg.dropna(subset=["open", "high", "low", "close"])


def sessions(bars: pd.DataFrame) -> list[pd.Timestamp]:
    """Distinct calendar dates present in the bar index, in order."""
    return sorted({ts.date() for ts in bars.index})


def session_slice(
    bars: pd.DataFrame,
    day,
    start: str = "09:30",
    end: str = "16:00",
) -> pd.DataFrame:
    """Bars for one date between two wall-clock ET times, end-inclusive."""
    day_bars = bars[[ts.date() == day for ts in bars.index]]
    return day_bars.between_time(start, end, inclusive="both")


def coverage_report(bars: pd.DataFrame, start: str = "09:30", end: str = "16:00") -> pd.DataFrame:
    """Bars per regular-hours session, for spotting holes in the feed."""
    rth = bars.between_time(start, end, inclusive="left")
    grouped = rth.groupby([ts.date() for ts in rth.index])
    return pd.DataFrame(
        {
            "bars": grouped.size(),
            "first": grouped.apply(lambda g: g.index[0].strftime("%H:%M")),
            "last": grouped.apply(lambda g: g.index[-1].strftime("%H:%M")),
        }
    )


# Index futures that trade the same New York open. Testing one set of rules
# across all four is the cheapest out-of-sample check available here: the
# instruments are correlated enough that the setup should appear in each, and
# independent enough that a result true of only one is probably luck.
CONTRACTS = {
    "NQ=F": Contract("NQ=F", tick_size=0.25, point_value=20.0, commission_round_turn=4.50),
    "ES=F": Contract("ES=F", tick_size=0.25, point_value=50.0, commission_round_turn=4.50),
    "YM=F": Contract("YM=F", tick_size=1.00, point_value=5.00, commission_round_turn=4.50),
    "RTY=F": Contract("RTY=F", tick_size=0.10, point_value=50.0, commission_round_turn=4.50),
}

# NQ's absolute thresholds (5 point minimum gap, 100 point maximum stop) as
# fractions of its own median session range, so they can be carried to an
# instrument priced at 2,900 or 53,000 without meaning something different.
GAP_FRACTION = 0.0133
MAX_STOP_FRACTION = 0.267


def median_session_range(bars: pd.DataFrame, days: list) -> float:
    """Median high-to-low of the regular session, used to scale thresholds."""
    ranges = []
    for day in days:
        session = session_slice(bars, day, "09:30", "15:55")
        if not session.empty:
            ranges.append(float(session["high"].max() - session["low"].min()))
    return float(pd.Series(ranges).median()) if ranges else 0.0
