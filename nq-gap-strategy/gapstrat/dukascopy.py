"""Long-history intraday bars from Dukascopy's public tick feed.

Yahoo serves 60 days of 5-minute data, which is why every result in this
project rests on 49 sessions. Dukascopy publishes hourly tick files going back
years, free and without a key, which is the only route to a real holdout here.

    https://datafeed.dukascopy.com/datafeed/<INSTRUMENT>/<YYYY>/<MM>/<DD>/<HH>h_ticks.bi5

Two things about that URL bite: the month is **zero-indexed** (05 is June), and
the hour is UTC. Each file is raw LZMA holding 20-byte big-endian records --
milliseconds into the hour, ask, bid, ask volume, bid volume -- with prices as
integers scaled by 1000 for an index.

WHAT THIS DATA IS: `USATECHIDXUSD` is Dukascopy's Nasdaq 100 **index CFD**, not
the NQ futures contract. It tracks the same underlying through the same
sessions, so it is a fair test of whether a rule describes the New York open,
but it is not the instrument traded: no exchange volume, a broker's spread, and
prices that sit at the index rather than at the futures basis. Results from it
belong in the "does this rule generalise" column, never in the "this is what NQ
did" column.
"""

from __future__ import annotations

import lzma
import random
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import DATA_DIR, ET

FEED = "https://datafeed.dukascopy.com/datafeed/{inst}/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
NASDAQ_CFD = "USATECHIDXUSD"
PRICE_SCALE = 1000.0
RECORD = struct.Struct(">IIIff")

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0"})
_session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=48, pool_maxsize=48))


def _fetch_hour(instrument: str, day: date, hour: int, tries: int = 7) -> bytes:
    """One hour of ticks. Empty bytes means the market was shut, not an error.

    The egress proxy drops tunnels when several TLS handshakes start at once,
    so this backs off with jitter rather than treating a reset as a real miss.
    """
    url = FEED.format(inst=instrument, y=day.year, m=day.month - 1, d=day.day, h=hour)
    for attempt in range(tries):
        try:
            response = _session.get(url, timeout=45)
            if response.status_code == 200:
                return response.content
            if response.status_code == 404:
                return b""
        except requests.RequestException:
            pass
        time.sleep(min(2 ** attempt, 20) * (0.6 + random.random() * 0.8))
    raise RuntimeError(f"could not fetch {url}")


def _decode(raw: bytes, day: date, hour: int) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    body = lzma.decompress(raw, format=lzma.FORMAT_ALONE)
    count = len(body) // RECORD.size
    if not count:
        return pd.DataFrame()
    rows = np.frombuffer(body[: count * RECORD.size], dtype=">u4,>u4,>u4,>f4,>f4")
    millis = rows["f0"].astype("int64")
    ask = rows["f1"].astype("float64") / PRICE_SCALE
    bid = rows["f2"].astype("float64") / PRICE_SCALE
    base = pd.Timestamp(day.year, day.month, day.day, hour, tz="UTC")
    return pd.DataFrame(
        {"price": (ask + bid) / 2.0, "volume": rows["f3"].astype("float64")},
        index=base + pd.to_timedelta(millis, unit="ms"),
    )


def _trading_hours(day: date) -> list[int]:
    """UTC hours worth requesting. The week runs Sunday 21:00 to Friday 21:00."""
    weekday = day.weekday()  # Monday is 0
    if weekday == 5:  # Saturday: shut all day
        return []
    if weekday == 6:  # Sunday: only the reopen
        return list(range(20, 24))
    if weekday == 4:  # Friday: shuts at 21:00 UTC
        return list(range(0, 22))
    return list(range(24))


def cache_file(instrument: str, day: date) -> Path:
    return DATA_DIR / "dukascopy" / instrument / f"{day:%Y-%m-%d}.csv"


def day_bars(instrument: str, day: date, rule: str = "5min", workers: int = 10) -> pd.DataFrame:
    """Five-minute bars for one day, cached so a rerun costs nothing."""
    path = cache_file(instrument, day)
    if path.exists():
        bars = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("UTC")
        return bars.tz_convert(ET) if str(bars.index.tz) != ET else bars

    hours = _trading_hours(day)
    if not hours:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).to_csv(path, index_label="timestamp")
        return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        chunks = list(pool.map(lambda h: _decode(_fetch_hour(instrument, day, h), day, h), hours))
    ticks = pd.concat([c for c in chunks if not c.empty]) if any(not c.empty for c in chunks) else pd.DataFrame()

    if ticks.empty:
        bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    else:
        ticks = ticks.sort_index()
        bars = ticks["price"].resample(rule).ohlc()
        bars["volume"] = ticks["volume"].resample(rule).sum()
        bars = bars.dropna(subset=["open", "high", "low", "close"])

    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(path, index_label="timestamp")
    if bars.empty:
        return bars
    return bars.tz_convert(ET)


def prefetch(instrument: str, days: list[date], rule: str = "5min", workers: int = 32,
             chunk_days: int = 7) -> list[date]:
    """Warm the cache for many days at once, and report the days that failed.

    Fetching a day at a time leaves the pool idle between bursts, which is the
    difference between ninety seconds a day and a few. Whole weeks of hours go
    into one pool instead, chunked so memory stays bounded.
    """
    pending = [d for d in days if not cache_file(instrument, d).exists() and _trading_hours(d)]
    failed: list[date] = []

    for start in range(0, len(pending), chunk_days):
        block = pending[start : start + chunk_days]
        jobs = [(d, h) for d in block for h in _trading_hours(d)]
        results: dict[tuple[date, int], pd.DataFrame | None] = {}

        def grab(job):
            d, h = job
            try:
                return job, _decode(_fetch_hour(instrument, d, h), d, h)
            except Exception:  # noqa: BLE001 - recorded as a hole, not raised
                return job, None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for job, frame in pool.map(grab, jobs):
                results[job] = frame

        for d in block:
            chunks = [results[(d, h)] for h in _trading_hours(d)]
            if any(c is None for c in chunks):
                failed.append(d)  # a hole would fake a gap; drop the whole day
                continue
            ticks = [c for c in chunks if not c.empty]
            path = cache_file(instrument, d)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not ticks:
                pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).to_csv(
                    path, index_label="timestamp"
                )
                continue
            merged = pd.concat(ticks).sort_index()
            bars = merged["price"].resample(rule).ohlc()
            bars["volume"] = merged["volume"].resample(rule).sum()
            bars = bars.dropna(subset=["open", "high", "low", "close"])
            bars.to_csv(path, index_label="timestamp")
        print(f"  cached through {block[-1]}  ({len(failed)} incomplete)", flush=True)
    return failed


def load_range(
    instrument: str = NASDAQ_CFD,
    start: date | str = "2025-06-01",
    end: date | str = "2026-06-26",
    rule: str = "5min",
    workers: int = 32,
    progress_every: int = 10,
) -> pd.DataFrame:
    """Every cached day between the two dates, downloading what is missing.

    A day whose hours cannot all be retrieved is skipped rather than cached
    half-filled: a session with a hole in it would silently produce a fake gap
    or a missing sweep, which is worse than one fewer session.
    """
    start = pd.Timestamp(start).date() if isinstance(start, str) else start
    end = pd.Timestamp(end).date() if isinstance(end, str) else end

    every_day = []
    day = start
    while day <= end:
        every_day.append(day)
        day += timedelta(days=1)

    failed = prefetch(instrument, every_day, rule=rule, workers=workers)
    if failed:
        print(f"  incomplete, excluded: {len(failed)} day(s); first: {failed[0]}", flush=True)

    frames = []
    for day in every_day:
        if day in failed or not cache_file(instrument, day).exists():
            continue
        frames.append(day_bars(instrument, day, rule=rule))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    bars = pd.concat(frames).sort_index()
    bars = bars[~bars.index.duplicated(keep="first")]
    bars.index.name = "timestamp"
    return bars
