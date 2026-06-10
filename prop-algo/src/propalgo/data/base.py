"""DataSource interface and a shared parquet cache with incremental append.

All sources return a DataFrame with a UTC DatetimeIndex named 'timestamp'
and columns: open, high, low, close, volume.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR

COLUMNS = ["open", "high", "low", "close", "volume"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df = df[[c for c in COLUMNS if c in df.columns]]
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = 0.0
    df = df[COLUMNS].astype("float64")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df.index = idx
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


class DataSource(ABC):
    """Fetches bars from a remote source, caching to parquet."""

    name: str = "base"

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = Path(cache_dir or DATA_DIR) / self.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _fetch_remote(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch raw 15m bars for a logical symbol (e.g. 'NQ')."""

    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}_15m.parquet"

    def load_cached(self, symbol: str) -> pd.DataFrame:
        path = self._cache_path(symbol)
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))

    def fetch(self, symbol: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        """Return bars in [start, end], pulling only what the cache is missing."""
        end = end or datetime.now(timezone.utc)
        start = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        end = end if end.tzinfo else end.replace(tzinfo=timezone.utc)

        cached = self.load_cached(symbol)
        fetch_from = start
        if not cached.empty and cached.index[0] <= start:
            fetch_from = cached.index[-1]  # incremental append only
        if cached.empty or fetch_from < end:
            try:
                fresh = normalize(self._fetch_remote(symbol, fetch_from, end))
            except Exception as exc:  # network failures fall back to cache
                if cached.empty:
                    raise
                print(f"[{self.name}] fetch failed for {symbol} ({exc}); using cache only")
                fresh = pd.DataFrame()
            if not fresh.empty:
                cached = normalize(pd.concat([cached, fresh])) if not cached.empty else fresh
                cached.to_parquet(self._cache_path(symbol))
        return cached.loc[(cached.index >= start) & (cached.index <= end)]
