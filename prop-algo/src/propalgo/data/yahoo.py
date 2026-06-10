"""Recent true-futures 15m bars from Yahoo Finance (~60 days, ~10 min delayed).

Powers the live signal loop and the proxy-vs-futures parity check.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..config import Instrument
from .base import DataSource


class YahooSource(DataSource):
    name = "yahoo"

    def __init__(self, instruments: dict[str, Instrument], cache_dir=None):
        super().__init__(cache_dir)
        self.instruments = instruments

    def _fetch_remote(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        import yfinance as yf

        ticker = self.instruments[symbol].yahoo
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="15m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
