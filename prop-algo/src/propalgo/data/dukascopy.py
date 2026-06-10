"""Multi-year 15m history from Dukascopy's free datafeed (index-CFD futures proxies)."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..config import Instrument
from .base import DataSource


class DukascopySource(DataSource):
    name = "dukascopy"

    def __init__(self, instruments: dict[str, Instrument], cache_dir=None):
        super().__init__(cache_dir)
        self.instruments = instruments

    def _fetch_remote(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        import dukascopy_python

        ducas_id = self.instruments[symbol].dukascopy
        df = dukascopy_python.fetch(
            ducas_id,
            dukascopy_python.INTERVAL_MIN_15,
            dukascopy_python.OFFER_SIDE_BID,
            start.replace(tzinfo=None),
            end.replace(tzinfo=None),
        )
        return df if df is not None else pd.DataFrame()
