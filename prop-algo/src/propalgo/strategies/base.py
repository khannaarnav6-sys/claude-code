"""Strategy interface.

Strategies see 15m bars one at a time (UTC index, ET session columns added
by the engine) and may emit a Signal on bar close. Entries happen at the
next bar's open. `grade` lets sizing go bigger on A+ setups.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Signal:
    strategy: str
    symbol: str
    side: int               # +1 long, -1 short
    entry_ref: float        # reference price at signal (next bar open is the fill)
    stop: float
    target: float
    grade: str = "A"        # "A+" sized up by sizing.aplus_multiplier
    note: str = ""

    @property
    def risk_points(self) -> float:
        return abs(self.entry_ref - self.stop)


class Strategy(ABC):
    name: str = "base"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @abstractmethod
    def on_session(self, symbol: str, session: pd.DataFrame, daily_atr: float) -> list[tuple[int, Signal]]:
        """Scan one RTH session of 15m bars; return [(bar_index, Signal)].

        bar_index is the position within `session` of the bar on whose CLOSE
        the signal fires. The engine fills at session.iloc[bar_index + 1].open.
        One signal per session per strategy keeps trades high-conviction.
        """
