"""Trend-day continuation.

A meaningful overnight gap (vs prior RTH close) that holds: price stays on
the gap side of session VWAP for `vwap_hold_bars` consecutive bars after the
open. Enters in the gap direction targeting a trend day. Gap >= 2x the
threshold grades A+.
"""
from __future__ import annotations

import pandas as pd

from .base import Signal, Strategy


class TrendDayMomentum(Strategy):
    name = "momentum"

    def on_session(self, symbol: str, session: pd.DataFrame, daily_atr: float) -> list[tuple[int, Signal]]:
        gap_atr = float(self.params.get("gap_atr", 0.30))
        hold_bars = int(self.params.get("vwap_hold_bars", 3))
        target_r = float(self.params.get("target_r", 2.5))

        prev_close = session.attrs.get("prev_rth_close")
        if prev_close is None or daily_atr <= 0 or len(session) <= hold_bars + 1:
            return []
        gap = session["open"].iloc[0] - prev_close
        if abs(gap) < gap_atr * daily_atr:
            return []
        side = 1 if gap > 0 else -1

        typical = (session["high"] + session["low"] + session["close"]) / 3.0
        vol = session["volume"].clip(lower=1e-9)
        vwap = (typical * vol).cumsum() / vol.cumsum()

        held = 0
        for i in range(len(session) - 1):
            on_side = (session["close"].iloc[i] - vwap.iloc[i]) * side > 0
            held = held + 1 if on_side else 0
            if held >= hold_bars:
                close = session["close"].iloc[i]
                stop_dist = max(abs(close - vwap.iloc[i]), 0.25 * daily_atr)
                stop = close - side * stop_dist
                target = close + side * target_r * stop_dist
                grade = "A+" if abs(gap) >= 2 * gap_atr * daily_atr else "A"
                return [(i, Signal(
                    strategy=self.name, symbol=symbol, side=side,
                    entry_ref=close, stop=stop, target=target, grade=grade,
                    note=f"gap {gap:+.2f} ({abs(gap) / daily_atr:.2f} ATR) holding VWAP",
                ))]
        return []
