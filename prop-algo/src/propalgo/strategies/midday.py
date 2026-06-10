"""Midday range breakout — the afternoon counterpart of ORB.

Range = 11:30-13:30 ET (lunch consolidation). A close beyond the range
between 13:30 and 15:00 enters in the breakout direction with the range's
other side as the stop (capped at 0.75x daily ATR), 2R target, EOD exit.
Afternoon breakouts from a tight lunch range are trend-day continuation —
the same regime momentum and ORB exploit, but a second, independent entry
window that raises trades/day (low-risk eval grinds need frequency).
Same short handling as ORB: `shorts: trade|skip|veto`.
"""
from __future__ import annotations

import pandas as pd

from .base import Signal, Strategy

ET = "America/New_York"


class MiddayBreakout(Strategy):
    name = "midday"

    def on_session(self, symbol: str, session: pd.DataFrame, daily_atr: float) -> list[tuple[int, Signal]]:
        target_r = float(self.params.get("target_r", 2.0))
        max_range_atr = float(self.params.get("max_range_atr", 0.5))
        stop_cap_atr = float(self.params.get("stop_cap_atr", 0.75))
        shorts = self.params.get("shorts", "trade")

        et = session.index.tz_convert(ET)
        in_range = (et.time >= pd.Timestamp("11:30").time()) & \
                   (et.time < pd.Timestamp("13:30").time())
        entry_window = (et.time >= pd.Timestamp("13:30").time()) & \
                       (et.time < pd.Timestamp("15:00").time())
        if daily_atr <= 0 or not in_range.any() or not entry_window.any():
            return []
        rng_high = session["high"][in_range].max()
        rng_low = session["low"][in_range].min()
        rng = rng_high - rng_low
        # only tight consolidations break out cleanly
        if rng <= 0 or rng > max_range_atr * daily_atr:
            return []

        for i in range(len(session) - 1):
            if not entry_window[i]:
                continue
            close = session["close"].iloc[i]
            side = 0
            if close > rng_high:
                side = 1
            elif close < rng_low:
                side = -1
            if side == 0 or (side < 0 and shorts == "skip"):
                continue
            stop_dist = min(rng, stop_cap_atr * daily_atr)
            stop = close - side * stop_dist
            target = close + side * target_r * stop_dist
            grade = "A+" if rng <= 0.5 * max_range_atr * daily_atr else "A"
            return [(i, Signal(
                strategy=self.name, symbol=symbol, side=side,
                entry_ref=close, stop=stop, target=target, grade=grade,
                note=f"midday range {rng_low:.2f}-{rng_high:.2f} break "
                     f"{'up' if side > 0 else 'down'}",
                action="veto" if (side < 0 and shorts == "veto") else "trade",
            ))]
        return []
