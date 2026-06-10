"""Opening Range Breakout.

Range = first `or_minutes` of RTH (9:30 ET). A close beyond the range in the
following bars signals continuation; stop is the opposite side of the range
(capped at 1x daily ATR), target is `target_r` multiples of risk. Breakout
bars with above-average volume grade A+. Skips days whose opening range is
already wider than `max_or_atr` x ATR — the "big win" days are orderly
breakouts, not opening chop.
"""
from __future__ import annotations

import pandas as pd

from .base import Signal, Strategy


class OpeningRangeBreakout(Strategy):
    name = "orb"

    def on_session(self, symbol: str, session: pd.DataFrame, daily_atr: float) -> list[tuple[int, Signal]]:
        or_bars = max(1, int(self.params.get("or_minutes", 30)) // 15)
        target_r = float(self.params.get("target_r", 2.0))
        max_or_atr = float(self.params.get("max_or_atr", 1.5))
        min_vol_ratio = float(self.params.get("min_volume_ratio", 1.0))

        if len(session) <= or_bars + 1:
            return []
        or_high = session["high"].iloc[:or_bars].max()
        or_low = session["low"].iloc[:or_bars].min()
        or_range = or_high - or_low
        if or_range <= 0 or (daily_atr > 0 and or_range > max_or_atr * daily_atr):
            return []

        avg_vol = session["volume"].rolling(20, min_periods=3).mean()
        # leave one bar after the signal for the entry fill
        for i in range(or_bars, len(session) - 1):
            close = session["close"].iloc[i]
            side = 0
            if close > or_high:
                side = 1
            elif close < or_low:
                side = -1
            if side == 0:
                continue
            stop_dist = min(or_range, daily_atr) if daily_atr > 0 else or_range
            stop = close - side * stop_dist
            target = close + side * target_r * stop_dist
            vol_ok = avg_vol.iloc[i] > 0 and session["volume"].iloc[i] >= min_vol_ratio * avg_vol.iloc[i]
            return [(i, Signal(
                strategy=self.name, symbol=symbol, side=side,
                entry_ref=close, stop=stop, target=target,
                grade="A+" if vol_ok else "A",
                note=f"OR {or_low:.2f}-{or_high:.2f} break {'up' if side > 0 else 'down'}",
            ))]
        return []
