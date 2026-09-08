"""Fair value gap (imbalance) detection and mitigation tracking.

A fair value gap is the unfilled space left by a three-bar displacement: bar 2
moves far enough that bar 1 and bar 3 never overlap. The untouched range
between them is the gap.

    bullish (up displacement)        bearish (down displacement)
        low[i] > high[i-2]               high[i] < low[i-2]
        zone = high[i-2] .. low[i]       zone = high[i] .. low[i-2]

A gap is "unmitigated" while price has not traded back into that range. This
module only finds and tracks gaps; what to do about one lives in strategy.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Gap:
    """One three-bar imbalance and its mitigation history."""

    direction: int  # +1 bullish, -1 bearish
    formed_at: pd.Timestamp  # open time of the third bar
    formed_index: int  # position of the third bar in the signal frame
    low: float  # bottom of the untouched range
    high: float  # top of the untouched range
    impulse_high: float  # highest high of the three bars
    impulse_low: float  # lowest low of the three bars
    close: float = 0.0  # close of the third bar: the price when the order is placed
    touched_at: pd.Timestamp | None = field(default=None)
    filled_at: pd.Timestamp | None = field(default=None)

    @property
    def size(self) -> float:
        return self.high - self.low

    @property
    def midpoint(self) -> float:
        """Consequent encroachment: the 50% level of the gap."""
        return (self.high + self.low) / 2.0

    @property
    def proximal(self) -> float:
        """Edge price reaches first on a retracement back toward the gap."""
        return self.high if self.direction > 0 else self.low

    @property
    def distal(self) -> float:
        """Far edge: price through this level means the gap is fully filled."""
        return self.low if self.direction > 0 else self.high

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def entry_price(self, style: str) -> float:
        if style == "proximal":
            return self.proximal
        if style == "mid":
            return self.midpoint
        if style == "distal":
            return self.distal
        raise ValueError(f"unknown entry style {style!r}")


def find_gaps(bars: pd.DataFrame, min_size: float = 0.0) -> list[Gap]:
    """All three-bar imbalances in `bars`, oldest first.

    `min_size` filters out imbalances too small to be worth trading; on NQ a
    one-tick gap is noise, not displacement.
    """
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    closes = bars["close"].to_numpy()
    index = bars.index

    gaps: list[Gap] = []
    for i in range(2, len(bars)):
        if lows[i] > highs[i - 2]:
            low, high, direction = highs[i - 2], lows[i], 1
        elif highs[i] < lows[i - 2]:
            low, high, direction = highs[i], lows[i - 2], -1
        else:
            continue
        if high - low < min_size:
            continue
        gaps.append(
            Gap(
                direction=direction,
                formed_at=index[i],
                formed_index=i,
                low=low,
                high=high,
                impulse_high=float(max(highs[i - 2 : i + 1])),
                impulse_low=float(min(lows[i - 2 : i + 1])),
                close=float(closes[i]),
            )
        )
    return gaps


def mark_mitigation(gap: Gap, bars: pd.DataFrame) -> Gap:
    """Record when `gap` was first touched and when it was fully filled.

    Scanning starts on the bar after the gap forms; the three forming bars
    cannot mitigate their own imbalance.
    """
    forward = bars.iloc[gap.formed_index + 1 :]
    for ts, bar in forward.iterrows():
        if gap.touched_at is None and bar["low"] <= gap.high and bar["high"] >= gap.low:
            gap.touched_at = ts
        if gap.direction > 0 and bar["low"] <= gap.low:
            gap.filled_at = ts
            break
        if gap.direction < 0 and bar["high"] >= gap.high:
            gap.filled_at = ts
            break
    return gap


def unmitigated_at(gaps: list[Gap], when: pd.Timestamp) -> list[Gap]:
    """Gaps formed before `when` that price has not yet traded back into."""
    return [
        g
        for g in gaps
        if g.formed_at < when and (g.touched_at is None or g.touched_at >= when)
    ]
