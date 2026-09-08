"""Turning the first unmitigated gap of the New York session into an order.

The rule set, in words:

  1. Start watching at the New York open (09:30 ET). Nothing before it counts.
  2. Take the first three-bar imbalance that forms after the open and is large
     enough to matter. It is unmitigated by construction the moment it forms.
  3. Trade in the direction of the displacement that created it: a bullish gap
     is a long, a bearish gap is a short. (`bias="fade"` flips this, to test
     whether the direction carries any information at all.)
  4. Work a resting order -- a limit back inside the gap, or a stop through the
     high/low of the pattern -- until it fills or the cutoff passes.
  5. Protective stop beyond the far edge of the gap, or beyond the extreme of
     the three-bar pattern. Target a fixed multiple of that risk.
  6. Flat by the close. One trade per session.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from .data import Contract
from .gaps import Gap, find_gaps


@dataclass(frozen=True)
class StrategyConfig:
    # --- session windows (ET wall clock) ---
    session_open: str = "09:30"
    signal_cutoff: str = "11:00"  # last time a gap may form and still be traded
    entry_expiry: str = "12:00"  # unfilled orders are cancelled here
    exit_time: str = "15:55"  # flatten any open position

    # --- gap selection ---
    min_gap_points: float = 5.0
    max_gap_points: float | None = 120.0
    bias: str = "with"  # "with" the displacement, or "fade" it

    # --- entry ---
    entry_type: str = "limit"  # "limit" back into the gap, or "stop" through the pattern
    entry_style: str = "mid"  # limit level: "proximal" | "mid" | "distal"

    # --- risk ---
    stop_style: str = "gap_far"  # "gap_far" (beyond the gap) or "pattern" (beyond the 3 bars)
    stop_buffer_ticks: int = 4
    min_stop_points: float = 5.0
    max_stop_points: float | None = 100.0
    target_r: float = 2.0
    breakeven_at_r: float | None = None  # move stop to entry once this much is banked

    def validate(self) -> None:
        if self.bias not in {"with", "fade"}:
            raise ValueError(f"bad bias {self.bias!r}")
        if self.entry_type not in {"limit", "stop"}:
            raise ValueError(f"bad entry_type {self.entry_type!r}")
        if self.entry_style not in {"proximal", "mid", "distal"}:
            raise ValueError(f"bad entry_style {self.entry_style!r}")
        if self.stop_style not in {"gap_far", "pattern"}:
            raise ValueError(f"bad stop_style {self.stop_style!r}")
        if self.target_r <= 0:
            raise ValueError("target_r must be positive")


@dataclass
class PlannedTrade:
    """A resting order plus the exit levels it will carry once filled."""

    day: object
    direction: int  # +1 long, -1 short
    entry_type: str
    entry_price: float
    stop_price: float
    target_price: float
    working_from: pd.Timestamp  # first bar the order can fill on
    expires_at: pd.Timestamp
    exit_at: pd.Timestamp
    gap: Gap
    breakeven_r: float | None = None
    reference_price: float = 0.0  # market price when the order was placed

    @property
    def risk_points(self) -> float:
        return abs(self.entry_price - self.stop_price)


def _at(day, clock: str, tz) -> pd.Timestamp:
    hour, minute = (int(part) for part in clock.split(":"))
    return pd.Timestamp(
        year=day.year, month=day.month, day=day.day, hour=hour, minute=minute, tz=tz
    )


def plan_session(
    signal_bars: pd.DataFrame,
    day,
    config: StrategyConfig,
    contract: Contract,
) -> PlannedTrade | None:
    """Find the day's setup, or return None if the session offers none.

    `signal_bars` is the whole dataset; this slices out the session itself so
    gap detection never sees a bar from a neighbouring day.
    """
    config.validate()
    tz = signal_bars.index.tz
    open_ts = _at(day, config.session_open, tz)
    cutoff_ts = _at(day, config.signal_cutoff, tz)

    window = signal_bars[(signal_bars.index >= open_ts) & (signal_bars.index <= cutoff_ts)]
    if len(window) < 3:
        return None

    workable = workable_trades(window, day, config, contract, tz)
    return workable[0] if workable else None  # the first one, and only that one


def workable_trades(
    window: pd.DataFrame,
    day,
    config: StrategyConfig,
    contract: Contract,
    tz,
) -> list[PlannedTrade]:
    """Every tradable gap in the window, in formation order.

    `plan_session` takes the first of these. The control tests sample from the
    whole list to ask whether being first is what matters.
    """
    trades = []
    for gap in find_gaps(window, min_size=config.min_gap_points):
        if config.max_gap_points is not None and gap.size > config.max_gap_points:
            continue  # a gap this wide means the stop is wider than the day's range
        trade = _build_trade(gap, day, config, contract, tz)
        if trade is not None:
            trades.append(trade)
    return trades


def all_session_trades(
    signal_bars: pd.DataFrame,
    day,
    config: StrategyConfig,
    contract: Contract,
) -> list[PlannedTrade]:
    """`workable_trades` for one session, slicing the window itself."""
    config.validate()
    tz = signal_bars.index.tz
    window = signal_bars[
        (signal_bars.index >= _at(day, config.session_open, tz))
        & (signal_bars.index <= _at(day, config.signal_cutoff, tz))
    ]
    if len(window) < 3:
        return []
    return workable_trades(window, day, config, contract, tz)


def _build_trade(
    gap: Gap,
    day,
    config: StrategyConfig,
    contract: Contract,
    tz,
) -> PlannedTrade | None:
    direction = gap.direction if config.bias == "with" else -gap.direction
    buffer = config.stop_buffer_ticks * contract.tick_size
    reference = gap.close  # last traded price when the order goes in

    entry_type = config.entry_type
    if config.bias == "fade":
        # Fading means betting the gap fills through. Price sits on the far side
        # of the gap, so the trigger is price breaking back *into* it -- a stop
        # order at the near edge, never a limit.
        entry_type = "stop"
        entry = gap.proximal
    elif config.entry_type == "limit":
        entry = gap.entry_price(config.entry_style)
    else:  # stop entry: break of the pattern extreme in the traded direction
        entry = (
            gap.impulse_high + contract.tick_size
            if direction > 0
            else gap.impulse_low - contract.tick_size
        )

    # An order has to rest on the correct side of the market, or it is really a
    # market order wearing a limit's clothes and the fill model would flatter it.
    if entry_type == "limit" and (
        (direction > 0 and entry > reference) or (direction < 0 and entry < reference)
    ):
        return None
    if entry_type == "stop" and (
        (direction > 0 and entry < reference) or (direction < 0 and entry > reference)
    ):
        return None

    if config.bias == "fade":
        # Invalidation is a new extreme in the displacement's direction.
        stop = (
            gap.impulse_low - buffer if direction > 0 else gap.impulse_high + buffer
        )
    elif config.stop_style == "gap_far":
        stop = gap.low - buffer if direction > 0 else gap.high + buffer
    else:
        stop = (
            gap.impulse_low - buffer if direction > 0 else gap.impulse_high + buffer
        )

    risk = abs(entry - stop)
    if risk < config.min_stop_points:
        return None
    if config.max_stop_points is not None and risk > config.max_stop_points:
        return None
    if direction > 0 and stop >= entry:
        return None
    if direction < 0 and stop <= entry:
        return None

    target = entry + direction * config.target_r * risk

    return PlannedTrade(
        day=day,
        direction=direction,
        entry_type=entry_type,
        entry_price=round(entry / contract.tick_size) * contract.tick_size,
        stop_price=round(stop / contract.tick_size) * contract.tick_size,
        target_price=round(target / contract.tick_size) * contract.tick_size,
        working_from=gap.formed_at,
        expires_at=_at(day, config.entry_expiry, tz),
        exit_at=_at(day, config.exit_time, tz),
        gap=gap,
        breakeven_r=config.breakeven_at_r,
        reference_price=reference,
    )


def describe(config: StrategyConfig) -> str:
    """One-line summary of a config, for labelling sweep results."""
    entry_type = "stop" if config.bias == "fade" else config.entry_type
    if config.bias == "fade":
        shape = "into-gap"
    elif entry_type == "limit":
        shape = config.entry_style
    else:
        shape = "pattern-break"
    parts = [f"{config.bias}/{entry_type}", shape, f"stop={config.stop_style}", f"{config.target_r:g}R"]
    if config.breakeven_at_r:
        parts.append(f"be@{config.breakeven_at_r:g}R")
    return " ".join(parts)


def variants(base: StrategyConfig, **overrides) -> StrategyConfig:
    return replace(base, **overrides)
