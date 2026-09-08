"""Turning the first unmitigated gap of the New York session into an order.

The rule set, in words:

  1. Start watching at the New York open (09:30 ET). Nothing before it counts.
  2. Take the first three-bar imbalance that forms after the open and is large
     enough to matter. It is unmitigated by construction the moment it forms.
  3. Trade in the direction of the displacement that created it: a bullish gap
     is a long, a bearish gap is a short. (`displacement="fade"` flips this, to test
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

from . import bias as bias_module
from .bias import SessionLevels, session_levels
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
    displacement: str = "with"  # trade "with" the displacement, or "fade" it

    # --- session bias ---
    bias_method: str = "none"  # see gapstrat.bias.METHODS
    require_bias: bool = True  # skip gaps that argue against the session bias

    # --- entry ---
    entry_type: str = "limit"  # "limit" back into the gap, or "stop" through the pattern
    entry_style: str = "mid"  # limit level: "proximal" | "mid" | "distal"

    # --- risk ---
    # "gap_far" (beyond the gap), "pattern" (beyond the 3 bars), or "swing"
    # (beyond the session extreme so far -- the structural stop)
    stop_style: str = "gap_far"
    # How many signal bars back a "swing" stop looks for its structural extreme.
    # The whole session is usually far too wide: by late morning the session low
    # can be a hundred points away, which makes a 2R target an implausible move.
    swing_lookback: int = 6
    stop_buffer_ticks: int = 4
    min_stop_points: float = 5.0
    max_stop_points: float | None = 100.0
    target_r: float = 2.0
    breakeven_at_r: float | None = None  # move stop to entry once this much is banked

    def validate(self) -> None:
        if self.displacement not in {"with", "fade"}:
            raise ValueError(f"bad displacement {self.displacement!r}")
        if self.entry_type not in {"limit", "stop"}:
            raise ValueError(f"bad entry_type {self.entry_type!r}")
        if self.entry_style not in {"proximal", "mid", "distal"}:
            raise ValueError(f"bad entry_style {self.entry_style!r}")
        if self.stop_style not in {"gap_far", "pattern", "swing"}:
            raise ValueError(f"bad stop_style {self.stop_style!r}")
        if self.bias_method not in bias_module.METHODS:
            raise ValueError(f"bad bias_method {self.bias_method!r}")
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
    gap: Gap | None = None
    breakeven_r: float | None = None
    reference_price: float = 0.0  # market price when the order was placed
    session_bias: int = 0  # the bias in force when the order was placed

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
    trades = all_session_trades(signal_bars, day, config, contract)
    return trades[0] if trades else None  # the first workable gap, and only that one


def _previous_session(signal_bars: pd.DataFrame, day):
    """The most recent calendar date in the data before `day`."""
    earlier = [d for d in {ts.date() for ts in signal_bars.index} if d < day]
    return max(earlier) if earlier else None


def workable_trades(
    window: pd.DataFrame,
    day,
    config: StrategyConfig,
    contract: Contract,
    tz,
    levels: SessionLevels | None = None,
) -> list[PlannedTrade]:
    """Every tradable gap in the window, in formation order.

    `plan_session` takes the first of these. The control tests sample from the
    whole list to ask whether being first is what matters.
    """
    trades = []
    for gap in find_gaps(window, min_size=config.min_gap_points):
        if config.max_gap_points is not None and gap.size > config.max_gap_points:
            continue  # a gap this wide means the stop is wider than the day's range
        # Only the bars up to and including the one that completed the gap were
        # on the screen when the order would have been placed.
        seen = window.iloc[: gap.formed_index + 1]
        trade = _build_trade(gap, day, config, contract, tz, seen, levels)
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
    levels = (
        session_levels(signal_bars, day, _previous_session(signal_bars, day))
        if config.bias_method != "none"
        else None
    )
    return workable_trades(window, day, config, contract, tz, levels)


def _build_trade(
    gap: Gap,
    day,
    config: StrategyConfig,
    contract: Contract,
    tz,
    seen: pd.DataFrame | None = None,
    levels: SessionLevels | None = None,
) -> PlannedTrade | None:
    direction = gap.direction if config.displacement == "with" else -gap.direction
    buffer = config.stop_buffer_ticks * contract.tick_size
    reference = gap.close  # last traded price when the order goes in

    session_bias = 0
    if config.bias_method != "none" and seen is not None and levels is not None:
        session_bias = bias_module.determine(config.bias_method, seen, levels)
        if config.require_bias and session_bias != direction:
            # Either the session has not picked a side yet, or this gap argues
            # against the side it picked. Both are passes.
            return None

    entry_type = config.entry_type
    if config.displacement == "fade":
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

    if config.displacement == "fade":
        # Invalidation is a new extreme in the displacement's direction.
        stop = (
            gap.impulse_low - buffer if direction > 0 else gap.impulse_high + buffer
        )
    elif config.stop_style == "swing":
        # Structural stop: beyond the recent extreme, which on a
        # stop-run-and-reverse is the far side of the wick that swept the low.
        if seen is None or seen.empty:
            return None
        recent = seen.iloc[-config.swing_lookback :] if config.swing_lookback > 0 else seen
        stop = (
            float(recent["low"].min()) - buffer
            if direction > 0
            else float(recent["high"].max()) + buffer
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
        session_bias=session_bias,
    )


def describe(config: StrategyConfig) -> str:
    """One-line summary of a config, for labelling sweep results."""
    entry_type = "stop" if config.displacement == "fade" else config.entry_type
    if config.displacement == "fade":
        shape = "into-gap"
    elif entry_type == "limit":
        shape = config.entry_style
    else:
        shape = "pattern-break"
    parts = [f"{config.displacement}/{entry_type}", shape, f"stop={config.stop_style}", f"{config.target_r:g}R"]
    if config.bias_method != "none":
        parts.append(f"bias={config.bias_method}")
    if config.breakeven_at_r:
        parts.append(f"be@{config.breakeven_at_r:g}R")
    return " ".join(parts)


def variants(base: StrategyConfig, **overrides) -> StrategyConfig:
    return replace(base, **overrides)
