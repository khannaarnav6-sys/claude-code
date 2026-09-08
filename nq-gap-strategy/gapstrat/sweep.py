"""The stop-run-and-reverse setup: sweep the liquidity, reclaim, buy the retest.

This is a different animal from the gap scanner in `strategy.py`, and the
difference is geometry rather than taste.

The gap scanner enters *after* a displacement, on a retracement into the gap
the displacement left behind. Pair that with a stop below the low the move
started from and the trade is badly shaped: the stop is as far away as ever
while most of the move it was measuring has already happened.

This setup anchors both ends to the same event. Price runs a known low, fails
to hold below it, and reclaims it; the entry is a limit back at that reclaimed
level and the stop sits just under the wick that swept it. Entry and stop are
both at the origin of the move, so the risk is small relative to the room
above it -- which is what makes a 2R target a plausible ask rather than a
hopeful one.

    ─────────────────────  reference low (overnight or prior day)
         │  ← wick sweeps under it, closes back above
         ▼
    stop sits below the wick;  entry rests at the reclaimed level
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .bias import SessionLevels, session_levels
from .data import Contract
from .gaps import find_gaps
from .strategy import PlannedTrade, _at, _previous_session

REFERENCES = ("overnight", "prior_day", "opening_range")
ANCHORS = ("level", "sweep_mid", "sweep_close")


@dataclass(frozen=True)
class SweepConfig:
    # --- session windows (ET wall clock) ---
    session_open: str = "09:30"
    signal_cutoff: str = "11:00"  # last time a sweep may be confirmed
    entry_expiry: str = "12:00"
    exit_time: str = "15:55"

    # --- what counts as a sweep ---
    reference: str = "overnight"  # which liquidity pool gets run
    opening_range_bars: int = 3  # used when reference is "opening_range"
    min_penetration: float = 1.0  # points price must trade beyond the level
    confirm_within: int = 4  # bars allowed for the reclaim to be confirmed
    require_displacement: bool = True  # a fair value gap must form on the reclaim

    # --- order placement ---
    entry_anchor: str = "level"  # "level" | "sweep_mid" | "sweep_close"
    stop_buffer_ticks: int = 4
    min_stop_points: float = 5.0
    max_stop_points: float | None = 80.0
    target_r: float = 2.0
    breakeven_at_r: float | None = None

    def validate(self) -> None:
        if self.reference not in REFERENCES:
            raise ValueError(f"bad reference {self.reference!r}")
        if self.entry_anchor not in ANCHORS:
            raise ValueError(f"bad entry_anchor {self.entry_anchor!r}")
        if self.target_r <= 0:
            raise ValueError("target_r must be positive")
        if self.confirm_within < 1:
            raise ValueError("confirm_within must be at least 1")


def _reference_levels(
    config: SweepConfig, levels: SessionLevels, window: pd.DataFrame
) -> tuple[float | None, float | None]:
    """The high and low this session is hunting, or (None, None) if unknown."""
    if config.reference == "overnight":
        return levels.overnight_high, levels.overnight_low
    if config.reference == "prior_day":
        return levels.prior_high, levels.prior_low
    n = config.opening_range_bars
    if len(window) <= n:
        return None, None
    opening = window.iloc[:n]
    return float(opening["high"].max()), float(opening["low"].min())


def plan_sweep_session(
    signal_bars: pd.DataFrame,
    day,
    config: SweepConfig,
    contract: Contract,
) -> PlannedTrade | None:
    """The session's first confirmed sweep-and-reclaim, as a resting order."""
    config.validate()
    tz = signal_bars.index.tz
    open_ts = _at(day, config.session_open, tz)
    cutoff_ts = _at(day, config.signal_cutoff, tz)
    window = signal_bars[(signal_bars.index >= open_ts) & (signal_bars.index <= cutoff_ts)]
    if len(window) < 3:
        return None

    levels = session_levels(signal_bars, day, _previous_session(signal_bars, day))
    ref_high, ref_low = _reference_levels(config, levels, window)
    if ref_high is None and ref_low is None:
        return None

    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    closes = window["close"].to_numpy(dtype=float)
    first = config.opening_range_bars if config.reference == "opening_range" else 0

    for i in range(first, len(window)):
        for direction, level in ((1, ref_low), (-1, ref_high)):
            if level is None:
                continue
            # A sweep is a wick through the level that does not hold: price
            # trades beyond it and closes back on the original side.
            if direction > 0:
                swept = lows[i] <= level - config.min_penetration and closes[i] > level
            else:
                swept = highs[i] >= level + config.min_penetration and closes[i] < level
            if not swept:
                continue

            trade = _confirm_and_build(
                window, i, direction, level, day, config, contract, tz
            )
            if trade is not None:
                return trade  # the session's first confirmed sweep, and only that
    return None


def _confirm_and_build(
    window: pd.DataFrame,
    sweep_i: int,
    direction: int,
    level: float,
    day,
    config: SweepConfig,
    contract: Contract,
    tz,
) -> PlannedTrade | None:
    """Look for the reclaim that turns a sweep into a trade."""
    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    closes = window["close"].to_numpy(dtype=float)

    sweep_high, sweep_low = highs[sweep_i], lows[sweep_i]
    last = min(sweep_i + config.confirm_within, len(window) - 1)

    for j in range(sweep_i + 1, last + 1):
        # Confirmation: a close beyond the sweep bar's own extreme, which says
        # the reversal has taken control rather than merely paused.
        confirmed = closes[j] > sweep_high if direction > 0 else closes[j] < sweep_low
        if not confirmed:
            continue
        if config.require_displacement:
            # The reclaim should be violent enough to leave an imbalance behind.
            span = window.iloc[sweep_i : j + 1]
            gaps = [g for g in find_gaps(span) if g.direction == direction]
            if not gaps:
                continue

        buffer = config.stop_buffer_ticks * contract.tick_size
        if config.entry_anchor == "level":
            entry = level
        elif config.entry_anchor == "sweep_mid":
            entry = (sweep_high + sweep_low) / 2
        else:
            entry = closes[sweep_i]

        stop = sweep_low - buffer if direction > 0 else sweep_high + buffer
        reference = closes[j]  # market price when the order goes in

        # The entry is a limit behind the market: price has to come back to it.
        if direction > 0 and not (stop < entry <= reference):
            return None
        if direction < 0 and not (reference <= entry < stop):
            return None

        risk = abs(entry - stop)
        if risk < config.min_stop_points:
            return None
        if config.max_stop_points is not None and risk > config.max_stop_points:
            return None

        tick = contract.tick_size
        return PlannedTrade(
            day=day,
            direction=direction,
            entry_type="limit",
            entry_price=round(entry / tick) * tick,
            stop_price=round(stop / tick) * tick,
            target_price=round((entry + direction * config.target_r * risk) / tick) * tick,
            working_from=window.index[j],
            expires_at=_at(day, config.entry_expiry, tz),
            exit_at=_at(day, config.exit_time, tz),
            gap=None,
            breakeven_r=config.breakeven_at_r,
            reference_price=reference,
            session_bias=direction,
        )
    return None


def describe(config: SweepConfig) -> str:
    parts = [
        f"sweep/{config.reference}",
        f"entry={config.entry_anchor}",
        f"{config.target_r:g}R",
        f"confirm<={config.confirm_within}",
    ]
    if not config.require_displacement:
        parts.append("no-displacement")
    return " ".join(parts)


def run_sweep(
    signal_bars: pd.DataFrame,
    exec_bars: pd.DataFrame,
    config: SweepConfig,
    contract: Contract,
    execution=None,
    days: list | None = None,
):
    """Backtest the sweep setup across every session (mirrors backtest.run)."""
    from .backtest import BacktestResult, ExecutionConfig, rth_sessions, simulate_trade

    execution = execution or ExecutionConfig()
    days = days if days is not None else rth_sessions(signal_bars)

    out = BacktestResult()
    for day in days:
        out.sessions_scanned += 1
        plan = plan_sweep_session(signal_bars, day, config, contract)
        if plan is None:
            continue
        out.sessions_with_setup += 1
        out.trades.append(simulate_trade(plan, exec_bars, contract, execution))
    return out
