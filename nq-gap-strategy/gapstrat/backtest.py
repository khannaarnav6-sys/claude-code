"""Bar-by-bar execution simulator.

OHLC bars do not record the path price took inside a bar, so every ambiguous
case is resolved against the strategy:

  * a limit order fills only once price trades through it by `through_ticks`,
    never merely by touching it;
  * a profit target is a resting limit, so it fills at its own price and never
    at a better one, however far past it the market runs;
  * on the bar that fills the entry, the bar's open is unusable -- it happened
    before the fill -- so exits on that bar can only be the exact stop or
    target level, and the bar contributes no excursion statistics;
  * if the filling bar also contains the protective stop, the trade is treated
    as filled and then stopped on that same bar;
  * if one bar contains both the stop and the target, the stop is taken;
  * stop-loss exits, stop entries and time exits pay slippage, limit fills
    do not.

Bars that contained both stop and target are counted and reported, so the
weight resting on that last assumption stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Contract
from .strategy import PlannedTrade, StrategyConfig, plan_session


@dataclass
class ExecutionConfig:
    slippage_ticks: int = 1  # applied to stop entries, stop exits and time exits
    through_ticks: int = 1  # ticks a limit must be beaten by before it is a fill
    contracts: int = 1


@dataclass
class TradeResult:
    day: object
    direction: int
    entry_type: str
    planned_entry: float
    stop_price: float
    target_price: float
    risk_points: float
    gap_size: float
    gap_formed_at: pd.Timestamp | None
    filled: bool
    entry_time: pd.Timestamp | None = None
    entry_fill: float | None = None
    exit_time: pd.Timestamp | None = None
    exit_fill: float | None = None
    exit_reason: str | None = None
    points: float = 0.0
    gross_dollars: float = 0.0
    net_dollars: float = 0.0
    r_multiple: float = 0.0
    max_favorable_r: float = 0.0
    max_adverse_r: float = 0.0
    ambiguous_bar: bool = False

    def as_row(self) -> dict:
        return {
            "day": str(self.day),
            "side": "long" if self.direction > 0 else "short",
            "entry_type": self.entry_type,
            "gap_formed": self.gap_formed_at.strftime("%H:%M") if self.gap_formed_at is not None else "",
            "gap_size": round(self.gap_size, 2),
            "planned_entry": self.planned_entry,
            "stop": self.stop_price,
            "target": self.target_price,
            "risk_pts": round(self.risk_points, 2),
            "filled": self.filled,
            "entry_time": self.entry_time.strftime("%H:%M") if self.entry_time is not None else "",
            "entry_fill": self.entry_fill,
            "exit_time": self.exit_time.strftime("%H:%M") if self.exit_time is not None else "",
            "exit_fill": self.exit_fill,
            "exit_reason": self.exit_reason or "",
            "points": round(self.points, 2),
            "r": round(self.r_multiple, 3),
            "net_dollars": round(self.net_dollars, 2),
            "mfe_r": round(self.max_favorable_r, 2),
            "mae_r": round(self.max_adverse_r, 2),
            "ambiguous": self.ambiguous_bar,
        }


@dataclass
class BacktestResult:
    trades: list[TradeResult] = field(default_factory=list)
    sessions_scanned: int = 0
    sessions_with_setup: int = 0

    @property
    def filled(self) -> list[TradeResult]:
        return [t for t in self.trades if t.filled]

    def frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.as_row() for t in self.trades])


def _fill_limit(bo, bh, bl, price, direction, tick, through):
    """Fill price for a resting limit, or None if the bar never reaches it."""
    if direction > 0:
        if bo <= price:
            return min(bo, price)  # already below us at the open: filled there
        return price if bl <= price - through * tick else None
    if bo >= price:
        return max(bo, price)
    return price if bh >= price + through * tick else None


def _fill_stop(bo, bh, bl, price, direction, tick, slippage):
    """Fill price for a stop order, including slippage, or None if untouched."""
    slip = slippage * tick
    if direction > 0:
        if bo >= price:
            return bo + slip
        return price + slip if bh >= price else None
    if bo <= price:
        return bo - slip
    return price - slip if bl <= price else None


def simulate_trade(
    plan: PlannedTrade,
    exec_bars: pd.DataFrame,
    contract: Contract,
    execution: ExecutionConfig,
) -> TradeResult:
    """Run one planned trade forward through the execution bars."""
    tick = contract.tick_size
    slip = execution.slippage_ticks * tick
    through = execution.through_ticks * tick

    result = TradeResult(
        day=plan.day,
        direction=plan.direction,
        entry_type=plan.entry_type,
        planned_entry=plan.entry_price,
        stop_price=plan.stop_price,
        target_price=plan.target_price,
        risk_points=plan.risk_points,
        gap_size=plan.gap.size if plan.gap is not None else 0.0,
        gap_formed_at=plan.gap.formed_at if plan.gap is not None else None,
        filled=False,
    )

    # The signal bar closes before the order exists, so start on the next bar.
    window = exec_bars[(exec_bars.index > plan.working_from) & (exec_bars.index <= plan.exit_at)]
    if window.empty:
        return result

    index = window.index
    opens = window["open"].to_numpy(dtype=float)
    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    closes = window["close"].to_numpy(dtype=float)

    direction = plan.direction
    risk = plan.risk_points
    target = plan.target_price
    stop = plan.stop_price
    entry_fill: float | None = None
    entry_bar_pos = -1
    breakeven_armed = False

    for k in range(len(index)):
        ts = index[k]
        bo, bh, bl = opens[k], highs[k], lows[k]

        if entry_fill is None:
            if ts > plan.expires_at:
                return result  # order cancelled unfilled
            if plan.entry_type == "limit":
                entry_fill = _fill_limit(bo, bh, bl, plan.entry_price, direction, tick, execution.through_ticks)
            else:
                entry_fill = _fill_stop(bo, bh, bl, plan.entry_price, direction, tick, execution.slippage_ticks)
            if entry_fill is None:
                continue
            result.filled = True
            result.entry_time = ts
            result.entry_fill = entry_fill
            entry_bar_pos = k
            # Fall through: this same bar may already resolve the trade.

        # On the entry bar the open preceded the fill, so it is neither a valid
        # exit price nor a legitimate excursion reading.
        entry_bar = k == entry_bar_pos
        favorable = 0.0
        if not entry_bar:
            if direction > 0:
                favorable, adverse = bh - entry_fill, entry_fill - bl
            else:
                favorable, adverse = entry_fill - bl, bh - entry_fill
            result.max_favorable_r = max(result.max_favorable_r, favorable / risk)
            result.max_adverse_r = max(result.max_adverse_r, adverse / risk)

        if direction > 0:
            hit_stop = bl <= stop
            hit_target = bh >= target + through or (not entry_bar and bo >= target)
        else:
            hit_stop = bh >= stop
            hit_target = bl <= target - through or (not entry_bar and bo <= target)

        if hit_stop and hit_target:
            result.ambiguous_bar = True

        if hit_stop:
            if entry_bar:
                fill = stop - slip if direction > 0 else stop + slip
            elif direction > 0:
                fill = (min(bo, stop) if bo <= stop else stop) - slip
            else:
                fill = (max(bo, stop) if bo >= stop else stop) + slip
            reason = "breakeven" if breakeven_armed and stop == entry_fill else "stop"
            return _close(result, ts, fill, reason, contract, execution, risk)

        if hit_target:
            # A resting limit fills at its own price, never better.
            return _close(result, ts, target, "target", contract, execution, risk)

        if (
            plan.breakeven_r is not None
            and not breakeven_armed
            and not entry_bar  # the runup on the entry bar may predate the fill
            and favorable >= plan.breakeven_r * risk
        ):
            # Armed on this bar, live from the next: within a bar we cannot know
            # whether the runup came before or after the pullback.
            breakeven_armed = True
            stop = entry_fill

    if entry_fill is None:
        return result  # worked to the end of the session without filling

    # Still open: flatten on the last bar's close, paying the spread to get out.
    exit_fill = closes[-1] - direction * slip
    return _close(result, index[-1], exit_fill, "time", contract, execution, risk)


def _close(
    result: TradeResult,
    ts: pd.Timestamp,
    fill: float,
    reason: str,
    contract: Contract,
    execution: ExecutionConfig,
    risk: float,
) -> TradeResult:
    result.exit_time = ts
    result.exit_fill = round(fill, 2)
    result.exit_reason = reason
    result.points = (fill - result.entry_fill) * result.direction
    result.gross_dollars = result.points * contract.point_value * execution.contracts
    result.net_dollars = result.gross_dollars - contract.commission_round_turn * execution.contracts
    result.r_multiple = result.points / risk if risk else 0.0
    return result


def rth_sessions(bars: pd.DataFrame, min_bars: int = 60) -> list:
    """Dates with a full regular-hours session, which is what we trade."""
    rth = bars.between_time("09:30", "16:00", inclusive="left")
    counts = pd.Series(1, index=[ts.date() for ts in rth.index]).groupby(level=0).sum()
    return sorted(counts[counts >= min_bars].index)


def run(
    signal_bars: pd.DataFrame,
    exec_bars: pd.DataFrame,
    config: StrategyConfig,
    contract: Contract,
    execution: ExecutionConfig | None = None,
    days: list | None = None,
) -> BacktestResult:
    """Backtest one configuration across every session in the data."""
    execution = execution or ExecutionConfig()
    if days is None:
        days = rth_sessions(signal_bars)

    out = BacktestResult()
    for day in days:
        out.sessions_scanned += 1
        plan = plan_session(signal_bars, day, config, contract)
        if plan is None:
            continue
        out.sessions_with_setup += 1
        out.trades.append(simulate_trade(plan, exec_bars, contract, execution))
    return out


def plans_for(
    signal_bars: pd.DataFrame,
    config: StrategyConfig,
    contract: Contract,
    days: list | None = None,
) -> list[PlannedTrade]:
    """Every session's planned trade, without simulating any of them."""
    days = days if days is not None else rth_sessions(signal_bars)
    plans = [plan_session(signal_bars, day, config, contract) for day in days]
    return [p for p in plans if p is not None]


def run_plans(
    plans: list[PlannedTrade],
    exec_bars: pd.DataFrame,
    contract: Contract,
    execution: ExecutionConfig | None = None,
    sessions: int = 0,
) -> BacktestResult:
    """Simulate a list of already-built plans (used by the control tests)."""
    execution = execution or ExecutionConfig()
    out = BacktestResult(sessions_scanned=sessions or len(plans), sessions_with_setup=len(plans))
    out.trades = [simulate_trade(p, exec_bars, contract, execution) for p in plans]
    return out


def r_multiples(result: BacktestResult) -> np.ndarray:
    return np.array([t.r_multiple for t in result.filled], dtype=float)
