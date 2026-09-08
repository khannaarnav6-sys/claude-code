"""Null hypotheses for the strategy to beat.

A backtest that only reports its own equity curve cannot tell you whether the
rules did anything. Each control here keeps most of the strategy intact and
destroys exactly one claim it makes, then re-runs the whole thing a few hundred
times to build a distribution of outcomes under that null.

  random_direction -- same gaps, same levels, same risk, coin-flip direction.
      Kills the claim "trade in the direction of the displacement".

  random_gap -- trade a randomly chosen gap from the session instead of the
      first one. Kills the claim "the first gap of the day is the one".

If the real strategy's expectancy sits comfortably inside a control's
distribution, the rule it tests is decoration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .backtest import BacktestResult, ExecutionConfig, r_multiples, run_plans
from .data import Contract
from .strategy import PlannedTrade, StrategyConfig, all_session_trades


@dataclass
class ControlResult:
    name: str
    draws: int
    expectancies: np.ndarray
    actual: float

    @property
    def mean(self) -> float:
        return float(self.expectancies.mean())

    @property
    def percentile_of_actual(self) -> float:
        """Share of control runs the real strategy beat, as a percentage."""
        return float((self.expectancies < self.actual).mean() * 100)

    @property
    def p_value(self) -> float:
        """One-sided: how often the control matched or beat the real result."""
        return float((self.expectancies >= self.actual).mean())

    def summary(self) -> str:
        lo, hi = np.percentile(self.expectancies, [5, 95])
        return (
            f"{self.name:<18} control mean {self.mean:+.3f}R  "
            f"90% band [{lo:+.3f}, {hi:+.3f}]  "
            f"actual {self.actual:+.3f}R  beats {self.percentile_of_actual:.0f}% of runs  "
            f"p={self.p_value:.3f}"
        )


def _flip(plan: PlannedTrade, direction: int, target_r: float) -> PlannedTrade:
    """Mirror a setup to the other side of the market, keeping its geometry.

    Flipping the sign alone would be cheating: a buy limit resting below the
    market becomes a sell limit below the market, which is marketable on
    arrival and fills at a price the real setup never had. Reflecting entry,
    stop and target around the price at order time keeps the order the same
    distance away, on the correct side, with the same risk.
    """
    if direction == plan.direction:
        return plan
    risk = plan.risk_points
    entry = 2 * plan.reference_price - plan.entry_price
    return replace(
        plan,
        direction=direction,
        entry_price=entry,
        stop_price=entry - direction * risk,
        target_price=entry + direction * target_r * risk,
    )


def random_direction(
    plans: list[PlannedTrade],
    exec_bars: pd.DataFrame,
    contract: Contract,
    execution: ExecutionConfig,
    config: StrategyConfig,
    draws: int = 500,
    seed: int = 11,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for d in range(draws):
        sides = rng.choice([-1, 1], size=len(plans))
        flipped = [_flip(p, int(s), config.target_r) for p, s in zip(plans, sides)]
        r = r_multiples(run_plans(flipped, exec_bars, contract, execution))
        out[d] = r.mean() if r.size else 0.0
    return out


def random_gap(
    signal_bars: pd.DataFrame,
    days: list,
    exec_bars: pd.DataFrame,
    config: StrategyConfig,
    contract: Contract,
    execution: ExecutionConfig,
    draws: int = 500,
    seed: int = 12,
) -> np.ndarray:
    """Trade a random gap from each session rather than the first."""
    per_day = [all_session_trades(signal_bars, day, config, contract) for day in days]
    per_day = [candidates for candidates in per_day if candidates]
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for d in range(draws):
        picks = [candidates[rng.integers(len(candidates))] for candidates in per_day]
        r = r_multiples(run_plans(picks, exec_bars, contract, execution))
        out[d] = r.mean() if r.size else 0.0
    return out


def run_controls(
    actual: BacktestResult,
    plans: list[PlannedTrade],
    signal_bars: pd.DataFrame,
    exec_bars: pd.DataFrame,
    days: list,
    config: StrategyConfig,
    contract: Contract,
    execution: ExecutionConfig,
    draws: int = 500,
) -> list[ControlResult]:
    r = r_multiples(actual)
    observed = float(r.mean()) if r.size else 0.0
    return [
        ControlResult(
            "random direction",
            draws,
            random_direction(plans, exec_bars, contract, execution, config, draws),
            observed,
        ),
        ControlResult(
            "random gap",
            draws,
            random_gap(signal_bars, days, exec_bars, config, contract, execution, draws),
            observed,
        ),
    ]
