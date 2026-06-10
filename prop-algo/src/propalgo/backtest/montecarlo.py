"""Bootstrap Monte Carlo: P(pass), days-to-pass, expected attempts and cost.

Resamples whole *days* of trades with replacement (preserving intraday
correlation between trades) to build synthetic 30-trading-day eval months,
then replays each through the eval rules at a given contract size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AccountRules
from ..rules.apex import EvalStatus, EvalTracker
from .engine import TradeRecord


@dataclass
class MonteCarloResult:
    contracts: int
    p_pass: float                 # P(pass within `horizon_days` trading days)
    p_bust: float
    median_days_to_pass: float | None
    expected_attempts: float | None   # geometric: 1 / p_pass
    expected_cost: float | None       # attempts * eval fee
    avg_final_balance: float


def _trades_by_day(trades: list[TradeRecord]) -> list[list[TradeRecord]]:
    days: dict = {}
    for t in sorted(trades, key=lambda t: t.entry_time):
        days.setdefault(t.day, []).append(t)
    return list(days.values())


def simulate_attempt(day_pool: list[list[TradeRecord]], rules: AccountRules,
                     contracts: int, aplus_multiplier: float,
                     horizon_days: int, rng: np.random.Generator) -> tuple[EvalStatus, int, float]:
    """One synthetic eval attempt; returns (status, trading_days_used, balance)."""
    tracker = EvalTracker(rules)
    days_used = 0
    for _ in range(horizon_days):
        day = day_pool[rng.integers(len(day_pool))]
        days_used += 1
        for tr in day:
            qty = contracts
            if tr.grade == "A+":
                qty = int(min(rules.max_contracts, round(contracts * aplus_multiplier)))
            qty = max(1, min(qty, rules.max_contracts))
            if tracker.target_pending():
                qty = 1
            tracker.mark_trading_day(days_used)
            for lo, hi in zip(tr.bar_low_pnl, tr.bar_high_pnl):
                if tracker.on_equity_extremes(tracker.balance + qty * lo,
                                              tracker.balance + qty * hi) is EvalStatus.BUSTED:
                    break
            if tracker.status is EvalStatus.ACTIVE:
                tracker.on_trade_closed(qty * tr.pnl_per_contract)
            if tracker.status is not EvalStatus.ACTIVE:
                return tracker.status, days_used, tracker.balance
    return tracker.status, days_used, tracker.balance


def run_montecarlo(trades: list[TradeRecord], rules: AccountRules, contracts: int,
                   aplus_multiplier: float = 1.5, horizon_days: int = 30,
                   n_sims: int = 10_000, seed: int = 7) -> MonteCarloResult:
    day_pool = _trades_by_day(trades)
    if not day_pool:
        raise ValueError("no trades to bootstrap from")
    rng = np.random.default_rng(seed)
    passes, busts, days_to_pass, finals = 0, 0, [], []
    for _ in range(n_sims):
        status, days, bal = simulate_attempt(day_pool, rules, contracts,
                                             aplus_multiplier, horizon_days, rng)
        finals.append(bal)
        if status is EvalStatus.PASSED:
            passes += 1
            days_to_pass.append(days)
        elif status is EvalStatus.BUSTED:
            busts += 1
    p_pass = passes / n_sims
    return MonteCarloResult(
        contracts=contracts,
        p_pass=p_pass,
        p_bust=busts / n_sims,
        median_days_to_pass=float(np.median(days_to_pass)) if days_to_pass else None,
        expected_attempts=(1.0 / p_pass) if p_pass > 0 else None,
        expected_cost=(rules.eval_fee / p_pass) if p_pass > 0 else None,
        avg_final_balance=float(np.mean(finals)),
    )


def sweep_contracts(trades: list[TradeRecord], rules: AccountRules,
                    aplus_multiplier: float = 1.5, horizon_days: int = 30,
                    n_sims: int = 10_000) -> list[MonteCarloResult]:
    return [run_montecarlo(trades, rules, c, aplus_multiplier, horizon_days, n_sims)
            for c in range(1, rules.max_contracts + 1)]
