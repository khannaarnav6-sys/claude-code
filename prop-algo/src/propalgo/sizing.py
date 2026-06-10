"""Pick the contract size that maximizes P(pass) via the Monte Carlo sweep."""
from __future__ import annotations

from .backtest.engine import TradeRecord
from .backtest.montecarlo import MonteCarloResult, sweep_contracts
from .config import AccountRules


def recommend_contracts(trades: list[TradeRecord], rules: AccountRules,
                        aplus_multiplier: float = 1.5, horizon_days: int = 30,
                        n_sims: int = 5_000) -> tuple[int, list[MonteCarloResult]]:
    results = sweep_contracts(trades, rules, aplus_multiplier, horizon_days, n_sims)
    # maximize P(pass); among (near-)ties, prefer the faster median pass
    best = max(results, key=lambda r: (round(r.p_pass, 3),
                                       -(r.median_days_to_pass or horizon_days)))
    return best.contracts, results
