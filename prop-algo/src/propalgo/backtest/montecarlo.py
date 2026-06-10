"""Bootstrap Monte Carlo: P(pass), days-to-pass, expected attempts and cost.

Resamples whole *days* of trades with replacement (preserving intraday
correlation between trades) to build synthetic 30-trading-day eval months,
then replays each through the eval rules at a given per-trade risk fraction
of the trailing drawdown. Size is micro-granular (Apex 50K allows 100 micros
= 10 minis), so wide-stop days simply get small size instead of being
untradeable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AccountRules
from ..rules.apex import EvalStatus, EvalTracker
from .engine import TradeRecord, size_micros

RISK_FRACS = [0.05, 0.08, 0.085, 0.09, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.50]


@dataclass
class MonteCarloResult:
    risk_frac: float              # per-trade risk as fraction of trailing DD
    p_pass: float                 # P(pass within `horizon_days` trading days)
    p_bust: float
    median_days_to_pass: float | None
    expected_attempts: float | None   # geometric: 1 / p_pass
    expected_cost: float | None       # fee x expected total months until a pass
    avg_final_balance: float
    avg_micros: float             # average position size actually used
    avg_months: float = 1.0       # billing months consumed per attempt (21 td/mo)

TRADING_DAYS_PER_MONTH = 21


def _trades_by_day(trades: list[TradeRecord]) -> list[list[TradeRecord]]:
    days: dict = {}
    for t in sorted(trades, key=lambda t: t.entry_time):
        days.setdefault(t.day, []).append(t)
    return list(days.values())


def simulate_attempt(day_pool: list[list[TradeRecord]], rules: AccountRules,
                     risk_frac: float, aplus_multiplier: float,
                     horizon_days: int, rng: np.random.Generator,
                     sizes_out: list | None = None,
                     policy: dict | None = None) -> tuple[EvalStatus, int, float]:
    """One synthetic eval attempt; returns (status, trading_days_used, balance).

    `policy` options:
      stop_after_loss: skip the rest of a day's signals after a losing trade.
      adaptive: 'timid' scales the risk budget with the remaining buffer above
        the trailing threshold (risk less when wounded); 'bold' scales it with
        distance left to the target (risk more when behind).
    """
    policy = policy or {}
    stop_after_loss = bool(policy.get("stop_after_loss"))
    adaptive = policy.get("adaptive")
    taper = bool(policy.get("taper"))
    tracker = EvalTracker(rules)
    days_used = 0
    for _ in range(horizon_days):
        day = day_pool[rng.integers(len(day_pool))]
        days_used += 1
        day_lost = False
        for tr in day:
            if day_lost:
                continue
            budget = risk_frac * rules.trailing_drawdown
            if adaptive == "timid":
                budget = risk_frac * max(0.0, tracker.balance - tracker.threshold)
            elif adaptive == "bold":
                remaining = max(0.0, tracker.target_balance - tracker.balance)
                budget = risk_frac * rules.trailing_drawdown * \
                    max(0.5, remaining / rules.profit_target)
            if tr.grade == "A+":
                budget *= aplus_multiplier
            if taper:
                # never risk much more than what's left to the target
                remaining = max(0.0, tracker.target_balance - tracker.balance)
                budget = min(budget, max(remaining, 0.04 * rules.trailing_drawdown))
            micros = size_micros(budget, tr.risk_per_contract, rules.max_contracts)
            if tracker.target_pending():
                micros = min(micros, 1)
            if micros == 0:
                continue
            if sizes_out is not None:
                sizes_out.append(micros)
            scale = micros / 10.0
            tracker.mark_trading_day(days_used)
            for lo, hi in zip(tr.bar_low_pnl, tr.bar_high_pnl):
                if tracker.on_equity_extremes(tracker.balance + scale * lo,
                                              tracker.balance + scale * hi) is EvalStatus.BUSTED:
                    break
            if tracker.status is EvalStatus.ACTIVE:
                tracker.on_trade_closed(scale * tr.pnl_per_contract)
                if stop_after_loss and tr.pnl_per_contract < 0:
                    day_lost = True
            if tracker.status is not EvalStatus.ACTIVE:
                return tracker.status, days_used, tracker.balance
    return tracker.status, days_used, tracker.balance


def run_montecarlo(trades: list[TradeRecord], rules: AccountRules, risk_frac: float,
                   aplus_multiplier: float = 1.5, horizon_days: int = 30,
                   n_sims: int = 10_000, seed: int = 7,
                   policy: dict | None = None) -> MonteCarloResult:
    day_pool = _trades_by_day(trades)
    if not day_pool:
        raise ValueError("no trades to bootstrap from")
    rng = np.random.default_rng(seed)
    passes, busts, days_to_pass, finals, sizes, months = 0, 0, [], [], [], []
    for _ in range(n_sims):
        status, days, bal = simulate_attempt(day_pool, rules, risk_frac,
                                             aplus_multiplier, horizon_days, rng,
                                             sizes, policy)
        finals.append(bal)
        months.append(max(1, int(np.ceil(days / TRADING_DAYS_PER_MONTH))))
        if status is EvalStatus.PASSED:
            passes += 1
            days_to_pass.append(days)
        elif status is EvalStatus.BUSTED:
            busts += 1
    p_pass = passes / n_sims
    avg_months = float(np.mean(months))
    return MonteCarloResult(
        risk_frac=risk_frac,
        p_pass=p_pass,
        p_bust=busts / n_sims,
        median_days_to_pass=float(np.median(days_to_pass)) if days_to_pass else None,
        expected_attempts=(1.0 / p_pass) if p_pass > 0 else None,
        # total billing months across however many attempts a pass takes
        expected_cost=(rules.eval_fee * avg_months / p_pass) if p_pass > 0 else None,
        avg_final_balance=float(np.mean(finals)),
        avg_micros=float(np.mean(sizes)) if sizes else 0.0,
        avg_months=avg_months,
    )


def sweep_risk(trades: list[TradeRecord], rules: AccountRules,
               aplus_multiplier: float = 1.5, horizon_days: int = 30,
               n_sims: int = 10_000,
               fracs: list[float] | None = None,
               policy: dict | None = None) -> list[MonteCarloResult]:
    if policy is None:
        policy = {"taper": True}
    return [run_montecarlo(trades, rules, f, aplus_multiplier, horizon_days,
                           n_sims, policy=policy)
            for f in (fracs or RISK_FRACS)]
