from datetime import date

import pandas as pd

from propalgo.backtest.engine import TradeRecord
from propalgo.backtest.montecarlo import run_montecarlo
from propalgo.config import AccountRules
from propalgo.sizing import recommend_risk


def _trade(day_n, pnl, risk=200.0):
    d = date(2026, 1, 1 + day_n)
    ts = pd.Timestamp(f"{d} 15:00", tz="UTC")
    return TradeRecord(symbol="NQ", strategy="t", grade="A", side=1, day=d,
                       entry_time=ts, exit_time=ts, entry=0, exit=0,
                       exit_reason="x", pnl_per_contract=pnl, risk_per_contract=risk,
                       bar_low_pnl=[min(pnl, -50.0)], bar_high_pnl=[max(pnl, 50.0)])


def test_all_winners_pass_certain():
    trades = [_trade(i, 300.0) for i in range(10)]
    rules = AccountRules(min_trading_days=1)
    res = run_montecarlo(trades, rules, risk_frac=0.3, n_sims=200, horizon_days=30)
    assert res.p_pass == 1.0
    assert res.expected_attempts == 1.0


def test_all_losers_never_pass():
    trades = [_trade(i, -300.0) for i in range(10)]
    rules = AccountRules(min_trading_days=1)
    res = run_montecarlo(trades, rules, risk_frac=0.3, n_sims=200, horizon_days=30)
    assert res.p_pass == 0.0
    assert res.expected_cost is None


def test_recommend_picks_pass_probability_maximizer():
    trades = [_trade(i, 400.0 if i % 4 else -200.0) for i in range(20)]
    rules = AccountRules(min_trading_days=1)
    best, results = recommend_risk(trades, rules, n_sims=300)
    best_res = next(r for r in results if r.risk_frac == best)
    # recommended risk level is never beaten on (rounded) pass probability
    assert all(round(best_res.p_pass, 3) >= round(r.p_pass, 3) for r in results)


def test_more_risk_passes_faster_when_it_passes():
    trades = [_trade(i, 400.0 if i % 4 else -200.0) for i in range(20)]
    rules = AccountRules(min_trading_days=1)
    _, results = recommend_risk(trades, rules, n_sims=300)
    days = {r.risk_frac: r.median_days_to_pass for r in results
            if r.median_days_to_pass is not None}
    assert days[min(days)] >= days[max(days)]
