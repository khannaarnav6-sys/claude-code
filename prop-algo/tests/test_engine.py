from datetime import date

import numpy as np
import pandas as pd
import pytest

from propalgo.backtest.engine import (TradeRecord, rth_sessions, run_eval_sequence,
                                      simulate_trade)
from propalgo.config import AccountRules, Costs, Instrument
from propalgo.strategies.base import Signal

NQ = Instrument(symbol="NQ", dukascopy="E_NQ-100", yahoo="NQ=F",
                point_value=20.0, tick_size=0.25)
NO_COSTS = Costs(commission_rt=0.0, slippage_ticks=0)


def make_session(rows, start="2026-01-05 14:30"):
    """rows: list of (open, high, low, close) -> 15m UTC session frame."""
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    df.index.name = "timestamp"
    return df


def sig(side, stop, target, entry_ref=100.0):
    return Signal(strategy="t", symbol="NQ", side=side,
                  entry_ref=entry_ref, stop=stop, target=target)


def test_long_target_hit():
    s = make_session([(100, 101, 99, 100.5),
                      (100.5, 102, 100, 101.5),
                      (101.5, 106, 101, 105.5)])
    tr = simulate_trade(s, 0, sig(+1, stop=95, target=105), NQ, NO_COSTS)
    assert tr.exit_reason == "target"
    assert tr.pnl_per_contract == pytest.approx((105 - 100) * 20)


def test_long_stop_hit():
    s = make_session([(100, 101, 99, 100.5),
                      (100.5, 101, 94, 95.0)])
    tr = simulate_trade(s, 0, sig(+1, stop=96, target=110), NQ, NO_COSTS)
    assert tr.exit_reason == "stop"
    assert tr.pnl_per_contract == pytest.approx((96 - 100) * 20)


def test_stop_and_target_same_bar_assumes_stop():
    s = make_session([(100, 100.5, 99.5, 100),
                      (100, 120, 90, 110)])      # bar hits both
    tr = simulate_trade(s, 0, sig(+1, stop=96, target=105), NQ, NO_COSTS)
    assert tr.exit_reason == "stop"


def test_eod_flatten():
    s = make_session([(100, 101, 99, 100.5),
                      (100.5, 101.5, 100, 101.0)])
    tr = simulate_trade(s, 0, sig(+1, stop=90, target=120), NQ, NO_COSTS)
    assert tr.exit_reason == "eod"
    assert tr.pnl_per_contract == pytest.approx((101 - 100) * 20)


def test_short_target():
    s = make_session([(100, 101, 99, 100),
                      (100, 100.5, 94, 95)])
    tr = simulate_trade(s, 0, sig(-1, stop=104, target=95), NQ, NO_COSTS)
    assert tr.exit_reason == "target"
    assert tr.pnl_per_contract == pytest.approx((100 - 95) * 20)


def test_costs_reduce_pnl():
    s = make_session([(100, 101, 99, 100.5),
                      (100.5, 106, 100, 105.5)])
    costs = Costs(commission_rt=4.0, slippage_ticks=1)
    tr = simulate_trade(s, 0, sig(+1, stop=95, target=105), NQ, costs)
    # entry 100 + 0.25 slip, exit 105 - 0.25 slip, minus $4 commission
    assert tr.pnl_per_contract == pytest.approx((104.75 - 100.25) * 20 - 4.0)


def test_excursions_track_unrealized_extremes():
    s = make_session([(100, 101, 99, 100.5),
                      (100.5, 103, 98, 102)])
    tr = simulate_trade(s, 0, sig(+1, stop=90, target=120), NQ, NO_COSTS)
    assert tr.bar_low_pnl[1] == pytest.approx((98 - 100) * 20)
    assert tr.bar_high_pnl[1] == pytest.approx((103 - 100) * 20)


def test_rth_sessions_filters_overnight():
    idx = pd.date_range("2026-01-05 00:00", periods=96, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 1.0}, index=idx)
    sessions = rth_sessions(df)
    for day, sess in sessions.items():
        et = sess.index.tz_convert("America/New_York")
        assert et.time.min() >= pd.Timestamp("09:30").time()
        assert et.time.max() < pd.Timestamp("16:00").time()


def _trade(day, pnl, lows, highs):
    ts = pd.Timestamp(f"{day} 15:00", tz="UTC")
    return TradeRecord(symbol="NQ", strategy="t", grade="A", side=1, day=day,
                       entry_time=ts, exit_time=ts, entry=0, exit=0,
                       exit_reason="target", pnl_per_contract=pnl,
                       bar_low_pnl=lows, bar_high_pnl=highs)


def test_eval_sequence_pass_and_bust():
    rules = AccountRules(min_trading_days=1)
    # +$800/contract winners: at 4 contracts that's +$3,200 -> pass in one trade
    win = _trade(date(2026, 1, 5), 800.0, [-100.0, 200.0], [100.0, 800.0])
    attempts = run_eval_sequence([win], rules, base_contracts=4)
    assert len(attempts) == 1 and attempts[0].passed

    # -$700/contract loser at 4 contracts = -$2,800 -> bust mid-trade
    lose = _trade(date(2026, 1, 6), -700.0, [-700.0], [50.0])
    attempts = run_eval_sequence([lose], rules, base_contracts=4)
    assert len(attempts) == 1 and not attempts[0].passed


def test_eval_sequence_restarts_after_bust():
    rules = AccountRules(min_trading_days=1)
    lose = _trade(date(2026, 1, 5), -700.0, [-700.0], [0.0])
    win = _trade(date(2026, 1, 6), 800.0, [-100.0], [800.0])
    attempts = run_eval_sequence([lose, win], rules, base_contracts=4)
    assert [a.passed for a in attempts] == [False, True]
