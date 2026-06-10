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


def _trade(day, pnl, lows, highs, risk=200.0):
    ts = pd.Timestamp(f"{day} 15:00", tz="UTC")
    return TradeRecord(symbol="NQ", strategy="t", grade="A", side=1, day=day,
                       entry_time=ts, exit_time=ts, entry=0, exit=0,
                       exit_reason="target", pnl_per_contract=pnl,
                       risk_per_contract=risk, bar_low_pnl=lows, bar_high_pnl=highs)


def test_size_micros_granularity():
    from propalgo.backtest.engine import size_micros
    # $375 budget, $200/mini risk -> $20/micro -> 18 micros (1.8 minis)
    assert size_micros(375.0, 200.0, max_contracts=10) == 18
    assert size_micros(375.0, 200.0, max_contracts=1) == 10   # capped
    assert size_micros(10.0, 200.0, max_contracts=10) == 0    # too wide
    assert size_micros(375.0, 0.0, max_contracts=10) == 0     # degenerate


def test_veto_signal_blocks_window_but_is_not_traded():
    from propalgo.backtest.engine import generate_trades
    from propalgo.strategies.base import Strategy

    class Fake(Strategy):
        def __init__(self, name, emissions):
            super().__init__({})
            self.name, self.emissions = name, emissions

        def on_session(self, symbol, session, daily_atr):
            return self.emissions

    # one RTH session of flat bars (26 bars from 9:30 ET)
    idx = pd.date_range("2026-01-05 14:30", periods=26, freq="15min", tz="UTC")
    bars = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5,
                         "close": 100.0, "volume": 1.0}, index=idx)
    veto = Signal(strategy="v", symbol="NQ", side=-1, entry_ref=100,
                  stop=110, target=90, action="veto")        # never exits -> blocks to EOD
    late = Signal(strategy="w", symbol="NQ", side=1, entry_ref=100,
                  stop=90, target=110)
    trades = generate_trades({"NQ": bars},
                             [Fake("v", [(2, veto)]), Fake("w", [(5, late)])],
                             {"NQ": NQ}, NO_COSTS)
    assert trades == []  # veto not traded, and it blocked the later signal

    # without the veto, the later signal trades normally
    trades = generate_trades({"NQ": bars}, [Fake("w", [(5, late)])],
                             {"NQ": NQ}, NO_COSTS)
    assert len(trades) == 1 and trades[0].strategy == "w"


def test_eval_sequence_pass_and_bust():
    rules = AccountRules(min_trading_days=1)
    # risk 50% of DD = $1,250 budget; $200/mini risk -> 62 micros -> 6.2 minis
    # +$800/contract -> +$4,960 -> pass in one trade
    win = _trade(date(2026, 1, 5), 800.0, [-100.0, 200.0], [100.0, 800.0])
    attempts = run_eval_sequence([win], rules, risk_frac=0.5)
    assert len(attempts) == 1 and attempts[0].passed

    # excursion 3x the stated risk (gap through stop): 6.2 * -600 = -$3,720 -> bust
    lose = _trade(date(2026, 1, 6), -600.0, [-600.0], [50.0])
    attempts = run_eval_sequence([lose], rules, risk_frac=0.5)
    assert len(attempts) == 1 and not attempts[0].passed


def test_eval_sequence_restarts_after_bust():
    rules = AccountRules(min_trading_days=1)
    lose = _trade(date(2026, 1, 5), -600.0, [-600.0], [0.0])
    win = _trade(date(2026, 1, 6), 800.0, [-100.0], [800.0])
    attempts = run_eval_sequence([lose, win], rules, risk_frac=0.5)
    assert [a.passed for a in attempts] == [False, True]


def test_eval_sequence_risk_sizing_survives_normal_stop():
    """At 15% risk the same stop-out is a survivable -$375, not a bust."""
    rules = AccountRules(min_trading_days=1)
    lose = _trade(date(2026, 1, 5), -200.0, [-200.0], [0.0])
    attempts = run_eval_sequence([lose], rules, risk_frac=0.15)
    assert attempts == []  # eval still active, neither passed nor busted
