"""Fill and exit mechanics, checked against sequences with known outcomes."""

from __future__ import annotations

import pandas as pd
import pytest

from gapstrat.backtest import ExecutionConfig, simulate_trade
from gapstrat.data import NQ
from gapstrat.gaps import find_gaps
from gapstrat.strategy import PlannedTrade, StrategyConfig, plan_session

ET = "America/New_York"
NO_COST = ExecutionConfig(slippage_ticks=0, through_ticks=0)


def frame(rows, start="2026-09-01 09:30", freq="5min"):
    index = pd.date_range(start, periods=len(rows), freq=freq, tz=ET)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index).assign(volume=0.0)


def make_plan(bars, direction=1, entry=100.0, stop=90.0, target=120.0, **kw):
    gap = find_gaps(frame([(100, 105, 99, 104), (104, 130, 103, 128), (128, 135, 120, 133)]))[0]
    return PlannedTrade(
        day=bars.index[0].date(),
        direction=direction,
        entry_type=kw.pop("entry_type", "limit"),
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        working_from=kw.pop("working_from", bars.index[0] - pd.Timedelta(minutes=5)),
        expires_at=kw.pop("expires_at", bars.index[-1]),
        exit_at=kw.pop("exit_at", bars.index[-1]),
        gap=gap,
        **kw,
    )


def test_limit_fills_and_target_pays_target_r():
    bars = frame([(110, 112, 99, 105), (105, 125, 104, 122)])
    plan = make_plan(bars)  # buy limit 100, stop 90, target 120 -> 2R
    res = simulate_trade(plan, bars, NQ, NO_COST)
    assert res.filled and res.entry_fill == 100.0
    assert res.exit_reason == "target"
    assert res.r_multiple == pytest.approx(2.0)
    assert res.net_dollars == pytest.approx(20 * 20 - NQ.commission_round_turn)


def test_limit_needs_to_trade_through_before_filling():
    bars = frame([(110, 112, 100, 105)])  # low touches exactly 100
    plan = make_plan(bars)
    assert simulate_trade(plan, bars, NQ, ExecutionConfig(through_ticks=1)).filled is False
    assert simulate_trade(plan, bars, NQ, NO_COST).filled is True


def test_open_below_limit_fills_at_the_better_open_price():
    bars = frame([(97, 99, 95, 96), (96, 125, 95, 122)])
    res = simulate_trade(make_plan(bars), bars, NQ, NO_COST)
    assert res.entry_fill == 97.0  # filled at the open, not at the 100 limit


def test_stop_and_target_in_one_bar_resolves_as_a_loss():
    bars = frame([(110, 112, 99, 105), (105, 125, 89, 100)])
    res = simulate_trade(make_plan(bars), bars, NQ, NO_COST)
    assert res.exit_reason == "stop"
    assert res.ambiguous_bar is True
    assert res.r_multiple == pytest.approx(-1.0)


def test_entry_and_stop_in_the_same_bar_is_a_loss_not_a_skip():
    bars = frame([(110, 112, 88, 95)])
    res = simulate_trade(make_plan(bars), bars, NQ, NO_COST)
    assert res.filled is True
    assert res.exit_reason == "stop"


def test_unfilled_order_expires_flat():
    bars = frame([(110, 115, 108, 112), (112, 118, 109, 117)])
    res = simulate_trade(make_plan(bars), bars, NQ, NO_COST)
    assert res.filled is False
    assert res.net_dollars == 0.0
    assert res.r_multiple == 0.0


def test_order_stops_working_after_expiry():
    bars = frame([(110, 112, 108, 111), (111, 112, 95, 96)])
    plan = make_plan(bars, expires_at=bars.index[0])
    assert simulate_trade(plan, bars, NQ, NO_COST).filled is False


def test_open_position_is_flattened_at_the_close():
    bars = frame([(110, 112, 99, 105), (105, 112, 104, 108)])
    res = simulate_trade(make_plan(bars), bars, NQ, NO_COST)
    assert res.exit_reason == "time"
    assert res.exit_fill == 108.0
    assert res.r_multiple == pytest.approx(0.8)


def test_gapping_through_the_stop_fills_worse_than_the_stop():
    bars = frame([(110, 112, 99, 105), (85, 86, 80, 82)])
    res = simulate_trade(make_plan(bars), bars, NQ, NO_COST)
    assert res.exit_fill == 85.0  # not 90: the stop became a market order at the open
    assert res.r_multiple == pytest.approx(-1.5)


def test_target_fills_at_the_limit_price_never_better():
    # The bar runs to 140, but a resting limit at 120 gets 120.
    bars = frame([(110, 140, 99, 105)])
    res = simulate_trade(make_plan(bars), bars, NQ, NO_COST)
    assert res.exit_reason == "target"
    assert res.exit_fill == 120.0
    assert res.r_multiple == pytest.approx(2.0)


def test_entry_bar_open_is_not_usable_as_an_exit_price():
    # The bar opens at 125 (above target), dips to 99 filling the limit, and
    # closes at 105. The open came before the entry, so it cannot be an exit.
    bars = frame([(125, 126, 99, 105), (105, 106, 104, 105)])
    res = simulate_trade(make_plan(bars), bars, NQ, NO_COST)
    assert res.entry_fill == 100.0
    assert res.exit_reason == "target"
    assert res.exit_fill == 120.0  # not 125, and not 126


def test_later_bar_gapping_past_the_target_still_fills_at_the_target():
    bars = frame([(110, 112, 99, 105), (125, 126, 124, 125)])
    res = simulate_trade(make_plan(bars), bars, NQ, NO_COST)
    assert res.exit_reason == "target"
    assert res.exit_fill == 120.0


def test_excursion_stats_exclude_the_entry_bar():
    bars = frame([(110, 118, 99, 105), (105, 106, 104, 105)])
    res = simulate_trade(make_plan(bars), bars, NQ, NO_COST)
    assert res.max_favorable_r == pytest.approx(0.6)  # from bar 2's high of 106


def test_short_side_mirrors_the_long_side():
    bars = frame([(90, 101, 89, 95), (95, 96, 79, 81)])
    plan = make_plan(bars, direction=-1, entry=100.0, stop=110.0, target=80.0)
    res = simulate_trade(plan, bars, NQ, NO_COST)
    assert res.filled and res.entry_fill == 100.0
    assert res.exit_reason == "target"
    assert res.r_multiple == pytest.approx(2.0)


def test_slippage_costs_are_charged_on_stop_exits():
    bars = frame([(110, 112, 99, 105), (105, 106, 89, 91)])
    res = simulate_trade(make_plan(bars), bars, NQ, ExecutionConfig(slippage_ticks=4, through_ticks=0))
    assert res.exit_fill == pytest.approx(89.0)  # 90 stop minus 4 ticks
    assert res.r_multiple == pytest.approx(-1.1)


def test_breakeven_is_never_armed_from_the_entry_bar():
    # The entry bar runs +1.8R, but we filled on its low and cannot know the
    # high came afterwards, so the stop must still be the original one on bar 2.
    bars = frame([(110, 118, 99, 105), (105, 106, 100, 101)])
    plan = make_plan(bars)
    plan.breakeven_r = 1.0
    res = simulate_trade(plan, bars, NQ, NO_COST)
    assert res.exit_reason == "time"
    assert res.r_multiple == pytest.approx(0.1)


def test_breakeven_arms_after_a_later_bar_reaches_the_trigger():
    bars = frame([(110, 112, 99, 105), (105, 110, 104, 108), (108, 109, 95, 96)])
    plan = make_plan(bars)
    plan.breakeven_r = 1.0
    res = simulate_trade(plan, bars, NQ, NO_COST)  # bar 2 arms it, bar 3 takes it
    assert res.exit_reason == "breakeven"
    assert res.exit_time == bars.index[2]
    assert res.r_multiple == pytest.approx(0.0)


def test_stop_entry_fills_on_a_break_of_the_level():
    bars = frame([(95, 99, 94, 98), (98, 125, 97, 122)])
    plan = make_plan(bars, entry_type="stop", entry=100.0, stop=90.0, target=120.0)
    res = simulate_trade(plan, bars, NQ, ExecutionConfig(slippage_ticks=1, through_ticks=0))
    assert res.entry_fill == pytest.approx(100.25)  # one tick of slippage
    assert res.exit_reason == "target"


def test_plan_session_picks_the_first_gap_and_only_that_one():
    bars = frame(
        [
            (100, 105, 99, 104),
            (104, 130, 103, 128),
            (128, 135, 120, 133),  # first gap forms here
            (133, 136, 130, 134),
            (134, 170, 133, 168),
            (168, 175, 160, 172),  # a second, larger gap forms later
        ]
    )
    cfg = StrategyConfig(min_gap_points=1.0, min_stop_points=1.0, max_stop_points=None, stop_buffer_ticks=0)
    plan = plan_session(bars, bars.index[0].date(), cfg, NQ)
    assert plan is not None
    assert plan.direction == 1
    assert plan.gap.formed_at == bars.index[2]
    assert plan.entry_price == 112.5  # midpoint of the 105-120 gap
    assert plan.stop_price == 105.0  # far edge of the gap
    assert plan.target_price == 127.5  # 2R


def test_gaps_before_the_open_are_ignored():
    bars = frame(
        [(100, 105, 99, 104), (104, 130, 103, 128), (128, 135, 120, 133)],
        start="2026-09-01 09:15",
    )
    cfg = StrategyConfig(min_gap_points=1.0, min_stop_points=1.0, stop_buffer_ticks=0)
    assert plan_session(bars, bars.index[0].date(), cfg, NQ) is None
