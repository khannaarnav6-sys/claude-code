"""Order construction: sides, levels and the guards around them."""

from __future__ import annotations

import pandas as pd
import pytest

from gapstrat.controls import _flip
from gapstrat.data import NQ
from gapstrat.gaps import Gap
from gapstrat.strategy import StrategyConfig, _build_trade

ET = "America/New_York"
TS = pd.Timestamp("2026-09-01 09:45", tz=ET)


def bullish_gap(close: float = 133.0) -> Gap:
    return Gap(
        direction=1,
        formed_at=TS,
        formed_index=2,
        low=105.0,
        high=120.0,
        impulse_high=135.0,
        impulse_low=99.0,
        close=close,
    )


def cfg(**kw) -> StrategyConfig:
    base = {"min_gap_points": 1.0, "min_stop_points": 1.0, "max_stop_points": None, "stop_buffer_ticks": 0}
    return StrategyConfig(**{**base, **kw})


def test_with_bias_places_a_buy_limit_inside_the_gap():
    trade = _build_trade(bullish_gap(), TS.date(), cfg(), NQ, ET)
    assert trade.direction == 1
    assert trade.entry_type == "limit"
    assert trade.entry_price == 112.5  # gap midpoint
    assert trade.stop_price == 105.0  # far edge of the gap
    assert trade.entry_price < trade.reference_price  # resting below the market


def test_stop_entry_breaks_the_pattern_high():
    trade = _build_trade(bullish_gap(), TS.date(), cfg(entry_type="stop"), NQ, ET)
    assert trade.entry_type == "stop"
    assert trade.entry_price == 135.25  # one tick above the impulse high
    assert trade.entry_price > trade.reference_price


def test_fade_uses_a_stop_into_the_gap_not_a_limit():
    # Price is above a bullish gap; fading it means selling as price breaks back
    # down into the gap, which is a sell stop at the near edge.
    trade = _build_trade(bullish_gap(), TS.date(), cfg(displacement="fade"), NQ, ET)
    assert trade.direction == -1
    assert trade.entry_type == "stop"
    assert trade.entry_price == 120.0  # proximal edge
    assert trade.entry_price < trade.reference_price  # sell stop below the market
    assert trade.stop_price == 135.0  # above the impulse high


def test_order_resting_on_the_wrong_side_of_the_market_is_rejected():
    # A bullish gap whose bar closed below the entry would make the buy limit
    # marketable on arrival; the fill model would flatter it, so refuse it.
    marketable = bullish_gap(close=110.0)
    assert _build_trade(marketable, TS.date(), cfg(), NQ, ET) is None


def test_stops_that_are_too_tight_or_too_wide_are_skipped():
    gap = bullish_gap()
    assert _build_trade(gap, TS.date(), cfg(min_stop_points=100.0), NQ, ET) is None
    assert _build_trade(gap, TS.date(), cfg(max_stop_points=1.0), NQ, ET) is None


def test_target_sits_at_the_configured_r_multiple():
    trade = _build_trade(bullish_gap(), TS.date(), cfg(target_r=3.0), NQ, ET)
    risk = trade.entry_price - trade.stop_price
    assert trade.target_price == pytest.approx(trade.entry_price + 3 * risk)


def test_flip_mirrors_the_order_to_the_other_side_of_the_market():
    trade = _build_trade(bullish_gap(), TS.date(), cfg(), NQ, ET)
    flipped = _flip(trade, -1, target_r=2.0)
    assert flipped.direction == -1
    # entry was 20.5 below the market; the mirror is 20.5 above it
    assert flipped.entry_price == pytest.approx(2 * 133.0 - 112.5)
    assert flipped.entry_price > flipped.reference_price  # a real sell limit
    assert flipped.risk_points == pytest.approx(trade.risk_points)
    assert flipped.stop_price > flipped.entry_price
    assert flipped.target_price < flipped.entry_price


def test_flip_to_the_same_direction_changes_nothing():
    trade = _build_trade(bullish_gap(), TS.date(), cfg(), NQ, ET)
    assert _flip(trade, 1, target_r=2.0) is trade
