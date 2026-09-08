"""Threshold scaling across instruments priced decades apart."""

from __future__ import annotations

import pandas as pd
import pytest

from gapstrat.crossmarket import scale_to_instrument
from gapstrat.data import CONTRACTS, GAP_FRACTION, median_session_range
from gapstrat.strategy import StrategyConfig

ET = "America/New_York"


def test_scaling_reproduces_the_original_nq_thresholds():
    # The fractions were calibrated so NQ's own median range gives back the
    # absolute 5-point minimum gap the strategy was written with.
    scaled = scale_to_instrument(StrategyConfig(), session_range=375.2)
    assert scaled.min_gap_points == pytest.approx(5.0, abs=0.05)
    assert scaled.min_stop_points == pytest.approx(5.0, abs=0.05)
    assert scaled.max_stop_points == pytest.approx(100.0, abs=1.0)


def test_a_cheaper_instrument_gets_proportionally_smaller_thresholds():
    # RTY ranges about 31 points a session against NQ's 375: a 5-point gap
    # filter there would reject nearly everything.
    rty = scale_to_instrument(StrategyConfig(), session_range=30.9)
    nq = scale_to_instrument(StrategyConfig(), session_range=375.2)
    assert rty.min_gap_points < 0.5
    assert rty.min_gap_points / nq.min_gap_points == pytest.approx(30.9 / 375.2, rel=1e-3)


def test_scaling_drops_the_absolute_gap_cap():
    # max_gap_points is an absolute NQ number with no scaled equivalent, so it
    # must not silently filter another instrument.
    assert scale_to_instrument(StrategyConfig(), 58.0).max_gap_points is None


def test_scaling_leaves_the_rules_themselves_alone():
    base = StrategyConfig(target_r=3.0, entry_style="proximal", bias_method="prior_day")
    scaled = scale_to_instrument(base, 58.0)
    assert (scaled.target_r, scaled.entry_style, scaled.bias_method) == (3.0, "proximal", "prior_day")


def test_median_session_range_uses_regular_hours_only():
    index = pd.date_range("2026-09-01 04:00", periods=180, freq="5min", tz=ET)
    frame = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 0.0}, index=index
    )
    # An overnight spike outside 09:30-15:55 must not widen the measured range.
    frame.loc[frame.index[2], "high"] = 500.0
    assert median_session_range(frame, [index[0].date()]) == pytest.approx(2.0)


def test_every_contract_has_sane_specs():
    for symbol, contract in CONTRACTS.items():
        assert contract.symbol == symbol
        assert contract.tick_size > 0
        assert contract.point_value > 0
        assert contract.tick_value > 0
