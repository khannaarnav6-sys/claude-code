"""The holdout runner must not refit anything."""

from __future__ import annotations

import pandas as pd
import pytest

from gapstrat.holdout import PROXY, by_period, scale_gap_config
from gapstrat.strategy import StrategyConfig


def test_thresholds_scale_the_same_way_as_the_cross_market_check():
    from gapstrat.crossmarket import scale_to_instrument

    base = StrategyConfig()
    assert scale_gap_config(base, 375.2).min_gap_points == scale_to_instrument(base, 375.2).min_gap_points
    assert scale_gap_config(base, 58.0).max_stop_points == scale_to_instrument(base, 58.0).max_stop_points


def test_scaling_changes_no_rule_only_the_units():
    base = StrategyConfig(target_r=3.0, entry_style="proximal", bias_method="prior_day",
                          signal_cutoff="10:30")
    scaled = scale_gap_config(base, 400.0)
    assert (scaled.target_r, scaled.entry_style, scaled.bias_method, scaled.signal_cutoff) == (
        3.0, "proximal", "prior_day", "10:30")


def test_the_proxy_contract_borrows_nq_specs():
    # A CFD has no contract size; these exist so tick rounding and costs match.
    assert PROXY.tick_size == 0.25
    assert PROXY.point_value == 20.0


class _Trade:
    def __init__(self, day, r):
        self.day, self.r_multiple, self.filled = day, r, True


class _Result:
    def __init__(self, trades):
        self._t = trades

    @property
    def filled(self):
        return self._t


def test_by_period_splits_trades_into_quarters():
    trades = [_Trade(pd.Timestamp("2025-07-15").date(), 2.0),
              _Trade(pd.Timestamp("2025-08-20").date(), -1.0),
              _Trade(pd.Timestamp("2025-11-05").date(), 3.0)]
    table = by_period(_Result(trades), freq="QE")
    assert list(table["trades"]) == [2, 1]
    assert table["expectancy_r"].iloc[0] == pytest.approx(0.5)
    assert table["expectancy_r"].iloc[1] == pytest.approx(3.0)


def test_by_period_on_no_trades_is_empty_not_an_error():
    assert by_period(_Result([])).empty
