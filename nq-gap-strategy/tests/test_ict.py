"""The A+ sequence: sweep, structure shift, imbalance, discount, draw."""

from __future__ import annotations

import pandas as pd
import pytest

from gapstrat import liquidity as liq
from gapstrat.data import NQ
from gapstrat.ict import ICTConfig, plan_ict_session, swing_points

ET = "America/New_York"

# Asia ran 80-120; the model should hunt the 80 low and target the 120 high.
MAP = liq.LiquidityMap(day=None, asia_high=120.0, asia_low=80.0, session_open=100.0)


def frame(rows, start="2026-09-01 08:30"):
    index = pd.date_range(start, periods=len(rows), freq="5min", tz=ET)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index).astype(float).assign(volume=0.0)


def cfg(**kw):
    base = {"min_penetration": 1.0, "min_stop_points": 1.0, "max_stop_points": None,
            "stop_buffer_ticks": 0, "min_target_r": 0.5, "max_target_r": None}
    return ICTConfig(**{**base, **kw})


def plan(bars, config, monkeypatch, lmap=MAP):
    monkeypatch.setattr("gapstrat.ict.liq.build", lambda *a, **k: lmap)
    return plan_ict_session(bars, bars.index[0].date(), config, NQ)


def a_plus_session():
    """Sweeps the 80 Asia low, shifts structure, leaves an imbalance."""
    return frame([
        (100, 104, 98, 100),   # 0
        (100, 106, 99, 105),   # 1  <- fractal swing high at 106, the MSS reference
        (105, 105, 96, 98),    # 2
        (98,  99, 74, 96),     # 3  SWEEP: through 80, closes back above
        (96, 103, 95, 102),    # 4
        (102, 130, 101, 128),  # 5  displacement closes above 106: MSS + FVG
        (128, 133, 120, 130),  # 6
    ])


# --- fractals --------------------------------------------------------------

def test_swing_points_finds_local_extremes():
    highs = [10, 12, 11, 15, 13]
    lows = [5, 7, 4, 8, 6]
    sh, sl = swing_points(highs, lows, strength=1)
    assert sh == [1, 3]
    assert sl == [2]


def test_a_flat_series_has_no_swings():
    assert swing_points([5, 5, 5, 5], [1, 1, 1, 1], strength=1) == ([], [])


# --- the full sequence -----------------------------------------------------

def test_full_sequence_produces_a_long_targeting_the_asia_high(monkeypatch):
    p = plan(a_plus_session(), cfg(), monkeypatch)
    assert p is not None
    assert p.direction == 1
    assert p.swept == "asia_low"
    assert p.target_name == "asia_high"
    assert p.target_price == 120.0  # the Asia high, not a fixed multiple
    assert p.stop_price == 74.0  # below the wick that swept the low
    assert "mss" in p.notes and "fvg" in p.notes and "discount" in p.notes


def test_the_order_rests_below_the_market(monkeypatch):
    p = plan(a_plus_session(), cfg(), monkeypatch)
    assert p.entry_price <= p.reference_price
    assert p.stop_price < p.entry_price < p.target_price


def test_a_sweep_that_closes_below_the_level_is_a_breakdown_not_a_sweep(monkeypatch):
    bars = a_plus_session()
    bars.iloc[3, bars.columns.get_loc("close")] = 76.0  # closes under the 80 low
    assert plan(bars, cfg(), monkeypatch) is None
    assert plan(bars, cfg(require_close_back=False), monkeypatch) is not None


def test_penetration_must_be_deep_enough(monkeypatch):
    bars = a_plus_session()
    bars.iloc[3, bars.columns.get_loc("low")] = 79.5  # only half a point through
    assert plan(bars, cfg(min_penetration=1.0), monkeypatch) is None
    assert plan(bars, cfg(min_penetration=0.25), monkeypatch) is not None


def test_structure_shift_requires_taking_the_prior_swing(monkeypatch):
    # Same sweep, but the rebound stalls under the 106 swing high: no shift.
    bars = frame([
        (100, 104, 98, 100),
        (100, 106, 99, 105),   # the swing high the reversal would have to take
        (105, 105, 96, 98),
        (98,  99, 74, 96),     # sweep
        (96, 103, 95, 102),
        (102, 105, 101, 104),  # closes 104, still below 106
        (104, 105, 100, 104),
    ])
    assert plan(bars, cfg(require_mss=True), monkeypatch) is None


def test_shift_must_arrive_inside_the_confirmation_window(monkeypatch):
    assert plan(a_plus_session(), cfg(confirm_within=1), monkeypatch) is None
    assert plan(a_plus_session(), cfg(confirm_within=2), monkeypatch) is not None


def test_sweeps_before_the_killzone_are_ignored(monkeypatch):
    # Same session shifted an hour earlier, so the sweep lands at 07:30.
    bars = a_plus_session()
    bars.index = bars.index - pd.Timedelta(hours=1)
    assert plan(bars, cfg(killzone_start="08:30"), monkeypatch) is None


# --- entry models ----------------------------------------------------------

def test_entry_models_place_the_limit_differently(monkeypatch):
    bars = a_plus_session()
    at_fvg = plan(bars, cfg(entry_model="fvg"), monkeypatch)
    at_ote = plan(bars, cfg(entry_model="ote"), monkeypatch)
    at_level = plan(bars, cfg(entry_model="level"), monkeypatch)
    assert at_level.entry_price == 80.0  # the swept level itself
    assert at_fvg.entry_price != at_ote.entry_price
    # Entering at the swept level is the deepest, so it risks least.
    assert at_level.risk_points < at_fvg.risk_points


def test_discount_filter_rejects_an_entry_above_equilibrium(monkeypatch):
    # A shallow sweep and a modest rebound put the dealing range's midpoint at
    # 101, below the 104 proximal edge of the imbalance.
    shallow = liq.LiquidityMap(day=None, asia_high=120.0, asia_low=95.0, session_open=100.0)
    bars = frame([
        (100, 104, 98, 100),
        (100, 106, 99, 105),
        (105, 105, 96, 98),
        (98,  99, 90, 96),     # sweeps 95, closes back above
        (96, 103, 95, 102),
        (102, 112, 104, 110),  # shift; imbalance runs 99 -> 104
    ])
    strict = plan(bars, cfg(fvg_entry="proximal", require_discount=True), monkeypatch, shallow)
    loose = plan(bars, cfg(fvg_entry="proximal", require_discount=False), monkeypatch, shallow)
    assert strict is None  # 104 sits in premium
    assert loose is not None and loose.entry_price == 104.0


# --- the draw --------------------------------------------------------------

def test_a_target_too_close_to_pay_for_the_risk_is_skipped(monkeypatch):
    near = liq.LiquidityMap(day=None, asia_high=104.0, asia_low=80.0, session_open=100.0)
    assert plan(a_plus_session(), cfg(min_target_r=2.0), monkeypatch, near) is None


def test_a_fantasy_target_is_skipped(monkeypatch):
    far = liq.LiquidityMap(day=None, asia_high=9000.0, asia_low=80.0, session_open=100.0)
    assert plan(a_plus_session(), cfg(max_target_r=10.0), monkeypatch, far) is None


def test_no_pool_left_above_means_no_trade(monkeypatch):
    behind = liq.LiquidityMap(day=None, asia_high=90.0, asia_low=80.0, session_open=100.0)
    assert plan(a_plus_session(), cfg(), monkeypatch, behind) is None


def test_fixed_target_falls_back_to_an_r_multiple(monkeypatch):
    p = plan(a_plus_session(), cfg(target="fixed", fixed_target_r=3.0), monkeypatch)
    assert p.target_name == "3R"
    assert p.target_price == pytest.approx(p.entry_price + 3 * p.risk_points)


def test_only_the_named_sweep_pools_are_hunted(monkeypatch):
    lmap = liq.LiquidityMap(day=None, asia_high=120.0, asia_low=80.0,
                            london_low=80.0, session_open=100.0)
    p = plan(a_plus_session(), cfg(sweep_pools=("london",)), monkeypatch, lmap)
    assert p is not None and p.swept == "london_low"
    assert plan(a_plus_session(), cfg(sweep_pools=("prior_week",)), monkeypatch, lmap) is None


def test_higher_timeframe_filter_blocks_a_trade_into_an_opposing_gap(monkeypatch):
    blocked = liq.LiquidityMap(day=None, asia_high=120.0, asia_low=80.0, session_open=100.0,
                               weekly_fvgs=[(110.0, 115.0, -1)])  # bearish gap in the path
    assert plan(a_plus_session(), cfg(respect_htf_fvg=True), monkeypatch, blocked) is None
    assert plan(a_plus_session(), cfg(respect_htf_fvg=False), monkeypatch, blocked) is not None


def test_configuration_is_validated():
    for bad in (ICTConfig(entry_model="vibes"), ICTConfig(target="the_moon"),
                ICTConfig(sweep_pools=("mars",)), ICTConfig(ote_level=0.1)):
        with pytest.raises(ValueError):
            bad.validate()
