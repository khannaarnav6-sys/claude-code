"""The sweep-and-reclaim setup, on sequences built to a known answer."""

from __future__ import annotations

import pandas as pd
import pytest

from gapstrat.bias import SessionLevels
from gapstrat.data import NQ
from gapstrat.sweep import SweepConfig, plan_sweep_session

ET = "America/New_York"
LEVELS = SessionLevels(
    prior_high=None, prior_low=None, prior_close=None,
    overnight_high=120.0, overnight_low=80.0, session_open=100.0,
)


def frame(rows, start="2026-09-01 09:30"):
    index = pd.date_range(start, periods=len(rows), freq="5min", tz=ET)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index).assign(volume=0.0)


def cfg(**kw):
    base = {"min_stop_points": 1.0, "max_stop_points": None, "stop_buffer_ticks": 0,
            "require_displacement": False}
    return SweepConfig(**{**base, **kw})


def plan(bars, config, monkeypatch):
    monkeypatch.setattr("gapstrat.sweep.session_levels", lambda *a, **k: LEVELS)
    return plan_sweep_session(bars, bars.index[0].date(), config, NQ)


def swept_session():
    """Runs the 80 overnight low to 74, closes back above, then reclaims."""
    return frame([
        (100, 101, 99, 100),
        (100, 101, 74, 88),    # the sweep: through 80, closes back above it
        (88, 95, 87, 94),      # close 94 > sweep high of 101? no -- not yet
        (94, 130, 93, 128),    # closes 128, above the sweep bar's high: confirmed
        (128, 132, 120, 130),
    ])


# --- detection -------------------------------------------------------------

def test_sweep_and_reclaim_produces_a_long_at_the_reclaimed_level(monkeypatch):
    p = plan(swept_session(), cfg(entry_anchor="level"), monkeypatch)
    assert p is not None
    assert p.direction == 1
    assert p.entry_price == 80.0  # the reclaimed overnight low
    assert p.stop_price == 74.0  # under the wick that swept it
    assert p.risk_points == 6.0
    assert p.target_price == 92.0  # 2R above entry
    assert p.entry_type == "limit"


def test_order_only_starts_working_at_the_confirmation_bar(monkeypatch):
    bars = swept_session()
    p = plan(bars, cfg(), monkeypatch)
    assert p.working_from == bars.index[3]  # not the sweep bar at index 1


def test_a_wick_that_does_not_reclaim_is_not_a_sweep(monkeypatch):
    bars = frame([
        (100, 101, 99, 100),
        (100, 101, 74, 76),  # closes below 80: a genuine breakdown, not a sweep
        (76, 78, 70, 72),
        (72, 130, 71, 128),
    ])
    assert plan(bars, cfg(), monkeypatch) is None


def test_sweep_without_confirmation_in_time_is_dropped(monkeypatch):
    bars = frame([
        (100, 101, 99, 100),
        (100, 101, 74, 88),   # sweep
        (88, 90, 86, 87),     # drifts sideways
        (87, 89, 85, 86),
        (86, 130, 85, 128),   # confirms, but 3 bars later
    ])
    assert plan(bars, cfg(confirm_within=2), monkeypatch) is None
    assert plan(bars, cfg(confirm_within=3), monkeypatch) is not None


def test_failed_run_at_the_highs_produces_a_short(monkeypatch):
    bars = frame([
        (100, 101, 99, 100),
        (100, 126, 99, 112),  # sweeps 120, closes back below
        (112, 114, 60, 62),   # closes under the sweep bar's low: confirmed
    ])
    p = plan(bars, cfg(entry_anchor="level"), monkeypatch)
    assert p is not None
    assert p.direction == -1
    assert p.entry_price == 120.0
    assert p.stop_price == 126.0
    assert p.target_price == 108.0


def test_penetration_must_be_deep_enough(monkeypatch):
    bars = frame([
        (100, 101, 99, 100),
        (100, 101, 79.5, 88),  # only half a point through the level
        (88, 130, 87, 128),
    ])
    # the resulting stop is only half a point away, so the risk floor has to be
    # lowered too or the trade is dropped for a reason other than penetration
    assert plan(bars, cfg(min_penetration=1.0, min_stop_points=0.25), monkeypatch) is None
    assert plan(bars, cfg(min_penetration=0.25, min_stop_points=0.25), monkeypatch) is not None


# --- entry anchors ---------------------------------------------------------

def test_entry_anchors_place_the_limit_at_different_levels(monkeypatch):
    bars = swept_session()
    at_level = plan(bars, cfg(entry_anchor="level"), monkeypatch)
    at_mid = plan(bars, cfg(entry_anchor="sweep_mid"), monkeypatch)
    at_close = plan(bars, cfg(entry_anchor="sweep_close"), monkeypatch)
    assert at_level.entry_price == 80.0
    assert at_mid.entry_price == 87.5  # midpoint of the 74-101 sweep bar
    assert at_close.entry_price == 88.0  # the sweep bar's close
    # a nearer entry is a wider stop, so risk grows as the entry rises
    assert at_level.risk_points < at_mid.risk_points < at_close.risk_points


def test_entry_above_the_market_is_rejected(monkeypatch):
    # The confirmation bar closes below the anchor, so the "limit" would be
    # marketable on arrival -- the fill model would flatter it.
    bars = frame([
        (100, 101, 99, 100),
        (100, 101, 74, 88),
        (88, 102, 87, 85),  # closes at 85, below a sweep_close anchor of 88
    ])
    assert plan(bars, cfg(entry_anchor="sweep_close"), monkeypatch) is None


# --- displacement filter ---------------------------------------------------

def test_displacement_filter_requires_an_imbalance_on_the_reclaim(monkeypatch):
    # Confirmation happens on overlapping bars, so no fair value gap forms.
    bars = frame([
        (100, 101, 99, 100),
        (100, 101, 74, 88),
        (88, 100, 87, 99),
        (99, 103, 90, 102),  # closes above the sweep high, but bars all overlap
    ])
    assert plan(bars, cfg(require_displacement=True), monkeypatch) is None
    assert plan(bars, cfg(require_displacement=False), monkeypatch) is not None


def test_risk_bounds_are_enforced(monkeypatch):
    bars = swept_session()
    assert plan(bars, cfg(min_stop_points=20.0), monkeypatch) is None
    assert plan(bars, cfg(max_stop_points=2.0), monkeypatch) is None


def test_bad_configuration_is_rejected():
    with pytest.raises(ValueError):
        SweepConfig(reference="tea_leaves").validate()
    with pytest.raises(ValueError):
        SweepConfig(entry_anchor="vibes").validate()
    with pytest.raises(ValueError):
        SweepConfig(target_r=0).validate()
