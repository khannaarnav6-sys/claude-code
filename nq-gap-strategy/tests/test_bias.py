"""Session bias detection and the structural stop it goes with."""

from __future__ import annotations

import pandas as pd
import pytest

from gapstrat.bias import SessionLevels, determine, session_levels
from gapstrat.data import NQ
from gapstrat.strategy import StrategyConfig, plan_session

ET = "America/New_York"


def frame(rows, start="2026-09-01 09:30", freq="5min"):
    index = pd.date_range(start, periods=len(rows), freq=freq, tz=ET)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index).assign(volume=0.0)


LEVELS = SessionLevels(
    prior_high=None, prior_low=None, prior_close=100.0,
    overnight_high=120.0, overnight_low=80.0, session_open=100.0,
)


# --- overnight break -------------------------------------------------------

def test_overnight_break_takes_the_side_price_is_outside():
    assert determine("overnight_break", frame([(100, 125, 99, 124)]), LEVELS) == 1
    assert determine("overnight_break", frame([(100, 101, 75, 76)]), LEVELS) == -1


def test_inside_the_overnight_range_is_no_bias():
    assert determine("overnight_break", frame([(100, 110, 95, 105)]), LEVELS) == 0


# --- opening range ---------------------------------------------------------

def test_opening_range_needs_the_range_to_finish_first():
    bars = frame([(100, 110, 95, 105), (105, 112, 104, 111)])
    assert determine("opening_range", bars, LEVELS) == 0  # only 2 of 3 range bars


def test_opening_range_break_sets_the_side():
    bars = frame([(100, 110, 95, 105), (105, 108, 99, 101), (101, 109, 100, 108),
                  (108, 118, 107, 116)])  # closes above the 110 range high
    assert determine("opening_range", bars, LEVELS) == 1


def test_most_recent_opening_range_break_wins():
    bars = frame([(100, 110, 95, 105), (105, 108, 99, 101), (101, 109, 100, 108),
                  (108, 118, 107, 116),  # breaks up
                  (116, 117, 90, 92)])   # then breaks back down
    assert determine("opening_range", bars, LEVELS) == -1


# --- sweep reversal (the pattern on the chart) -----------------------------

def test_sweep_of_the_lows_that_closes_back_above_turns_the_bias_up():
    # trades to 74, below the 80 overnight low, then closes back at 88
    bars = frame([(100, 101, 74, 88)])
    assert determine("sweep_reversal", bars, LEVELS) == 1


def test_breaking_the_low_and_staying_below_is_not_a_sweep():
    bars = frame([(100, 101, 74, 76)])  # closed below the level: a real break
    assert determine("sweep_reversal", bars, LEVELS) == 0


def test_failed_run_at_the_highs_turns_the_bias_down():
    bars = frame([(100, 126, 99, 112)])  # pokes above 120, closes back under
    assert determine("sweep_reversal", bars, LEVELS) == -1


# --- prior day -------------------------------------------------------------

def test_prior_day_compares_the_open_to_yesterdays_close():
    up = SessionLevels(None, None, 95.0, 120.0, 80.0, 100.0)
    down = SessionLevels(None, None, 105.0, 120.0, 80.0, 100.0)
    assert determine("prior_day", frame([(100, 101, 99, 100)]), up) == 1
    assert determine("prior_day", frame([(100, 101, 99, 100)]), down) == -1


def test_unknown_method_is_rejected_and_none_is_neutral():
    assert determine("none", frame([(100, 101, 99, 100)]), LEVELS) == 0
    with pytest.raises(ValueError):
        determine("crystal_ball", frame([(100, 101, 99, 100)]), LEVELS)


def test_missing_levels_produce_no_bias_rather_than_a_crash():
    empty = SessionLevels(None, None, None, None, None, None)
    for method in ("overnight_break", "sweep_reversal", "prior_day"):
        assert determine(method, frame([(100, 101, 99, 100)]), empty) == 0


# --- levels ----------------------------------------------------------------

def test_session_levels_read_the_overnight_window_not_the_session():
    overnight = frame([(100, 130, 70, 90)], start="2026-08-31 20:00")
    session = frame([(95, 200, 10, 150)], start="2026-09-01 09:30")
    bars = pd.concat([overnight, session])
    levels = session_levels(bars, pd.Timestamp("2026-09-01").date())
    assert levels.overnight_high == 130.0
    assert levels.overnight_low == 70.0
    assert levels.session_open == 95.0  # the 09:30 open, not the overnight open


# --- integration: bias filtering and the structural stop -------------------

def bullish_session():
    """Sweeps the lows, closes back up, then leaves a bullish gap."""
    return frame([
        (100, 101, 74, 88),    # the sweep: below 80, closes back above
        (88, 95, 87, 94),
        (94, 130, 93, 128),    # displacement
        (128, 135, 120, 133),  # gap: 101 -> 120 is untouched
    ])


def cfg(**kw):
    base = {"min_gap_points": 1.0, "min_stop_points": 1.0, "max_stop_points": None,
            "stop_buffer_ticks": 0, "signal_cutoff": "11:00"}
    return StrategyConfig(**{**base, **kw})


def test_bias_filter_lets_an_aligned_gap_through(monkeypatch):
    bars = bullish_session()
    monkeypatch.setattr("gapstrat.strategy.session_levels", lambda *a, **k: LEVELS)
    plan = plan_session(bars, bars.index[0].date(), cfg(bias_method="sweep_reversal"), NQ)
    assert plan is not None
    assert plan.direction == 1
    assert plan.session_bias == 1


def test_bias_filter_blocks_a_gap_that_argues_the_other_way(monkeypatch):
    # Same bars, but the levels make this a failed run at the highs -> short bias,
    # while the only gap available is bullish.
    bearish_levels = SessionLevels(None, None, 100.0, 95.0, 10.0, 100.0)
    monkeypatch.setattr("gapstrat.strategy.session_levels", lambda *a, **k: bearish_levels)
    bars = bullish_session()
    assert plan_session(bars, bars.index[0].date(), cfg(bias_method="sweep_reversal"), NQ) is None


def test_require_bias_off_takes_the_gap_regardless(monkeypatch):
    bearish_levels = SessionLevels(None, None, 100.0, 95.0, 10.0, 100.0)
    monkeypatch.setattr("gapstrat.strategy.session_levels", lambda *a, **k: bearish_levels)
    bars = bullish_session()
    plan = plan_session(bars, bars.index[0].date(),
                        cfg(bias_method="sweep_reversal", require_bias=False), NQ)
    assert plan is not None
    assert plan.direction == 1


def test_swing_stop_sits_below_the_swept_low():
    bars = bullish_session()
    plan = plan_session(bars, bars.index[0].date(), cfg(stop_style="swing"), NQ)
    assert plan is not None
    assert plan.stop_price == 74.0  # the low of the sweep bar, not the gap edge
    # and the target is the configured multiple of that much wider risk
    risk = plan.entry_price - plan.stop_price
    assert plan.target_price == pytest.approx(plan.entry_price + 2 * risk)


def test_swing_stop_is_wider_than_the_gap_stop():
    bars = bullish_session()
    day = bars.index[0].date()
    gap_stop = plan_session(bars, day, cfg(stop_style="gap_far"), NQ)
    swing_stop = plan_session(bars, day, cfg(stop_style="swing"), NQ)
    assert swing_stop.risk_points > gap_stop.risk_points
