"""The liquidity map: session windows, weekly gaps, and picking a draw."""

from __future__ import annotations

import pandas as pd
import pytest

from gapstrat import liquidity as liq

ET = "America/New_York"


def bars_from(spec: dict) -> pd.DataFrame:
    """Build a frame from {timestamp: (o,h,l,c)} entries."""
    index = pd.to_datetime(list(spec)).tz_localize(ET)
    rows = list(spec.values())
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index).assign(volume=0.0)


def test_asia_window_is_eight_pm_to_midnight_of_the_prior_evening():
    frame = bars_from({
        "2026-08-31 19:55": (100, 900, 90, 100),   # before 20:00: excluded
        "2026-08-31 20:05": (100, 150, 80, 120),   # inside Asia
        "2026-08-31 23:55": (120, 140, 95, 130),   # inside Asia
        "2026-09-01 00:05": (130, 800, 10, 140),   # after midnight: excluded
        "2026-09-01 09:30": (140, 141, 139, 140),
    })
    m = liq.build(frame, pd.Timestamp("2026-09-01").date())
    assert m.asia_high == 150.0
    assert m.asia_low == 80.0


def test_london_window_is_two_to_five_am():
    frame = bars_from({
        "2026-09-01 01:55": (100, 900, 90, 100),   # excluded
        "2026-09-01 02:05": (100, 160, 70, 120),   # inside London
        "2026-09-01 04:55": (120, 155, 75, 130),   # inside London
        "2026-09-01 05:05": (130, 800, 10, 140),   # excluded
        "2026-09-01 09:30": (140, 141, 139, 140),
    })
    m = liq.build(frame, pd.Timestamp("2026-09-01").date())
    assert (m.london_high, m.london_low) == (160.0, 70.0)


def test_session_open_is_the_first_bar_at_930():
    frame = bars_from({
        "2026-09-01 09:25": (50, 51, 49, 50),
        "2026-09-01 09:30": (140, 141, 139, 140),
    })
    assert liq.build(frame, pd.Timestamp("2026-09-01").date()).session_open == 140.0


def test_new_week_opening_gap_spans_friday_close_to_sunday_open():
    frame = bars_from({
        "2026-08-28 16:55": (200, 201, 199, 200),  # Friday, last bar before 17:00
        "2026-08-30 18:00": (215, 216, 214, 215),  # Sunday reopen
        "2026-08-31 09:30": (215, 216, 214, 215),  # Monday
    })
    low, high = liq.new_week_opening_gap(frame, pd.Timestamp("2026-08-31").date(), ET)
    assert (low, high) == (200.0, 215.0)


def test_new_week_gap_is_returned_low_first_when_the_week_gaps_down():
    frame = bars_from({
        "2026-08-28 16:55": (200, 201, 199, 220),
        "2026-08-30 18:00": (205, 206, 204, 205),
        "2026-08-31 09:30": (205, 206, 204, 205),
    })
    low, high = liq.new_week_opening_gap(frame, pd.Timestamp("2026-08-31").date(), ET)
    assert low < high and (low, high) == (205.0, 220.0)


def test_missing_weekend_data_gives_no_gap_rather_than_a_crash():
    frame = bars_from({"2026-08-31 09:30": (205, 206, 204, 205)})
    assert liq.new_week_opening_gap(frame, pd.Timestamp("2026-08-31").date(), ET) == (None, None)


# --- choosing the draw -----------------------------------------------------

def a_map(**kw) -> liq.LiquidityMap:
    base = dict(day=None, asia_high=120.0, asia_low=80.0, london_high=115.0,
                london_low=85.0, prior_day_high=140.0, prior_day_low=60.0)
    return liq.LiquidityMap(**{**base, **kw})


def test_draw_picks_the_named_pool_beyond_the_entry():
    m = a_map()
    assert m.draw(1, above=100.0, named="asia").name == "asia_high"
    assert m.draw(-1, above=100.0, named="asia").name == "asia_low"


def test_draw_ignores_a_pool_already_behind_price():
    # Price at 125 is above the Asia high, so it is no longer a target.
    assert a_map().draw(1, above=125.0, named="asia") is None


def test_nearest_and_furthest_pick_by_distance():
    m = a_map()
    assert m.draw(1, above=100.0, named="nearest").name == "london_high"
    assert m.draw(1, above=100.0, named="furthest").name == "prior_day_high"


def test_pools_omit_levels_that_were_never_built():
    m = liq.LiquidityMap(day=None, asia_high=120.0)
    assert [lv.name for lv in m.highs()] == ["asia_high"]
    assert m.lows() == []
