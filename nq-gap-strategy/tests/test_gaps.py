"""Hand-built bar sequences with known answers."""

from __future__ import annotations

import pandas as pd
import pytest

from gapstrat.gaps import find_gaps, mark_mitigation, unmitigated_at


def frame(rows: list[tuple[float, float, float, float]], start="2026-09-01 09:30") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(rows), freq="5min", tz="America/New_York")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index).assign(volume=0.0)


def test_bullish_gap_found_with_correct_zone():
    # bar 3's low (120) sits above bar 1's high (105): the 105-120 range is untouched
    bars = frame([(100, 105, 99, 104), (104, 130, 103, 128), (128, 135, 120, 133)])
    gaps = find_gaps(bars)
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.direction == 1
    assert (gap.low, gap.high) == (105, 120)
    assert gap.size == 15
    assert gap.midpoint == 112.5
    assert gap.proximal == 120  # first edge a pullback reaches
    assert gap.distal == 105


def test_bearish_gap_found_with_correct_zone():
    bars = frame([(130, 132, 125, 126), (126, 127, 100, 102), (102, 105, 98, 100)])
    gaps = find_gaps(bars)
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.direction == -1
    assert (gap.low, gap.high) == (105, 125)
    assert gap.proximal == 105
    assert gap.distal == 125


def test_overlapping_bars_are_not_a_gap():
    bars = frame([(100, 110, 95, 105), (105, 120, 104, 118), (118, 125, 108, 122)])
    assert find_gaps(bars) == []


def test_min_size_filters_small_imbalances():
    bars = frame([(100, 105, 99, 104), (104, 130, 103, 128), (128, 135, 107, 133)])
    assert len(find_gaps(bars)) == 1  # 105 -> 107 is a 2-point gap
    assert find_gaps(bars, min_size=5.0) == []


def test_mitigation_records_touch_and_fill():
    bars = frame(
        [
            (100, 105, 99, 104),
            (104, 130, 103, 128),
            (128, 135, 120, 133),
            (133, 134, 118, 119),  # dips to 118: inside the 105-120 zone, touched
            (119, 121, 104, 106),  # trades to 104: below 105, fully filled
        ]
    )
    gap = mark_mitigation(find_gaps(bars)[0], bars)
    assert gap.touched_at == bars.index[3]
    assert gap.filled_at == bars.index[4]


def test_gap_untouched_stays_unmitigated():
    bars = frame(
        [
            (100, 105, 99, 104),
            (104, 130, 103, 128),
            (128, 135, 120, 133),
            (133, 140, 125, 138),  # never trades back to 120
        ]
    )
    gap = mark_mitigation(find_gaps(bars)[0], bars)
    assert gap.touched_at is None
    assert gap.filled_at is None
    assert unmitigated_at([gap], bars.index[-1]) == [gap]


def test_forming_bars_cannot_mitigate_their_own_gap():
    bars = frame([(100, 105, 99, 104), (104, 130, 103, 128), (128, 135, 120, 133)])
    gap = mark_mitigation(find_gaps(bars)[0], bars)
    assert gap.touched_at is None


def test_entry_styles():
    bars = frame([(100, 105, 99, 104), (104, 130, 103, 128), (128, 135, 120, 133)])
    gap = find_gaps(bars)[0]
    assert gap.entry_price("proximal") == 120
    assert gap.entry_price("mid") == 112.5
    assert gap.entry_price("distal") == 105
    with pytest.raises(ValueError):
        gap.entry_price("nonsense")
