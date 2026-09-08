"""Decoding Dukascopy's tick files, and the URL gotchas around them."""

from __future__ import annotations

import lzma
import struct
from datetime import date

import pandas as pd
import pytest

from gapstrat import dukascopy as duka


def encode(ticks) -> bytes:
    """Build a bi5 payload: raw LZMA over 20-byte big-endian records."""
    body = b"".join(struct.pack(">IIIff", ms, ask, bid, av, bv) for ms, ask, bid, av, bv in ticks)
    return lzma.compress(body, format=lzma.FORMAT_ALONE)


def test_the_month_in_the_url_is_zero_indexed():
    # June is 05, not 06 -- getting this wrong silently returns the wrong month.
    url = duka.FEED.format(inst="X", y=2025, m=date(2025, 6, 12).month - 1, d=12, h=14)
    assert "/2025/05/12/14h_ticks.bi5" in url


def test_january_maps_to_month_zero():
    url = duka.FEED.format(inst="X", y=2026, m=date(2026, 1, 3).month - 1, d=3, h=9)
    assert "/2026/00/03/09h_ticks.bi5" in url


def test_decode_scales_prices_and_stamps_them_in_utc():
    raw = encode([(0, 21_837_181, 21_835_798, 1.5, 2.5),
                  (60_000, 21_840_000, 21_838_000, 1.0, 1.0)])
    frame = duka._decode(raw, date(2025, 6, 12), 14)
    assert len(frame) == 2
    # price is the mid of bid and ask, divided by the index scale of 1000
    assert frame["price"].iloc[0] == pytest.approx((21_837_181 + 21_835_798) / 2 / 1000)
    assert str(frame.index.tz) == "UTC"
    assert frame.index[0] == pd.Timestamp("2025-06-12 14:00:00", tz="UTC")
    assert frame.index[1] == pd.Timestamp("2025-06-12 14:01:00", tz="UTC")


def test_an_empty_hour_decodes_to_an_empty_frame():
    assert duka._decode(b"", date(2025, 6, 12), 14).empty
    assert duka._decode(encode([]), date(2025, 6, 12), 14).empty


def test_decoded_prices_land_in_a_sane_index_range():
    raw = encode([(0, 21_837_181, 21_835_798, 1.0, 1.0)])
    price = duka._decode(raw, date(2025, 6, 12), 14)["price"].iloc[0]
    assert 1_000 < price < 100_000  # an index level, not raw integers


# --- the trading week ------------------------------------------------------

def test_saturday_is_never_requested():
    assert duka._trading_hours(date(2025, 6, 14)) == []  # a Saturday


def test_sunday_only_covers_the_reopen():
    assert duka._trading_hours(date(2025, 6, 15)) == [20, 21, 22, 23]


def test_friday_stops_before_the_weekend_close():
    hours = duka._trading_hours(date(2025, 6, 13))
    assert hours[0] == 0 and hours[-1] == 21


def test_a_normal_weekday_is_a_full_day():
    assert duka._trading_hours(date(2025, 6, 11)) == list(range(24))


def test_cache_paths_are_per_instrument_and_day():
    path = duka.cache_file("USATECHIDXUSD", date(2025, 6, 12))
    assert path.name == "2025-06-12.csv"
    assert path.parent.name == "USATECHIDXUSD"
