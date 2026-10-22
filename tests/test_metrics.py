"""Unit tests for analytics/metrics.py with hand-computed expected values."""

import pandas as pd
import pytest

from analytics import metrics
from tests.fixtures.analytics_tables import distance_monthly


def test_long_haul_shares_per_airport_and_missing_stays_missing():
    s = metrics.long_haul_shares(distance_monthly(), long_haul_group=11)
    assert s["AAA"] == pytest.approx(30.0) and s["BBB"] == pytest.approx(0.0) and s["EEE"] == pytest.approx(100.0)
    assert "CCC" not in s.index  # no distance rows: unknown, never 0


def _pair(a, b, d, fare_factor, large_ms=0.5, year=2026, quarter=1, pax=100.0):
    return {"year": year, "quarter": quarter, "airport_1": a, "airport_2": b, "nsmiles": d,
            "passengers": pax, "fare": 10 * d ** 0.5 * fare_factor, "large_ms": large_ms}


def test_airfare_signals_fare_premium_against_distance_fit():
    # H-X fares are 10% above the power law, H-Y fares 10% below (in log terms): the fit stays on the law.
    pairs = pd.DataFrame([
        _pair("HHH", "XXX", 100, 1.1), _pair("HHH", "XXX", 1000, 1.1),
        _pair("YYY", "HHH", 100, 1 / 1.1), _pair("YYY", "HHH", 1000, 1 / 1.1),
        _pair("HHH", "XXX", 100, 5.0, year=2025, quarter=4),  # older quarter: ignored
    ])
    s = metrics.airfare_signals(pairs, quarters=1, min_pairs=1)
    assert s.loc["XXX", "fare_premium_pct"] == pytest.approx(10.0)
    assert s.loc["YYY", "fare_premium_pct"] == pytest.approx(-100 / 11)  # 1/1.1 - 1
    assert s.loc["HHH", "fare_premium_pct"] == pytest.approx(0.0, abs=1e-9)
    assert s.loc["HHH", "pairs"] == 4
    assert s.attrs["period"] == "2026-Q1..2026-Q1"


def test_airfare_signals_dominant_share_and_min_pairs():
    pairs = pd.DataFrame([
        _pair("HHH", "XXX", 100, 1.0, large_ms=0.9), _pair("HHH", "XXX", 1000, 1.0, large_ms=0.9),
        _pair("YYY", "HHH", 100, 1.0, large_ms=0.5), _pair("YYY", "HHH", 1000, 1.0, large_ms=0.5),
    ])
    s = metrics.airfare_signals(pairs, quarters=4, min_pairs=3)
    assert s.loc["XXX", "dominant_share_pct"] == pytest.approx(90.0)
    assert s.loc["HHH", "dominant_share_pct"] == pytest.approx(70.0)
    assert bool(s.loc["HHH", "sufficient"]) is True  # 4 pairs
    assert bool(s.loc["XXX", "sufficient"]) is False  # 2 pairs
