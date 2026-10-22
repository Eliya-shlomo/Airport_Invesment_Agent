"""Unit tests for analytics/confidence.py."""

from analytics import confidence


def test_coverage_warnings_partial_months_and_few_departures():
    assert confidence.coverage_warnings("AAA", months=12, flown=5000, first="2025-09", last="2026-08") == []
    w = confidence.coverage_warnings("EEE", months=3, flown=300, first="2026-06", last="2026-08")
    assert len(w) == 2
    assert "EEE" in w[0] and "3 of 12 months" in w[0] and "2026-06..2026-08" in w[0]
    assert "300" in w[1]


def test_measured_values_are_high_only_with_full_coverage():
    assert confidence.from_coverage(months=12, flown=1000) == "high"
    assert confidence.from_coverage(months=11, flown=5000) == "medium"
    assert confidence.from_coverage(months=12, flown=999) == "medium"


def test_a_proxy_is_one_level_lower():
    assert confidence.from_coverage(months=12, flown=5000, proxy=True) == "medium"
    assert confidence.from_coverage(months=11, flown=5000, proxy=True) == "low"


def test_lowest_picks_the_weakest_level():
    assert confidence.lowest(["high", "medium"]) == "medium"
    assert confidence.lowest(["medium", "low", "high"]) == "low"
