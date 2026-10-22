"""Unit tests for analytics/scoring.py with hand-computed expected values (see tests/fixtures)."""

import math

import pandas as pd
import pytest

from analytics import scoring
from tests.fixtures.analytics_tables import tiny_tables

WEIGHTS = {"congestion": 0.40, "size": 0.30, "growth": 0.30}


def test_percentile_rank_averages_ties_and_keeps_nan():
    s = pd.Series([10, 20, 20, 40, float("nan")], index=list("abcde"))
    p = scoring.percentile_rank(s)
    assert list(p[:4]) == [25.0, 62.5, 62.5, 100.0]
    assert math.isnan(p["e"])


def _universe_and_not_ranked():
    t = tiny_tables()
    return scoring.build_universe(t["faa_airports"], t["airport_ops"].set_index("airport"),
                                  t["airport_enplanements"].set_index("airport"),
                                  hub_classes=("L", "M", "S"), min_flown=1000)


def test_build_universe_gates_and_reasons():
    universe, not_ranked = _universe_and_not_ranked()
    assert sorted(universe.index) == ["AAA", "BBB", "CCC", "DDD"]
    assert not_ranked.loc["NNN", "reason"] == scoring.REASON_NONHUB
    assert not_ranked.loc["HHH", "reason"] == scoring.REASON_NO_ONTIME
    assert not_ranked.loc["EEE", "reason"] == scoring.REASON_FEW_DEPARTURES
    assert not_ranked.loc["HHH", "enplanements"] == 900  # metrics that exist are kept


def test_score_universe_hand_computed():
    universe, _ = _universe_and_not_ranked()
    s = scoring.score_universe(universe, WEIGHTS)
    # pct15 10,20,30,40 -> 25,50,75,100; excess taxi 1,3,2,4 -> 25,75,50,100
    assert s.loc["BBB", "pct_dep_15_pctl"] == 50.0
    assert s.loc["BBB", "excess_taxi_out_pctl"] == 75.0
    assert s.loc["AAA", "congestion"] == 25.0
    assert s.loc["BBB", "congestion"] == 62.5
    assert s.loc["DDD", "congestion"] == 100.0
    # size 1050, 3960, 3060, 2200 -> 25, 100, 75, 50; growth 5, -1, 2, 10 -> 75, 25, 50, 100
    assert s.loc["BBB", "size"] == 100.0
    assert s.loc["DDD", "growth"] == 100.0
    assert s.loc["AAA", "score"] == pytest.approx(40.0)  # .4*25 + .3*25 + .3*75
    assert s.loc["BBB", "score"] == pytest.approx(62.5)  # .4*62.5 + .3*100 + .3*25
    assert s.loc["CCC", "score"] == pytest.approx(62.5)  # .4*62.5 + .3*75 + .3*50
    assert s.loc["DDD", "score"] == pytest.approx(85.0)  # .4*100 + .3*50 + .3*100
    assert list(s.sort_values("rank").index[:1]) == ["DDD"]
    assert s.loc["BBB", "rank"] == 2 and s.loc["CCC", "rank"] == 2  # ties share the better rank
    assert s.loc["AAA", "rank"] == 4


def test_rank_ranges_bracket_the_rank():
    universe, _ = _universe_and_not_ranked()
    s = scoring.score_universe(universe, WEIGHTS)
    r = scoring.rank_ranges(universe, WEIGHTS, factors=(0.8, 1.0, 1.2))
    assert (r.rank_min <= s["rank"]).all() and (s["rank"] <= r.rank_max).all()
    assert (r.loc["DDD", "rank_min"], r.loc["DDD", "rank_max"]) == (1, 1)
    assert r.loc["AAA", "rank_max"] == 4


def test_unmet_demand_all_signals_strong():
    a = scoring.unmet_demand_assessment(congestion_pctl=90, growth_pctl=80, fare_pctl=70, dominance_pctl=10,
                                        official_status="schedule review", high=200 / 3,
                                        strong_min=3, moderate_min=2)
    assert a["signals_on"] == 4
    assert a["level"] == "strong"
    assert [s["status"] for s in a["signals"]] == ["on", "on", "on", "on"]


def test_unmet_demand_fare_with_high_dominance_is_ambiguous():
    a = scoring.unmet_demand_assessment(congestion_pctl=90, growth_pctl=10, fare_pctl=95, dominance_pctl=95,
                                        official_status=None, high=200 / 3, strong_min=3, moderate_min=2)
    fare = next(s for s in a["signals"] if s["name"] == "fare_premium")
    assert fare["status"] == "ambiguous"
    assert a["signals_on"] == 1
    assert a["level"] == "weak"


def test_unmet_demand_missing_airfare_is_insufficient_not_off():
    a = scoring.unmet_demand_assessment(congestion_pctl=70, growth_pctl=70, fare_pctl=None, dominance_pctl=None,
                                        official_status=None, high=200 / 3, strong_min=3, moderate_min=2)
    fare = next(s for s in a["signals"] if s["name"] == "fare_premium")
    assert fare["status"] == "insufficient data"
    assert a["level"] == "moderate"
