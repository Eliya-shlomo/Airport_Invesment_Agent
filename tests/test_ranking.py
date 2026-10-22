"""Single-metric rankings: scoring.rank_by_metric and the top_airports_by tool, with hand-computed values on the
tiny fixture (see tests/fixtures/analytics_tables.py)."""

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from agent.tools import top_airports_by
from analytics import config, scoring

CONTRACT_KEYS = {"result", "method", "assumptions", "sources", "coverage", "confidence", "warnings"}


def call(ctx, **args):
    return json.loads(top_airports_by.invoke(args))


def test_rank_by_metric_orders_ties_and_drops_missing():
    v = pd.Series({"a": 5.0, "b": 9.0, "c": 5.0, "d": float("nan")})
    high = scoring.rank_by_metric(v, descending=True)
    assert list(high.index) == ["b", "a", "c"] and list(high["rank"]) == [1, 2, 2]
    low = scoring.rank_by_metric(v, descending=False)
    assert list(low.index) == ["a", "c", "b"] and list(low["rank"]) == [1, 1, 3]


def test_growth_highest_hand_computed(ctx):
    out = call(ctx, metric="growth", top_n=2)
    assert set(out) == CONTRACT_KEYS
    r = out["result"]
    assert r["metric"] == "growth" and r["unit"] == "percent" and r["order"] == "highest"
    assert [(x["airport"], x["rank_by_metric"], x["value"]) for x in r["ranked"]] == [("DDD", 1, 10.0), ("AAA", 2, 5.0)]
    assert r["ranked"][0]["expansion_score"] == pytest.approx(85.0) and r["ranked"][0]["rank_national"] == 1
    assert r["ranked_in_scope"] == 4 and r["universe_size"] == 4
    assert out["confidence"] == "medium"  # one year of growth
    assert any("CY2024" in a and "CY2025" in a for a in out["assumptions"])


def test_hubs_outside_the_ranked_set_are_listed_with_their_own_values(ctx):
    r = call(ctx, metric="growth")["result"]
    nr = {x["airport"]: x for x in r["not_ranked"]}
    assert set(nr) == {"HHH", "EEE"}  # hubs lacking data; the nonhub NNN is not listed nationally
    assert nr["HHH"]["growth_pct"] == pytest.approx(12.5)  # faster than DDD, so omitting it would mislead
    out = call(ctx, metric="growth")
    assert any("HHH" in w and "not ranked" in w for w in out["warnings"])


def test_lowest_order_and_measured_confidence(ctx):
    out = call(ctx, metric="late_departures", order="lowest", top_n=2)
    r = out["result"]
    assert r["order"] == "lowest"
    assert [(x["airport"], x["value"]) for x in r["ranked"]] == [("AAA", 10.0), ("BBB", 20.0)]
    assert out["confidence"] == "high"


def test_excess_taxi_out_is_medium(ctx):
    out = call(ctx, metric="excess_taxi_out", top_n=1)
    assert out["result"]["ranked"][0]["airport"] == "DDD" and out["result"]["ranked"][0]["value"] == pytest.approx(4.0)
    assert out["confidence"] == "medium"


def test_size_in_states_lists_nonhub_in_not_ranked(ctx):
    out = call(ctx, metric="size", states=["ma"])
    r = out["result"]
    assert r["scope"] == "states MA"
    assert [(x["airport"], x["value"]) for x in r["ranked"]] == [("BBB", 3960), ("AAA", 1050)]
    assert {x["airport"] for x in r["not_ranked"]} == {"NNN", "HHH"}
    assert out["confidence"] == "high"


def test_region_scope(ctx):
    r = call(ctx, metric="mean_departure_delay", region="new_england")["result"]
    assert [(x["airport"], x["value"]) for x in r["ranked"]] == [("BBB", 8.0), ("AAA", 5.0)]


def test_ties_share_the_better_rank(ctx):
    r = call(ctx, metric="cancellation_rate")["result"]  # every airport has 2 of 102 flights cancelled
    assert {x["rank_by_metric"] for x in r["ranked"]} == {1}
    assert r["ranked"][0]["value"] == pytest.approx(2 / 102 * 100, abs=0.05)


def test_missing_metric_is_reported_never_zero_or_dropped(ctx):
    out = call(ctx, metric="long_haul_share")
    r = out["result"]
    assert [(x["airport"], x["value"]) for x in r["ranked"]] == [("AAA", 30.0), ("BBB", 0.0)]
    assert r["missing_metric"] == ["CCC", "DDD"]
    assert any("CCC" in w and "DDD" in w for w in out["warnings"])
    assert out["confidence"] == "high"
    assert any("international" in w.lower() for w in out["warnings"])


def test_invalid_input_is_rejected(ctx):
    for args in [{"metric": "profit"}, {"metric": "growth", "order": "best"}, {"metric": "growth", "top_n": 0}, {}]:
        with pytest.raises(ValidationError):
            top_airports_by.invoke(args)
    assert "error" in call(ctx, metric="growth", region="new_england", states=["MA"])


def test_schema_lists_exactly_the_configured_metrics():
    schema = top_airports_by.tool_call_schema.model_json_schema()
    assert set(schema["properties"]["metric"]["enum"]) == set(config.RANK_METRICS)
