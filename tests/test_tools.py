"""Tests for agent/tools/: the tool result contract on the tiny fixture tables."""

import json

import pytest
from pydantic import ValidationError

from agent import tools
from agent.tools import common

CONTRACT_KEYS = {"result", "method", "assumptions", "sources", "coverage", "confidence", "warnings"}
BY_NAME = {t.name: t for t in tools.TOOLS}


def call(ctx, name, **args):
    """Run a tool as the agent does and read the JSON the model would see."""
    return json.loads(BY_NAME[name].invoke(args))


def assert_contract(out):
    assert set(out) == CONTRACT_KEYS, out
    assert out["confidence"] in {"high", "medium", "low"}
    for s in out["sources"]:
        assert set(s) == {"dataset", "period", "fetched"}


def test_rank_region_orders_by_national_rank_and_lists_not_ranked(ctx):
    out = call(ctx, "rank_airports", region="new_england")
    assert_contract(out)
    ranked = out["result"]["ranked"]
    assert [r["airport"] for r in ranked] == ["BBB", "AAA"]
    assert [r["rank_in_scope"] for r in ranked] == [1, 2]
    assert ranked[0]["rank_national"] == 2 and ranked[0]["score"] == pytest.approx(62.5)
    assert ranked[0]["components"]["congestion"]["score"] == pytest.approx(62.5)
    assert ranked[0]["components"]["size"]["enplanements"] == 3960
    assert ranked[0]["components"]["growth"]["growth_pct"] == pytest.approx(-1.0)
    assert out["result"]["universe_size"] == 4
    not_ranked = {r["airport"]: r["reason"] for r in out["result"]["not_ranked"]}
    assert set(not_ranked) == {"NNN", "HHH"}
    assert any("HHH" in w for w in out["warnings"])
    assert out["confidence"] == "medium"
    ontime = next(s for s in out["sources"] if s["dataset"] == "BTS On-Time Reporting")
    assert ontime == {"dataset": "BTS On-Time Reporting", "period": "2025-09..2026-08", "fetched": "2026-10-22"}


def test_rank_by_states_and_top_n(ctx):
    out = call(ctx, "rank_airports", states=["ca"], top_n=1)
    assert [r["airport"] for r in out["result"]["ranked"]] == ["DDD"]
    assert out["result"]["ranked_in_scope"] == 2
    assert [r["airport"] for r in out["result"]["not_ranked"]] == ["EEE"]


def test_invalid_arguments_are_rejected_by_the_typed_schema(ctx):
    for name, args in [("rank_airports", {"region": "atlantis"}), ("rank_airports", {"top_n": "ten"}),
                       ("rank_airports", {"top_n": 0}), ("long_haul_share", {})]:
        with pytest.raises(ValidationError):  # inside the agent, LangChain sends this error back to the model
            BY_NAME[name].invoke(args)


def test_region_and_states_together_is_a_structured_error(ctx):
    out = call(ctx, "rank_airports", region="new_england", states=["MA"])
    assert "error" in out and "not both" in out["error"]


def test_airport_scores_for_follow_ups(ctx):
    out = call(ctx, "airport_scores", airports=["bbb", "NNN"])
    assert_contract(out)
    assert [r["airport"] for r in out["result"]["ranked"]] == ["BBB"]
    # BBB and CCC tie at equal weights; more weight on size puts BBB ahead, more on growth puts CCC ahead.
    assert out["result"]["ranked"][0]["rank_range"] == {"min": 2, "max": 3}
    assert out["result"]["not_ranked"][0]["airport"] == "NNN"


def test_compare_congestion_computes_differences(ctx):
    out = call(ctx, "compare_congestion", airports=["AAA", "DDD"])
    assert_contract(out)
    rows = {r["airport"]: r for r in out["result"]["airports"]}
    assert rows["AAA"]["pct_dep_15"] == pytest.approx(10.0)
    assert rows["DDD"]["excess_taxi_out_min"] == pytest.approx(4.0)
    diff = out["result"]["difference_first_minus_second"]
    assert diff["pct_dep_15_points"] == pytest.approx(-30.0)
    assert diff["excess_taxi_out_min"] == pytest.approx(-3.0)
    assert diff["mean_dep_delay_min"] == pytest.approx(-10.0)
    assert out["confidence"] == "medium"
    assert out["result"]["verdict"] == "DDD is more congested"  # congestion 25 vs 100


def test_compare_congestion_small_gap_is_similar(ctx):
    out = call(ctx, "compare_congestion", airports=["BBB", "CCC"])  # both congestion 62.5
    assert out["result"]["verdict"] == "similar congestion"


def test_compare_unknown_airport_error_names_it(ctx):
    out = call(ctx, "compare_congestion", airports=["XYZ", "AAA"])
    assert "error" in out and "XYZ" in out["error"] and "US airports only" in out["error"]


def test_city_name_error_offers_the_airports_it_may_mean(ctx):
    out = call(ctx, "long_haul_share", airport="Brav")  # part of a city name: not read as an airport
    assert "not an airport code" in out["error"] and "US airports only" not in out["error"]
    assert out["valid_options"] == ["BBB (Bravo, MA)"]


def test_only_hub_in_a_city_of_that_name_is_read_as_that_airport_and_said(ctx):
    out = call(ctx, "long_haul_share", airport="bravo")
    assert out["result"]["airport"] == "BBB"
    assert any("'bravo' was read as BBB (Bravo, MA)" in a for a in out["assumptions"])


def test_only_nonhub_in_a_city_of_that_name_is_offered_not_picked(ctx):
    out = call(ctx, "long_haul_share", airport="November")  # NNN is a nonhub, like Lebanon NH for "Lebanon"
    assert out["valid_options"] == ["NNN (November, MA)"]


def test_unknown_place_name_error_says_what_works(ctx):
    out = call(ctx, "long_haul_share", airport="Atlantis")
    assert "matches no US airport code or city" in out["error"] and "US airports only" in out["error"]


def test_long_haul_share_full_year_is_high_confidence(ctx):
    out = call(ctx, "long_haul_share", airport="AAA")
    assert_contract(out)
    assert out["result"]["long_haul_share_pct"] == pytest.approx(30.0)
    assert out["result"]["departures"] == 1200
    assert out["confidence"] == "high"
    assert any("2,500" in a for a in out["assumptions"])


def test_long_haul_share_partial_year_warns(ctx):
    out = call(ctx, "long_haul_share", airport="EEE")
    assert out["confidence"] == "medium"
    assert any("3 of 12 months" in w for w in out["warnings"])


def test_unmet_demand_is_low_confidence_signal_count(ctx):
    out = call(ctx, "unmet_demand", airport="DDD")
    assert_contract(out)
    assert out["confidence"] == "low"
    assert out["result"]["level"] == "moderate"  # congestion and growth on, fare insufficient, no FAA status
    assert out["result"]["signals_on"] == 2
    assert "indirect" in out["method"].lower()


def test_unmet_demand_for_unranked_airport_is_an_error_with_reason(ctx):
    out = call(ctx, "unmet_demand", airport="NNN")
    assert "error" in out and "nonhub" in out["error"].lower()


def test_a_bug_in_a_tool_becomes_an_error_for_the_model(ctx, monkeypatch):
    monkeypatch.setattr("agent.tools.long_haul.resolve_one", lambda *a: 1 / 0)
    out = call(ctx, "long_haul_share", airport="AAA")
    assert out["error"].startswith("long_haul_share failed: ZeroDivisionError")


def test_tool_schemas_show_the_model_only_its_own_arguments():
    names = [t.name for t in tools.TOOLS]
    assert names == ["rank_airports", "top_airports_by", "airport_scores", "compare_congestion",
                     "long_haul_share", "unmet_demand"]
    schema = BY_NAME["rank_airports"].tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"region", "states", "top_n"} and not schema.get("required")
    assert "terminal-expansion" in BY_NAME["rank_airports"].description


def test_score_and_congestion_answers_carry_the_airside_caveat(ctx):
    # The brief asks about terminals; congestion is measured on the airfield.
    for name, args in [("rank_airports", {}), ("compare_congestion", {"airports": ["AAA", "BBB"]})]:
        out = call(ctx, name, **args)
        assert any("airside" in a and "terminal" in a for a in out["assumptions"]), name


def test_missing_database_is_a_structured_error(monkeypatch):
    def not_built():
        raise FileNotFoundError("data/processed/airports.duckdb not found; run `uv run python -m sources.build` first")

    tools.set_context(None)
    monkeypatch.setattr(common, "load_tables", not_built)
    out = json.loads(BY_NAME["rank_airports"].invoke({}))
    assert out["valid_options"] == [] and "sources.build" in out["error"] and "not available" in out["error"]
