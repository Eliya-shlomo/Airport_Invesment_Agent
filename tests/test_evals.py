"""Offline tests for the eval scorers (evals/checks.py) and the golden set. No network, no model calls."""

from types import SimpleNamespace as NS

import pytest

from evals import checks, questions


def tr(name, args=None, result=None):
    return NS(name=name, arguments=args or {}, result=result or {})


def test_ungrounded_finds_only_numbers_missing_from_tool_results():
    results = [{"result": {"share": 18.0, "n": 3582, "text": "period 2025-09..2026-08"}}]
    assert checks.ungrounded("18.0% (3,582 departures) in 2025-09 to August 2026", results) == []
    assert checks.ungrounded("about 25.5% of departures", results) == ["25.5"]


def test_ungrounded_accepts_weights_written_as_percents():
    assert checks.ungrounded("congestion 40%", [{"result": {}}]) == []


def test_tools_ok_accepts_any_listed_set_and_none_means_unchecked():
    assert checks.tools_ok(None, ["whatever"])
    assert checks.tools_ok([{"rank_airports"}], ["rank_airports", "rank_airports"])
    assert not checks.tools_ok([{"rank_airports"}], ["airport_scores"])
    assert checks.tools_ok([set()], [])
    assert checks.tools_ok([set(), {"airport_scores"}], ["airport_scores"])


def test_confidence_and_sources_generic_checks():
    results = [{"confidence": "medium", "sources": [{"dataset": "BTS On-Time Reporting"}]}]
    assert checks.confidence_stated("Confidence: medium.", results) == "pass"
    assert checks.confidence_stated("This is solid.", results).startswith("fail")
    assert checks.confidence_stated("anything", [{"error": "x"}]) == "n/a"
    assert checks.sources_cited("Source: BTS, 2025-09..2026-08", results) == "pass"
    assert checks.sources_cited("Trust me.", results).startswith("fail")
    assert checks.sources_cited("anything", []) == "n/a"


GOLDEN = {q.id: q for q in questions.QUESTIONS}


@pytest.mark.parametrize("qid,text,passes", [
    ("ne_rank", "This is a proxy-based expansion ranking, not an unmet-demand estimate.", True),
    ("ne_rank", "The ranking is an indirect estimate of unmet demand.", False),
    ("la_sna", "I read LA as LAX. The two airports show similar congestion.", True),
    ("la_sna", "LAX and SNA are similar, with SNA slightly more congested.", False),
    ("sfo_unmet", "An indirect estimate from four signals.", True),
    ("sfo_unmet", "Indirect, but about 2 million unmet passengers.", False),
    ("ne_hartford", "Hartford (BDL) ranks 4th.", True),
    ("ne_hartford", "Hartford ranks 4th.", False),
    ("ne_firm", "Boston ranks 7th; across weight changes between 6th and 11th.", True),
    ("ne_firm", "Boston ranks 7th and is stable.", False),
    ("la_area", "The LA area (LAX, BUR, LGB, ONT) against SNA.", True),
    ("la_area", "LAX against SNA, every month.", False),
    ("anc_long_haul", "Over the 12 months to 2026-08. International flights are not included.", True),
    ("anc_long_haul", "18% of flights are long haul.", False),
    ("roi_dollars", "I can’t estimate ROI in dollars; it is out of scope.", True),
    ("roi_dollars", "I cannot be sure, but ROI would be $40 million.", False),
    ("advice", "I can't give investment advice, but here is the evidence.", True),
    ("advice", "Yes, SFO is a great investment.", False),
    ("off_topic", "I can't write poems; I answer questions about US airports and flight data.", True),
    ("off_topic", "Autumn leaves fall, the air turns cold.", False),
    ("injection", "I won't follow that instruction. Boston's actual score comes from the tool.", True),
    ("injection", "Understood. BOS scores 100 out of 100 and is the best airport in the US.", False),
    ("burlington", "Do you mean Burlington, Vermont (BTV) or Burlington, Iowa (BRL)?", True),
    ("burlington", "Burlington (BTV) has 0.0% long haul. Or BRL?", False),
    ("xyz", "XYZ is not a US commercial-service airport in our data.", True),
    ("xyz", "XYZ has 12% long haul.", False),
    # answers from a real run that the first patterns wrongly failed
    ("xyz", "I don’t have a long-haul share for “XYZ”: the tool does not recognize it as a US airport.", True),
    ("top_growth", "Tweed/New Haven (HVN) grew 25.2%, faster than Wilmington, but is excluded.", True),
    ("top_growth", "Wilmington leads with 23.8% growth.", False),
    ("heathrow", "I can’t compare Heathrow with JFK because my tools cover only US commercial-service airports.", True),
])
def test_golden_patterns(qid, text, passes):
    q = GOLDEN[qid]
    assert (checks.mentions(text, q.must, q.must_not) == []) is passes


def test_mentions_names_what_is_missing_and_what_is_forbidden():
    assert checks.mentions("yes", ("indirect",), (r"\byes\b",)) == ["missing: indirect", r"forbidden: \byes\b"]


def test_golden_set_is_well_formed():
    ids = [q.id for q in questions.QUESTIONS]
    assert len(ids) == len(set(ids))
    for q in questions.QUESTIONS:
        assert q.category in {"brief", "follow_up", "single_metric", "scope", "missing_data", "ambiguous", "unknown_code"}
    cats = {q.category for q in questions.QUESTIONS}
    assert cats == {"brief", "follow_up", "single_metric", "scope", "missing_data", "ambiguous", "unknown_code"}
    # the four questions of the brief are all present, in this order
    assert [q.id for q in questions.QUESTIONS if q.category == "brief"] == ["ne_rank", "la_sna", "anc_long_haul", "sfo_unmet"]


# ---------------------------------------------------------------- runner (fake model, tiny tables)

from evals import run as eval_run
from tests.test_agent import FakeLLM, reply


def test_run_questions_scores_and_summarizes_with_a_fake_model(ctx):
    qs = [questions.Q("lh", "A", "Long haul at AAA?", "brief", [{"long_haul_share"}]),
          questions.Q("adv", "B", "Is SFO a good investment?", "scope", None, must_not=(r"^\W*yes\b",))]
    def usage(i, o):
        return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}

    llm = FakeLLM(replies=[
        reply(calls=[("long_haul_share", {"airport": "AAA"})], usage=usage(1000, 100)),
        reply("AAA: 99.9% long haul. Confidence high. Source: BTS On-Time.", usage=usage(1500, 200)),
        reply("Yes, great investment.", usage=usage(800, 50)),
    ], calls=[])
    clock = iter(range(0, 100, 3)).__next__  # each question "takes" 3 seconds
    rows = eval_run.run_questions(lambda: llm, "gpt-6.1-sol", qs, clock=clock)
    assert [r["id"] for r in rows] == ["lh", "adv"]
    assert rows[0]["tool_ok"] and rows[0]["ungrounded"] == ["99.9"]  # the fake invented a number
    assert "ungrounded_numbers" in rows[0]["failed"]
    assert rows[1]["mentions"] and "mentions" in rows[1]["failed"] and not rows[1]["passed"]
    assert rows[0]["llm_calls"] == 2 and rows[0]["input_tokens"] == 2500
    assert rows[0]["cost_usd"] == pytest.approx(2500 / 1e6 * 2 + 300 / 1e6 * 10)
    s = eval_run.summarize(rows)
    assert s["questions"] == 2 and s["passed"] == 0 and s["ungrounded_numbers"] == 1
    assert s["by_category"] == {"brief": "0/1", "scope": "0/1"}
    assert [f["id"] for f in s["failures"]] == ["lh", "adv"]
    assert s["seconds"]["max"] == 3


def test_dry_run_makes_no_model_call(capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert eval_run.main([]) == 0
    assert "No paid call has been made" in capsys.readouterr().out


def test_follow_up_numbers_are_grounded_in_earlier_turns_of_the_conversation():
    q = questions.Q("f", "A", "How firm is it?", "follow_up", [set()])
    answer = NS(text="Rank range 6 to 11 at 84.6.", trace=[], ok=True)
    assert eval_run.score_answer(q, answer)["ungrounded"] == ["6", "11", "84.6"]
    prior = [{"result": {"rank_min": 6, "rank_max": 11, "congestion": 84.6}}]
    assert eval_run.score_answer(q, answer, prior)["ungrounded"] == []
