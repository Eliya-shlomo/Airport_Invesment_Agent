"""Full agent evals: golden questions, answer checks, latency and cost.

    uv run python -m evals.run                      # free: prints what would run; no model call
    uv run python -m evals.run --confirm            # PAID: asks every golden question (needs the user's OK)

Without --confirm no model call is made.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import sys
import time
from typing import Callable

from evals import checks
from evals.questions import QUESTIONS, Q

PRICES = {"gpt-5.4-mini": (0.75, 4.50), "gpt-6.1-sol": (2.00, 10.00)}  # USD per 1M input/output tokens (list prices)


def cost_usd(model: str, usage: dict) -> float:
    pin, pout = PRICES[model]
    return round(usage["input_tokens"] / 1e6 * pin + usage["output_tokens"] / 1e6 * pout, 5)


def score_answer(q: Q, answer, prior_results: list[dict] | None = None) -> dict:
    """All checks for one answered question. Pure: takes the AgentAnswer, returns plain data.

    `prior_results` are tool results of earlier turns in the same conversation: a follow-up answered from
    memory may use their numbers (system prompt rule 2), so grounding is checked against them too.
    """
    called = [t.name for t in answer.trace]
    results = [t.result for t in answer.trace]
    generic = {"confidence_stated": checks.confidence_stated(answer.text, results),
               "sources_cited": checks.sources_cited(answer.text, results)}
    mentions = checks.mentions(answer.text, q.must, q.must_not)
    row = {
        "id": q.id, "category": q.category, "conv": q.conv, "question": q.text,
        "tools": called, "tool_args": [t.arguments for t in answer.trace],
        "tool_ok": checks.tools_ok(q.tools, called),
        "tool_errors": [r["error"] for r in results if "error" in r],
        "ungrounded": checks.ungrounded(answer.text, (prior_results or []) + results),
        "generic": generic, "mentions": mentions, "llm_ok": answer.ok, "answer": answer.text,
    }
    failed = [f"generic:{k}" for k, v in generic.items() if v.startswith("fail")]
    if mentions:
        failed.append("mentions")
    if not row["tool_ok"]:
        failed.append("tool_choice")
    if row["ungrounded"]:
        failed.append("ungrounded_numbers")
    row["failed"] = failed
    row["passed"] = not failed
    return row


def run_questions(make_llm: Callable[[], object], model: str, questions: list[Q] = QUESTIONS,
                  clock=time.time) -> list[dict]:
    """Ask every question in order; one Agent (memory) per conversation id. Stops early if the provider fails.

    One model and no fallback, so a run measures exactly that model."""
    from agent.loop import Agent
    agents: dict[str, Agent] = {}
    seen: dict[str, list[dict]] = {}  # tool results per conversation
    llm = make_llm()
    rows = []
    for q in questions:
        agent = agents.setdefault(q.conv, Agent(llm))
        t0 = clock()
        answer = agent.ask(q.text)
        seconds = round(clock() - t0, 1)
        row = score_answer(q, answer, seen.setdefault(q.conv, []))
        seen[q.conv] += [t.result for t in answer.trace]
        row.update(seconds=seconds, **answer.usage)
        row["cost_usd"] = cost_usd(model, answer.usage) if model in PRICES else None
        rows.append(row)
        flag = "ok " if row["passed"] else "FAIL"
        print(f"  {flag} {q.id:14} tools={row['tools']} {seconds}s ${row['cost_usd']} {row['failed'] or ''}", flush=True)
        if not answer.ok:
            print(f"  provider error, stopping: {answer.text[:200]}")
            break
    return rows


def summarize(rows: list[dict]) -> dict:
    """Pass counts by category, latency and cost statistics, and the failures with reasons."""
    secs = sorted(r["seconds"] for r in rows)
    by_cat: dict[str, list[int]] = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[0] += r["passed"]
        c[1] += 1
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    return {
        "questions": len(rows), "passed": sum(r["passed"] for r in rows),
        "by_category": {k: f"{a}/{b}" for k, (a, b) in by_cat.items()},
        "tool_choice_ok": f"{sum(r['tool_ok'] for r in rows)}/{len(rows)}",
        "ungrounded_numbers": sum(len(r["ungrounded"]) for r in rows),
        "seconds": {"mean": round(statistics.mean(secs), 1), "median": round(statistics.median(secs), 1),
                    "p90": round(secs[min(len(secs) - 1, int(0.9 * len(secs)))], 1), "max": secs[-1]},
        "cost_usd": {"per_question_mean": round(statistics.mean(costs), 4) if costs else None,
                     "total": round(sum(costs), 4) if costs else None},
        "llm_calls": sum(r.get("llm_calls", 0) for r in rows),
        "failures": [{"id": r["id"], "failed": r["failed"]} for r in rows if not r["passed"]],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true", help="make the paid model calls")
    ap.add_argument("--only", help="comma-separated question ids (conversations still start fresh)")
    ap.add_argument("--out-dir", default="evals/results")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(".env")
    model = os.environ.get("LLM_MODEL") or "gpt-6.1-sol"
    effort = os.environ.get("LLM_REASONING_EFFORT") or "medium"
    questions = [q for q in QUESTIONS if not args.only or q.id in args.only.split(",")]
    print(f"{len(questions)} golden questions in {len({q.conv for q in questions})} conversations, "
          f"model {model}, reasoning effort {effort}.")
    if not args.confirm:
        print("No paid call has been made. Add --confirm to run them.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (checked by name only).")
        return 2
    from agent.llm import chat_model
    rows = run_questions(lambda: chat_model(model, os.environ["OPENAI_API_KEY"], effort), model, questions)
    summary = summarize(rows)
    today = datetime.date.today().isoformat()
    os.makedirs(args.out_dir, exist_ok=True)
    out = f"{args.out_dir}/eval_{today}_{model}_{effort}.json"
    if os.path.exists(out):  # never overwrite an earlier run of the same day
        out = out.replace(".json", f"_{datetime.datetime.now():%H%M%S}.json")
    with open(out, "w") as f:
        json.dump({"model": model, "effort": effort, "date": today, "summary": summary, "rows": rows}, f,
                  indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
