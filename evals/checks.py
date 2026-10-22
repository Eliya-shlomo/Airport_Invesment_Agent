"""Scorers for agent answers. Pure functions, tested offline (tests/test_evals.py).

Known limit: these are text checks. They catch rule breaks that show up in wording or numbers, not every
weak explanation, so flagged and passing answers are both read before the scores are trusted.
"""
from __future__ import annotations

import re

DATE_LIKE = re.compile(r"\b\d{4}-(?:\d{2}|Q\d)(?:-\d{2})?\b|CY\d{4}|\b(?:19|20)\d{2}\b"
                       r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b")
NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
# Numbers that may appear without a tool result: configured weights, thresholds and small counts of signals.
ALWAYS = {40, 30, 15, 12, 5, 2500, 1000, 66.7, 20, 4, 3, 2, 1, 0, 100, 10}
SOURCE_WORDS = ("BTS", "On-Time", "FAA", "DOT", "OurAirports")


def numbers_in(obj, acc: set) -> set:
    """Every number in a tool result (numeric values, and numbers inside strings except dates)."""
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.add(abs(float(obj)))
    elif isinstance(obj, str):
        for n in NUM.findall(DATE_LIKE.sub(" ", obj)):
            acc.add(float(n.replace(",", "")))
    elif isinstance(obj, dict):
        for v in obj.values():
            numbers_in(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            numbers_in(v, acc)
    return acc


def ungrounded(text: str, results: list[dict]) -> list[str]:
    """Numbers in the answer that appear in no tool result (rounding to the shown digits is allowed)."""
    allowed = numbers_in(results, set()) | ALWAYS
    allowed |= {a * 100 for a in list(allowed) if a < 1}  # weights written as percents
    bad = []
    for n in NUM.findall(DATE_LIKE.sub(" ", text)):
        v = float(n.replace(",", ""))
        if not any(abs(v - a) <= 0.051 or round(a) == v for a in allowed):
            bad.append(n)
    return bad


def tools_ok(expected: list[set] | None, called: list[str]) -> bool:
    """The set of tools called equals one of the allowed sets; None means tool choice is not checked."""
    return expected is None or set(called) in expected


def _clean(results: list[dict]) -> list[dict]:
    return [r for r in results if "error" not in r]


def confidence_stated(text: str, results: list[dict]) -> str:
    """The answer says "confidence" and one of the levels the tools returned."""
    levels = {r["confidence"] for r in _clean(results) if r.get("confidence")}
    if not levels:
        return "n/a"
    t = text.lower()
    return "pass" if "confidence" in t and any(level in t for level in levels) else "fail: confidence level not stated"


def sources_cited(text: str, results: list[dict]) -> str:
    """The answer names at least one dataset (by a keyword of its name) from the tools' sources."""
    names = " ".join(s.get("dataset", "") for r in _clean(results) for s in r.get("sources", []))
    wanted = [w for w in SOURCE_WORDS if w.lower() in names.lower()]
    if not wanted:
        return "n/a"
    return "pass" if any(w.lower() in text.lower() for w in wanted) else "fail: no source named"


def mentions(text: str, must: tuple[str, ...], must_not: tuple[str, ...]) -> list[str]:
    """The question's required patterns the answer misses and forbidden patterns it contains (case-insensitive)."""
    failures = [f"missing: {p}" for p in must if not re.search(p, text, re.IGNORECASE)]
    failures += [f"forbidden: {p}" for p in must_not if re.search(p, text, re.IGNORECASE)]
    return failures
