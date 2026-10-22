"""Expansion ranking tools: rank_airports, airport_scores (follow-ups) and top_airports_by (one metric).

The score and ranks are computed once in ToolContext (analytics/scoring.py); these tools select, explain and label.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from agent.tools.common import (
    AIRPORT_HELP, AIRSIDE_CAVEAT, CONGESTION_PROXY, COVERAGE_ONTIME, EXCESS_TAXI_OUT, Metric, Region, ToolContext,
    airport_tool, as_int, contract, coverage_warnings, get_context, growth_period, not_ranked_record,
    not_ranked_warnings, num, resolve_many, scoped,
)
from analytics import confidence, config, scoring

RegionArg = Annotated[Region | None, Field(description="Named region")]
StatesArg = Annotated[list[str] | None, Field(description="Two-letter US state codes")]
TopN = Annotated[int, Field(ge=1, le=50, description="How many airports to return")]
METRIC_HELP = "; ".join(f"{k}: {v[0]}" for k, v in config.RANK_METRICS.items())


@airport_tool
def rank_airports(region: RegionArg = None, states: StatesArg = None, top_n: TopN = 10) -> dict:
    """Rank US airports as expansion candidates (use it for terminal-expansion questions) by the expansion
    opportunity score, nationally, in a named region, or in given states. Returns each pillar and airports that
    could not be ranked, with reasons."""
    ctx = get_context()
    ranked_codes, not_ranked_codes, scope = scoped(ctx, region, states)
    ranked = ctx.scores.loc[ranked_codes].sort_values(["rank", "score"])
    out = _score_contract(ctx, list(ranked.index[:top_n]), not_ranked_codes,
                          {"scope": scope, "ranked_in_scope": len(ranked), "universe_size": len(ctx.universe)}, [])
    for i, rec in enumerate(out["result"]["ranked"], start=1):
        rec["rank_in_scope"] = i
    return out


@airport_tool
def airport_scores(airports: Annotated[list[str], Field(description=AIRPORT_HELP)]) -> dict:
    """Score breakdown (rank, pillars, contributions, rank range) for specific airports. Use for follow-ups such as
    'and Boston?' or 'why is it ranked second?'."""
    ctx = get_context()
    notes: list[str] = []
    codes = resolve_many(ctx, airports, notes, 1, 10)
    ranked = [c for c in codes if c in ctx.universe.index]
    not_ranked = [c for c in codes if c in ctx.not_ranked.index]
    missing = [c for c in codes if c not in ranked and c not in not_ranked]
    out = _score_contract(ctx, ranked, not_ranked, {"universe_size": len(ctx.universe)}, notes)
    out["warnings"] += [f"{c}: has On-Time data but no FAA enplanement record, so it cannot be scored" for c in missing]
    return out


@airport_tool
def top_airports_by(metric: Annotated[Metric, Field(description=METRIC_HELP)],
                    order: Literal["highest", "lowest"] = "highest", top_n: TopN = 10,
                    region: RegionArg = None, states: StatesArg = None) -> dict:
    """Order the ranked US airports by ONE measured metric (for example fastest-growing, busiest, most late
    departures, highest long-haul share), nationally, in a named region or in given states. Use it for 'top N by X'
    questions; never sort numbers yourself."""
    ctx = get_context()
    codes, not_ranked_codes, scope = scoped(ctx, region, states)
    label, unit = config.RANK_METRICS[metric]

    values = ctx.universe[METRIC_COLUMNS[metric]].loc[codes]
    ranked = scoring.rank_by_metric(values, descending=order == "highest")
    missing = sorted(values.index[values.isna()])
    shown = list(ranked.index[:top_n])

    records = []
    for code in shown:
        u, sc = ctx.universe.loc[code], ctx.scores.loc[code]
        value = ranked.loc[code, "value"]
        records.append({
            "rank_by_metric": int(ranked.loc[code, "rank"]), "airport": code, "name": u.airport_name, "state": u.state,
            "hub_class": u.hub_class, "value": as_int(value) if metric == "size" else num(value),
            "expansion_score": num(sc.score), "rank_national": int(sc["rank"]),
        })
    warnings = [w for code in shown for w in coverage_warnings(ctx, code)]
    warnings += not_ranked_warnings(ctx, not_ranked_codes)
    if missing:
        warnings.append(f"{', '.join(missing)}: no value for this metric, so not ordered (see missing_metric)")
    if metric == "long_haul_share":
        warnings.append("International departures are not in BTS On-Time data, so shares cover domestic flights only")
    result = {
        "metric": metric, "metric_label": label, "unit": unit, "order": order, "scope": scope,
        "ranked_in_scope": len(ranked), "universe_size": len(ctx.universe), "ranked": records,
        "missing_metric": missing, "not_ranked": [not_ranked_record(ctx, c) for c in not_ranked_codes],
    }
    method = (f"Ranked airports ordered by {label.lower()} ({unit}), {order} first; ties share the better rank. "
              "Values are computed from the data, not by the model. 'expansion_score' and 'rank_national' are the "
              "airport's overall opportunity score and rank, for context.")
    assumptions = [
        f"Only the {len(ctx.universe)} ranked airports are ordered: FAA large, medium and small hubs with at least "
        f"{config.MIN_FLOWN_DEPARTURES:,} reporting-carrier departures in the window; smaller airports have "
        "extreme growth percentages on tiny counts, and hubs outside the set are listed in not_ranked with their own values",
        f"{growth_period(ctx)}; size is CY{ctx.latest_year} enplanements",
        f"Late means {config.LATE_DEPARTURE_MINUTES}+ minutes after schedule; long haul means "
        f"{config.LONG_HAUL_MIN_MILES:,} miles or more",
        EXCESS_TAXI_OUT,
        "A high value of a delay metric is not by itself a case for expansion; see the expansion ranking for the combined view",
    ]
    return contract(result, method, assumptions, [ctx.sources["ontime"], ctx.sources["faa_enplanements"]],
                    COVERAGE_ONTIME, _metric_confidence(ctx, metric, shown), warnings)


# ---------------------------------------------------------------- score records


def _ranked_record(ctx: ToolContext, code: str) -> dict:
    u, s = ctx.universe.loc[code], ctx.scores.loc[code]
    w = config.SCORE_WEIGHTS
    return {
        "airport": code, "name": u.airport_name, "city": u.city, "state": u.state, "hub_class": u.hub_class,
        "rank_national": int(s["rank"]), "ranked_airports_national": len(ctx.universe),
        "score": num(s.score),
        "rank_range": {"min": int(s.rank_min), "max": int(s.rank_max)},
        "components": {
            "congestion": {
                "score": num(s.congestion), "weight": w["congestion"], "points": num(w["congestion"] * s.congestion),
                "pct_dep_15": num(u.pct_dep_15), "pct_dep_15_percentile": num(s.pct_dep_15_pctl),
                "excess_taxi_out_min": num(u.excess_taxi_out_min),
                "excess_taxi_out_percentile": num(s.excess_taxi_out_pctl),
            },
            "size": {
                "score": num(s["size"]), "weight": w["size"], "points": num(w["size"] * s["size"]),
                "enplanements": as_int(u.enplanements),
            },
            "growth": {
                "score": num(s.growth), "weight": w["growth"], "points": num(w["growth"] * s.growth),
                "growth_pct": num(u.growth_pct), "enplanements_prior_year": as_int(u.enplanements_prior),
            },
        },
    }


def _score_method() -> str:
    w = config.SCORE_WEIGHTS
    return (
        f"Expansion opportunity score (0-100) = {w['congestion']:.0%} congestion + {w['size']:.0%} size + "
        f"{w['growth']:.0%} growth. Each pillar is a percentile among ranked US airports (higher = stronger case). "
        f"Congestion = mean percentile of the share of departures {config.LATE_DEPARTURE_MINUTES}+ minutes late and "
        "excess taxi-out time; size = latest-year enplanements; growth = enplanement change against the year before. "
        "'points' is each pillar's contribution to the score; rank_range is the best and worst rank when each "
        "weight is changed by plus or minus 20%."
    )


def _score_assumptions(ctx: ToolContext) -> list[str]:
    return [
        "Weights are a judgement (congestion is weighted highest because it is the evidence that capacity binds); "
        "rank_range shows how much they matter",
        f"Ranked set: FAA large, medium and small hubs with at least {config.MIN_FLOWN_DEPARTURES:,} reporting-carrier "
        f"departures in the window ({len(ctx.universe)} airports); ranks are national, regions filter them",
        CONGESTION_PROXY,
        AIRSIDE_CAVEAT + "; size and growth are the passenger demand that terminals are sized for",
        EXCESS_TAXI_OUT,
        growth_period(ctx),
    ]


def _score_contract(ctx: ToolContext, ranked_codes: list[str], not_ranked_codes: list[str], extra: dict,
                    notes: list[str]) -> dict:
    warnings = [w for code in ranked_codes for w in coverage_warnings(ctx, code)]
    warnings += not_ranked_warnings(ctx, not_ranked_codes)
    level = confidence.lowest([confidence.RANKING] + (["low"] if any("months" in w for w in warnings) else []))
    result = {**extra, "ranked": [_ranked_record(ctx, c) for c in ranked_codes],
              "not_ranked": [not_ranked_record(ctx, c) for c in not_ranked_codes]}
    return contract(result, _score_method(), notes + _score_assumptions(ctx),
                    [ctx.sources["ontime"], ctx.sources["faa_enplanements"]], COVERAGE_ONTIME, level, warnings)


# ---------------------------------------------------------------- single metrics

METRIC_COLUMNS = {"growth": "growth_pct", "size": "enplanements", "late_departures": "pct_dep_15",
                  "excess_taxi_out": "excess_taxi_out_min", "mean_departure_delay": "mean_dep_delay_min",
                  "cancellation_rate": "cancellation_rate_pct", "long_haul_share": "long_haul_share_pct"}


def _metric_confidence(ctx: ToolContext, metric: str, codes: list[str]) -> str:
    """Growth covers one year (medium); excess taxi-out approximates an unimpeded time (a proxy); the rest are
    direct counts or shares (high with full On-Time coverage). Size is an FAA count (high)."""
    if metric == "growth":
        return confidence.RANKING
    if metric == "size":
        return "high"
    levels = [confidence.from_coverage(int(ctx.ops.loc[c].months), int(ctx.ops.loc[c].flown),
                                       proxy=metric == "excess_taxi_out") for c in codes if c in ctx.ops.index]
    return confidence.lowest(levels) if levels else "low"
