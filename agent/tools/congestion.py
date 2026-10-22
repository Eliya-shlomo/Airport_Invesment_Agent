"""compare_congestion: congestion indicators of 2 to 6 airports side by side, differences and verdict computed here."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from agent.tools.common import (
    AIRPORT_HELP, AIRSIDE_CAVEAT, CONGESTION_PROXY, COVERAGE_ONTIME, airport_name, airport_tool, as_int, contract,
    coverage_warnings, get_context, num, resolve_many,
)
from analytics import confidence, config


@airport_tool
def compare_congestion(airports: Annotated[list[str], Field(description=AIRPORT_HELP)]) -> dict:
    """Compare congestion indicators of 2 to 6 airports, with differences computed by the tool."""
    ctx = get_context()
    notes: list[str] = []
    codes = resolve_many(ctx, airports, notes, 2, 6)
    runways = ctx.runways
    rows, warnings, levels = [], [], []
    for code in codes:
        row: dict[str, Any] = {"airport": code, "name": airport_name(ctx, code)}
        if code not in ctx.ops.index:
            warnings.append(f"{code}: no BTS On-Time data, so congestion cannot be measured")
            levels.append("low")
            rows.append(row)
            continue
        o = ctx.ops.loc[code]
        row.update({
            "period": f"{o.first_month}..{o.last_month}", "months": int(o.months), "departures": int(o.flown),
            "pct_dep_15": num(o.pct_dep_15), "excess_taxi_out_min": num(o.excess_taxi_out_min),
            "mean_taxi_out_min": num(o.mean_taxi_out_min), "mean_dep_delay_min": num(o.mean_dep_delay_min),
            "cancellation_rate_pct": num(o.cancellation_rate_pct, 2),
            "runways": as_int(runways.loc[code, "runway_count"]) if code in runways.index else None,
        })
        if code in ctx.scores.index:
            s = ctx.scores.loc[code]
            row.update({"pct_dep_15_percentile": num(s.pct_dep_15_pctl),
                        "excess_taxi_out_percentile": num(s.excess_taxi_out_pctl),
                        "congestion_percentile": num(s.congestion)})
        else:
            warnings.append(f"{code}: outside the ranked set, so no percentiles ({ctx.not_ranked.loc[code, 'reason']})"
                            if code in ctx.not_ranked.index else f"{code}: outside the ranked set, so no percentiles")
        warnings += coverage_warnings(ctx, code)
        levels.append(confidence.from_coverage(int(o.months), int(o.flown), proxy=True))
        rows.append(row)

    result: dict[str, Any] = {"airports": rows}
    a, b = codes[0], codes[1]
    if len(codes) == 2 and a in ctx.ops.index and b in ctx.ops.index:
        oa, ob = ctx.ops.loc[a], ctx.ops.loc[b]
        diff = {
            "first": a, "second": b,
            "pct_dep_15_points": num(oa.pct_dep_15 - ob.pct_dep_15),
            "excess_taxi_out_min": num(oa.excess_taxi_out_min - ob.excess_taxi_out_min),
            "mean_dep_delay_min": num(oa.mean_dep_delay_min - ob.mean_dep_delay_min),
            "cancellation_rate_points": num(oa.cancellation_rate_pct - ob.cancellation_rate_pct, 2),
        }
        if a in ctx.scores.index and b in ctx.scores.index:
            gap = ctx.scores.loc[a, "congestion"] - ctx.scores.loc[b, "congestion"]
            diff["congestion_percentile_points"] = num(gap)
            result["verdict"] = (
                "similar congestion" if abs(gap) < config.SIMILAR_CONGESTION_POINTS
                else f"{a if gap > 0 else b} is more congested"
            )
        result["difference_first_minus_second"] = diff

    method = (
        f"Congestion indicators per departure airport over the On-Time window: share of operated departures "
        f"{config.LATE_DEPARTURE_MINUTES}+ minutes late, excess taxi-out (mean minus the monthly "
        f"{config.TAXI_BASELINE_PERCENTILE}th percentile), mean departure delay and cancellation rate. Percentiles are "
        "among ranked US airports; differences are computed by the tool."
    )
    assumptions = notes + [
        CONGESTION_PROXY,
        AIRSIDE_CAVEAT,
        "Cancellation rate is mostly weather-driven and is context, not part of the congestion score",
        "Runway counts come from OurAirports (community data) and include every listed runway",
        f"The verdict calls two airports similar when their congestion percentiles differ by less than "
        f"{config.SIMILAR_CONGESTION_POINTS:g} points",
    ]
    sources = [ctx.sources["ontime"], ctx.sources["ourairports"]]
    return contract(result, method, assumptions, sources, COVERAGE_ONTIME, confidence.lowest(levels), warnings)
