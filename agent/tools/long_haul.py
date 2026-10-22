"""long_haul_share: share of an airport's departures of 2,500 miles or more, with the distance distribution."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from agent.tools.common import (
    AIRPORT_HELP, COVERAGE_ONTIME, ToolError, airport_name, airport_tool, contract, coverage_warnings, get_context,
    num, resolve_one,
)
from analytics import confidence, config


@airport_tool
def long_haul_share(airport: Annotated[str, Field(description=AIRPORT_HELP)]) -> dict:
    """Percentage of an airport's departures that are long haul (2,500 miles or more), with the distance
    distribution."""
    ctx = get_context()
    notes: list[str] = []
    code = resolve_one(ctx, airport, notes)
    flights = ctx.distance[ctx.distance["origin"] == code]
    if flights.empty or code not in ctx.ops.index:
        raise ToolError(f"{code} has no BTS On-Time data, so its flight distances are unknown.")
    o = ctx.ops.loc[code]
    by_group = flights.groupby("distance_group")["flights"].sum()
    total = int(by_group.sum())
    long_haul = int(by_group[by_group.index >= config.LONG_HAUL_DISTANCE_GROUP].sum())
    dist = [{"distance": _distance_label(g), "departures": int(n), "share_pct": num(n / total * 100)}
            for g, n in by_group.items()]
    result = {
        "airport": code, "name": airport_name(ctx, code),
        "long_haul_share_pct": num(long_haul / total * 100), "long_haul_departures": long_haul,
        "departures": total, "period": f"{o.first_month}..{o.last_month}", "months": int(o.months),
        "distance_distribution": dist,
    }
    method = (f"Departures of {config.LONG_HAUL_MIN_MILES:,} miles or more (BTS distance group "
              f"{config.LONG_HAUL_DISTANCE_GROUP}) divided by all operated departures, summed over the window")
    assumptions = notes + [
        f"Long haul means {config.LONG_HAUL_MIN_MILES:,} miles or more: there is no official US definition; this is "
        "close to the 4,000 km European long-haul band and matches the BTS distance groups (configurable)",
        "Twelve months are used because the share is seasonal",
    ]
    warnings = coverage_warnings(ctx, code) + [
        f"{code}: international departures are not in BTS On-Time data, so the share covers domestic flights only"
    ]
    return contract(result, method, assumptions, [ctx.sources["ontime"]], COVERAGE_ONTIME,
                    confidence.from_coverage(int(o.months), int(o.flown)), warnings)


def _distance_label(group: int) -> str:
    """BTS distance groups are 250-mile bands; the long-haul group is open-ended."""
    if group >= config.LONG_HAUL_DISTANCE_GROUP:
        return f"{config.LONG_HAUL_MIN_MILES:,}+ miles"
    lo, hi = (group - 1) * 250, group * 250 - 1
    return f"{lo:,}-{hi:,} miles"
