"""unmet_demand: indirect signals that demand exceeds capacity at one airport, with a level; never a passenger number."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from agent.tools.common import (
    AIRPORT_HELP, COVERAGE_AIRFARE, COVERAGE_ONTIME, ToolError, airport_name, airport_tool, as_int, contract,
    coverage_warnings, get_context, num, resolve_one,
)
from analytics import confidence, config, scoring

UNMET_DEMAND_CAVEAT = (
    "Unmet demand cannot be measured with public data (no bookings, seats or gate data); "
    "this is an indirect estimate from signals, not a number of passengers"
)


@airport_tool
def unmet_demand(airport: Annotated[str, Field(description=AIRPORT_HELP)]) -> dict:
    """Indirect estimate of unmet flight demand at one airport: which signals point to demand above capacity, and
    why. Never a passenger number."""
    ctx = get_context()
    notes: list[str] = []
    code = resolve_one(ctx, airport, notes)
    if code not in ctx.universe.index:
        reason = ctx.not_ranked.loc[code, "reason"] if code in ctx.not_ranked.index else "no FAA enplanement record"
        raise ToolError(f"{code} is outside the ranked set ({reason}), so its signals cannot be compared with peers.")
    u, s = ctx.universe.loc[code], ctx.scores.loc[code]
    official = ("slot-controlled" if code in config.FAA_SLOT_CONTROLLED
                else "schedule review" if code in config.FAA_SCHEDULE_REVIEW else None)
    af = ctx.airfare.loc[code] if code in ctx.airfare.index else None
    fare_ok = af is not None and bool(af.sufficient)
    a = scoring.unmet_demand_assessment(
        congestion_pctl=float(s.congestion), growth_pctl=float(s.growth),
        fare_pctl=float(af.fare_premium_pctl) if fare_ok else None,
        dominance_pctl=float(af.dominant_share_pctl) if fare_ok else None,
        official_status=official, high=config.SIGNAL_HIGH_PERCENTILE,
        strong_min=config.UNMET_STRONG_MIN_SIGNALS, moderate_min=config.UNMET_MODERATE_MIN_SIGNALS,
    )
    values = {
        "congestion": {"pct_dep_15": num(u.pct_dep_15), "excess_taxi_out_min": num(u.excess_taxi_out_min)},
        "growth": {"growth_pct": num(u.growth_pct), "enplanements": as_int(u.enplanements)},
        "fare_premium": {"fare_premium_pct": num(af.fare_premium_pct) if af is not None else None,
                         "dominant_share_pct": num(af.dominant_share_pct) if af is not None else None,
                         "airport_pairs": as_int(af.pairs) if af is not None else 0},
        "faa_constraint_status": {},
    }
    for sig in a["signals"]:
        sig.update(values[sig["name"]])
        for k in ("percentile", "dominant_share_percentile"):
            if k in sig:
                sig[k] = num(sig[k])
    result = {"airport": code, "name": airport_name(ctx, code), "estimate_type": "indirect",
              "level": a["level"], "signals_on": a["signals_on"], "signals_total": len(a["signals"]),
              "signals": a["signals"]}

    high = config.SIGNAL_HIGH_PERCENTILE
    method = (
        "Indirect estimate. Four signals, each 'on' when in the top third of ranked US airports "
        f"(percentile {high:.1f} or more): congestion, enplanement growth, fare premium over what distance predicts "
        "(counted only when the dominant carrier's share is not also in the top third), and official FAA constraint "
        f"status. Level: strong with {config.UNMET_STRONG_MIN_SIGNALS}+ signals on, moderate with "
        f"{config.UNMET_MODERATE_MIN_SIGNALS}, weak otherwise."
    )
    assumptions = notes + [
        "High fares together with one dominant carrier may reflect market power rather than scarce capacity, so that "
        "case is 'ambiguous' and not counted",
        f"Fare premium: passenger-weighted fit of log fare on log distance over all airport pairs in "
        f"{ctx.sources['airfare']['period']}; needs at least {config.MIN_AIRFARE_PAIRS} pairs",
        "FAA constraint status: slot-controlled JFK, LGA, DCA; schedule review ORD, LAX, EWR, SFO. Both reflect "
        "runway and airspace capacity, so this signal and congestion point to airfield limits, not terminal limits",
    ]
    warnings = [UNMET_DEMAND_CAVEAT] + coverage_warnings(ctx, code)
    if not fare_ok:
        warnings.append(f"{code}: fewer than {config.MIN_AIRFARE_PAIRS} airport pairs in DOT airfare data; "
                        "fare signal not assessed")
    sources = [ctx.sources["ontime"], ctx.sources["faa_enplanements"], ctx.sources["airfare"],
               ctx.sources["faa_constraints"]]
    coverage = f"{COVERAGE_ONTIME}. {COVERAGE_AIRFARE}"
    return contract(result, method, assumptions, sources, coverage, confidence.UNMET_DEMAND, warnings)
