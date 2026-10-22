"""Ranked universe, expansion opportunity score, rank ranges, single-metric ranking and unmet-demand signals."""

from __future__ import annotations

import itertools

import pandas as pd

REASON_NONHUB = "nonhub airport (under 0.05% of US boardings); not in the ranked set"
REASON_NO_ONTIME = "no departures by BTS reporting carriers in the On-Time data, so congestion is unknown"
REASON_FEW_DEPARTURES = "too few reporting-carrier departures in the window for stable congestion figures"
REASON_NO_GROWTH = "enplanements missing for one of the two years, so growth is unknown"


def percentile_rank(values: pd.Series) -> pd.Series:
    """Percentile 0..100 within `values`: average rank / count * 100; ties share; NaN stays NaN."""
    return values.rank(method="average", pct=True) * 100


def build_universe(
    faa_airports: pd.DataFrame,
    ops: pd.DataFrame,
    enpl: pd.DataFrame,
    hub_classes: tuple[str, ...],
    min_flown: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split FAA airports into the ranked universe and `not_ranked` (with a reason).

    Both frames are indexed by airport code and carry name, state, hub class, the ops summary
    columns and the enplanement columns (NaN where a source has no data).
    """
    base = faa_airports.set_index("locid")[["city", "airport_name", "state", "hub_class"]]
    base.index.name = "airport"
    joined = base.join(enpl, how="left").join(ops, how="left")
    reason = pd.Series(pd.NA, index=joined.index, dtype="object")
    reason[joined["growth_pct"].isna()] = REASON_NO_GROWTH
    reason[joined["flown"].fillna(0) < min_flown] = REASON_FEW_DEPARTURES
    reason[joined["flown"].isna()] = REASON_NO_ONTIME
    reason[~joined["hub_class"].isin(hub_classes)] = REASON_NONHUB
    joined["reason"] = reason
    universe = joined[reason.isna()].drop(columns="reason")
    not_ranked = joined[reason.notna()]
    return universe, not_ranked


def _composite(pillars: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = sum(weights.values())
    return sum(pillars[k] * w for k, w in weights.items()) / total


def pillars(universe: pd.DataFrame) -> pd.DataFrame:
    """Indicator percentiles and the three pillar values (0..100) for the ranked universe."""
    p = pd.DataFrame(index=universe.index)
    p["pct_dep_15_pctl"] = percentile_rank(universe["pct_dep_15"])
    p["excess_taxi_out_pctl"] = percentile_rank(universe["excess_taxi_out_min"])
    p["congestion"] = (p["pct_dep_15_pctl"] + p["excess_taxi_out_pctl"]) / 2
    p["size"] = percentile_rank(universe["enplanements"])
    p["growth"] = percentile_rank(universe["growth_pct"])
    return p


def score_universe(universe: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Pillars, weighted score (0..100) and national rank (1 = best; ties share the better rank)."""
    p = pillars(universe)
    p["score"] = _composite(p, weights)
    p["rank"] = p["score"].rank(ascending=False, method="min").astype(int)
    return p


def rank_ranges(universe: pd.DataFrame, weights: dict[str, float], factors: tuple[float, ...]) -> pd.DataFrame:
    """Best and worst rank of each airport over every combination of weight factors."""
    p = pillars(universe)
    keys = list(weights)
    ranks = [
        _composite(p, {k: weights[k] * f for k, f in zip(keys, combo)}).rank(ascending=False, method="min")
        for combo in itertools.product(factors, repeat=len(keys))
    ]
    r = pd.concat(ranks, axis=1)
    return pd.DataFrame({"rank_min": r.min(axis=1).astype(int), "rank_max": r.max(axis=1).astype(int)})


def rank_by_metric(values: pd.Series, descending: bool = True) -> pd.DataFrame:
    """Airports with a value, best first, as columns `value` and `rank` (1 = best; ties share the better rank).

    Airports whose value is missing are not in the result; the caller reports them (never as zero).
    """
    v = values.dropna()
    out = pd.DataFrame({"value": v, "rank": v.rank(ascending=not descending, method="min").astype(int)})
    return out.sort_values(["rank"], kind="stable")


def unmet_demand_assessment(
    congestion_pctl: float,
    growth_pctl: float,
    fare_pctl: float | None,
    dominance_pctl: float | None,
    official_status: str | None,
    high: float,
    strong_min: int,
    moderate_min: int,
) -> dict:
    """Four indirect signals of demand above capacity and a level. Never a passenger number.

    A high fare premium counts only when the dominant-carrier share is not also high; otherwise it is
    "ambiguous" (may be market power). Missing airfare coverage is "insufficient data", not "off".
    """
    def on_off(flag: bool) -> str:
        return "on" if flag else "off"

    if fare_pctl is None or dominance_pctl is None:
        fare_status = "insufficient data"
    elif fare_pctl >= high and dominance_pctl >= high:
        fare_status = "ambiguous"
    else:
        fare_status = on_off(fare_pctl >= high)
    signals = [
        {"name": "congestion", "percentile": congestion_pctl, "status": on_off(congestion_pctl >= high)},
        {"name": "growth", "percentile": growth_pctl, "status": on_off(growth_pctl >= high)},
        {"name": "fare_premium", "percentile": fare_pctl, "dominant_share_percentile": dominance_pctl,
         "status": fare_status},
        {"name": "faa_constraint_status", "value": official_status, "status": on_off(official_status is not None)},
    ]
    n = sum(s["status"] == "on" for s in signals)
    level = "strong" if n >= strong_min else "moderate" if n >= moderate_min else "weak"
    return {"signals": signals, "signals_on": n, "level": level}
