"""Per-airport metrics that are not plain sums. Pure functions: data frames in, data frames out.

Weighted delay, taxi and enplanement summaries are computed in SQL when the database is built (sources/build.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def long_haul_shares(distance_monthly: pd.DataFrame, long_haul_group: int) -> pd.Series:
    """Long-haul share (percent of operated departures in distance group `long_haul_group` or above) per airport.

    Airports with no distance rows are absent from the result (unknown, never 0).
    """
    total = distance_monthly.groupby("origin")["flights"].sum()
    long = distance_monthly[distance_monthly["distance_group"] >= long_haul_group].groupby("origin")["flights"].sum()
    return (long.reindex(total.index, fill_value=0) / total * 100).rename("long_haul_share_pct")


def airfare_signals(airfare_pairs: pd.DataFrame, quarters: int, min_pairs: int) -> pd.DataFrame:
    """Fare premium and dominant-carrier share per airport over the latest `quarters` quarters.

    Fare premium (percent): fit log(fare) on log(distance) by passenger-weighted least squares over
    all pairs; an airport's premium is exp(passenger-weighted mean residual of its pairs) - 1.
    Dominant share (percent): passenger-weighted mean of `large_ms`. An airport appears in either
    airport column, so both are used. `sufficient` is False below `min_pairs` pairs.
    `attrs["period"]` is the quarter range used, like "2025-Q2..2026-Q1".
    """
    d = airfare_pairs.dropna(subset=["nsmiles", "fare", "passengers"])
    d = d[(d["nsmiles"] > 0) & (d["fare"] > 0) & (d["passengers"] > 0)]
    yq = d["year"].astype(int) * 10 + d["quarter"].astype(int)
    used = sorted(yq.unique())[-quarters:]
    d = d[yq.isin(used)]
    x, y, w = np.log(d["nsmiles"]), np.log(d["fare"]), d["passengers"]
    slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
    d = d.assign(resid=y - (intercept + slope * x))
    both = pd.concat([d.assign(airport=d["airport_1"]), d.assign(airport=d["airport_2"])])
    both = both.assign(w_resid=both["resid"] * both["passengers"], w_ms=both["large_ms"] * both["passengers"])
    g = both.groupby("airport")
    s = g.agg(pairs=("resid", "size"), pax=("passengers", "sum"), w_resid=("w_resid", "sum"), w_ms=("w_ms", "sum"))
    s["fare_premium_pct"] = (np.exp(s.pop("w_resid") / s["pax"]) - 1) * 100
    s["dominant_share_pct"] = s.pop("w_ms") / s.pop("pax") * 100
    s["sufficient"] = s["pairs"] >= min_pairs
    label = lambda q: f"{q // 10}-Q{q % 10}"  # noqa: E731
    s.attrs.update(period=f"{label(used[0])}..{label(used[-1])}", slope=float(slope))
    return s
