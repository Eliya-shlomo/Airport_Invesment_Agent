"""What every tool shares: the data loaded once (ToolContext), the `airport_tool` decorator, the result contract,
shared assumptions, and airport lookup.

Every tool returns {result, method, assumptions, sources, coverage, confidence, warnings} as compact JSON, or
{error, valid_options} when the input cannot be answered. Arguments are checked by the tool's typed signature
(LangChain sends validation errors back to the model); `ToolError` covers what types cannot express.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Literal

import pandas as pd
from langchain_core.tools import BaseTool, tool

from analytics import confidence, config, metrics, scoring
from sources.tables import load_tables

DATASET_NAMES = {
    "ontime": "BTS On-Time Reporting",
    "faa_enplanements": "FAA commercial-service enplanements",
    "airfare": "DOT Consumer Airfare Report, table 1a",
    "ourairports": "OurAirports (community data)",
}
COVERAGE_ONTIME = "BTS reporting carriers' domestic flights only; some regional and all cargo flights are missing"
COVERAGE_AIRFARE = "Domestic airport-pair markets in DOT table 1a; small airports have few or no pairs"

# Assumptions that several tools state.
CONGESTION_PROXY = "Congestion is a proxy from delays and taxi time; there is no gate, slot or seat data"
EXCESS_TAXI_OUT = (
    f"Excess taxi-out = monthly mean taxi-out minus the monthly {config.TAXI_BASELINE_PERCENTILE}th percentile, an "
    "approximation of FAA's unimpeded taxi time; in winter it may include de-icing"
)
# Congestion is measured on the airfield; the brief asks about terminals.
AIRSIDE_CAVEAT = (
    "Congestion is measured airside (late departures, taxi-out time): it shows the airport as a whole is short of "
    "capacity, not which part. Terminal capacity (gates, security, curb) is not in public data, so a runway-limited "
    "airport may need airfield rather than terminal investment"
)

# Argument types the model sees as enums; the values come from analytics/config.py.
Region = Literal[tuple(config.REGIONS)]
Metric = Literal[tuple(config.RANK_METRICS)]
AIRPORT_HELP = "IATA code (e.g. 'SFO') or place name ('LA', 'LA area', 'Santa Ana')"


class ToolError(Exception):
    """Input the tool cannot answer; becomes {error, valid_options} for the model."""

    def __init__(self, message: str, valid_options: list[str] | None = None):
        super().__init__(message)
        self.valid_options = valid_options or []


# ---------------------------------------------------------------- data, loaded once


@dataclass(frozen=True)
class ToolContext:
    """The processed tables plus everything precomputed once: ranked universe, scores, airfare signals."""

    faa: pd.DataFrame  # FAA airports, indexed by code
    ops: pd.DataFrame  # On-Time summary per airport (airport_ops) plus the long-haul share
    distance: pd.DataFrame  # departures per airport, month and BTS distance group
    runways: pd.DataFrame  # OurAirports, indexed by IATA code
    universe: pd.DataFrame
    not_ranked: pd.DataFrame
    scores: pd.DataFrame
    airfare: pd.DataFrame
    sources: dict[str, dict]
    latest_year: int  # of FAA enplanements; growth compares it with the year before

    @property
    def prior_year(self) -> int:
        return self.latest_year - 1

    @classmethod
    def from_tables(cls, t: dict[str, pd.DataFrame]) -> "ToolContext":
        ops = t["airport_ops"].set_index("airport")
        ops["long_haul_share_pct"] = metrics.long_haul_shares(t["ops_distance_monthly"],
                                                              config.LONG_HAUL_DISTANCE_GROUP)
        enpl = t["airport_enplanements"].set_index("airport")
        universe, not_ranked = scoring.build_universe(
            t["faa_airports"], ops, enpl, config.RANKED_HUB_CLASSES, config.MIN_FLOWN_DEPARTURES
        )
        scores = scoring.score_universe(universe, config.SCORE_WEIGHTS).join(
            scoring.rank_ranges(universe, config.SCORE_WEIGHTS, config.WEIGHT_SENSITIVITY_FACTORS)
        )
        airfare = metrics.airfare_signals(t["airfare_pairs"], config.AIRFARE_QUARTERS, config.MIN_AIRFARE_PAIRS)
        covered = airfare[airfare["sufficient"] & airfare.index.isin(universe.index)]
        airfare = airfare.assign(
            fare_premium_pctl=scoring.percentile_rank(covered["fare_premium_pct"]),
            dominant_share_pctl=scoring.percentile_rank(covered["dominant_share_pct"]),
        )
        latest_year = int(enpl["latest_year"].max())
        runways = t["airports"].dropna(subset=["iata_code"]).drop_duplicates("iata_code").set_index("iata_code")
        return cls(t["faa_airports"].set_index("locid"), ops, t["ops_distance_monthly"], runways, universe,
                   not_ranked, scores, airfare, _source_entries(t["data_sources"], latest_year, airfare),
                   latest_year)


def _source_entries(data_sources: pd.DataFrame, latest_year: int, airfare: pd.DataFrame) -> dict[str, dict]:
    ds = data_sources.set_index("dataset")

    def fetched(key: str) -> str:
        return str(ds.loc[key, "fetched_at"])[:10] if key in ds.index else "unknown"

    def period(key: str) -> str:
        if key not in ds.index or pd.isna(ds.loc[key, "period_start"]):
            return "snapshot"
        return f"{ds.loc[key, 'period_start']}..{ds.loc[key, 'period_end']}"

    return {
        "ontime": {"dataset": DATASET_NAMES["ontime"], "period": period("ontime"), "fetched": fetched("ontime")},
        "faa_enplanements": {
            "dataset": DATASET_NAMES["faa_enplanements"], "period": f"CY{latest_year - 1}..CY{latest_year}",
            "fetched": fetched("faa_enplanements"),
        },
        "airfare": {"dataset": DATASET_NAMES["airfare"], "period": airfare.attrs["period"], "fetched": fetched("airfare")},
        "ourairports": {"dataset": DATASET_NAMES["ourairports"], "period": "snapshot", "fetched": fetched("ourairports")},
        "faa_constraints": {
            "dataset": config.FAA_CONSTRAINT_SOURCE["dataset"], "period": "current list",
            "fetched": config.FAA_CONSTRAINT_SOURCE["fetched"],
        },
    }


_context: ToolContext | None = None


def get_context() -> ToolContext:
    """The data, loaded on the first tool call. A missing database raises a message that says how to build it."""
    global _context
    if _context is None:
        _context = ToolContext.from_tables(load_tables())
    return _context


def set_context(ctx: ToolContext | None) -> None:
    """Use other data (tests use small fixture tables); None loads the database again on the next call."""
    global _context
    _context = ctx


# ---------------------------------------------------------------- the tool decorator and the contract


def airport_tool(fn: Callable[..., dict]) -> BaseTool:
    """Make `fn` a LangChain tool. Its docstring is the description the model reads and its typed signature the
    argument schema. The result goes to the model as compact JSON; every failure comes back as {error,
    valid_options}, so the model can correct its arguments or tell the user, and nothing ends the conversation."""

    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> str:
        try:
            out = fn(*args, **kwargs)
        except ToolError as e:  # input the tool cannot answer: say what is valid
            out = {"error": str(e), "valid_options": e.valid_options}
        except FileNotFoundError as e:  # the database was never built
            out = {"error": f"The data is not available, so no number can be given: {e}", "valid_options": []}
        except Exception as e:  # noqa: BLE001 - a bug in one tool must not end the conversation
            out = {"error": f"{fn.__name__} failed: {type(e).__name__}: {e}", "valid_options": []}
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))

    return tool(run)


def contract(result, method, assumptions, sources, coverage, level, warnings) -> dict:
    return {"result": result, "method": method, "assumptions": assumptions, "sources": sources,
            "coverage": coverage, "confidence": level, "warnings": warnings}


def growth_period(ctx: ToolContext) -> str:
    return f"Growth covers one year (CY{ctx.prior_year} to CY{ctx.latest_year})"


def num(x: Any, digits: int = 1) -> float | None:
    """Round for display; missing stays None (JSON null), never 0."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return None
    return round(float(x), digits)


def as_int(x: Any) -> int | None:
    return None if x is None or pd.isna(x) else int(x)


def coverage_warnings(ctx: ToolContext, code: str) -> list[str]:
    """Partial or thin On-Time coverage for one airport, as warnings the answer must mention."""
    if code not in ctx.ops.index:
        return []
    o = ctx.ops.loc[code]
    return confidence.coverage_warnings(code, int(o.months), int(o.flown), o.first_month, o.last_month)


def not_ranked_warnings(ctx: ToolContext, not_ranked_codes: list[str]) -> list[str]:
    """One warning per hub airport outside the ranked set, and one count for nonhub airports."""
    nonhubs = [c for c in not_ranked_codes if ctx.not_ranked.loc[c, "reason"] == scoring.REASON_NONHUB]
    warnings = [f"{c}: not ranked, {ctx.not_ranked.loc[c, 'reason']}" for c in not_ranked_codes if c not in nonhubs]
    if nonhubs:
        warnings.append(f"{len(nonhubs)} nonhub airport(s) in scope are not ranked (see not_ranked)")
    return warnings


def not_ranked_record(ctx: ToolContext, code: str) -> dict:
    n = ctx.not_ranked.loc[code]
    return {
        "airport": code, "name": n.airport_name, "state": n.state, "hub_class": n.hub_class, "reason": n.reason,
        "enplanements": as_int(n.enplanements), "growth_pct": num(n.growth_pct), "departures": as_int(n.flown),
    }


# ---------------------------------------------------------------- airport lookup


def airport_name(ctx: ToolContext, code: str) -> str | None:
    return ctx.faa.loc[code, "airport_name"] if code in ctx.faa.index else None


def resolve(ctx: ToolContext, text: str, notes: list[str]) -> list[str]:
    """Airport code or place name -> codes. Alias choices are recorded in `notes` for the answer."""
    if not text.strip():
        raise ToolError("An airport must be a non-empty IATA code or place name, for example 'SFO' or 'LA'.")
    key = text.strip().lower()
    if key in config.PLACE_ALIASES:
        codes = list(config.PLACE_ALIASES[key])
        if key in ("la", "los angeles"):
            notes.append("'LA' was read as LAX only; ask for the 'LA area' to include BUR, LGB and ONT")
        elif key == "la area":
            notes.append("'LA area' means LAX, BUR, LGB and ONT")
        else:
            notes.append(f"'{text.strip()}' was read as " + ", ".join(f"{c} ({airport_name(ctx, c)})" for c in codes))
    else:
        codes = [text.strip().upper()]
    known = set(ctx.faa.index) | set(ctx.ops.index)
    for code in codes:
        if code not in known:
            hits = ctx.faa[ctx.faa["city"].str.lower().str.contains(key, regex=False, na=False)]
            same_name = hits[hits["city"].str.lower() == key]
            if len(same_name) == 1 and same_name["hub_class"].iloc[0] in config.RANKED_HUB_CLASSES:
                # The only airport in a US city of that name, and a hub ("Boston"): read it as that airport and say
                # so, since a foreign city can share the name ("Manchester"). Two airports ("Burlington") stay a
                # question for the user.
                found, r = same_name.index[0], same_name.iloc[0]
                notes.append(f"'{text.strip()}' was read as {found} ({r.city}, {r.state}), the only airport in a US "
                             "city of that name")
                return [found]
            options = [f"{c} ({r.city}, {r.state})" for c, r in hits.head(5).iterrows()]
            if options:  # a city name, not a code: offer the airports it may mean
                raise ToolError(f"'{text.strip()}' is not an airport code; it matches these US airports.", options)
            if len(code) == 3 and code.isalpha():
                raise ToolError(
                    f"{code} is not a US commercial-service airport in our data. The agent covers US airports only.",
                )
            raise ToolError(f"'{text.strip()}' matches no US airport code or city in our data. The agent covers US "
                            "airports only; the IATA code (for example 'BDL') always works.")
    return codes


def resolve_one(ctx: ToolContext, text: str, notes: list[str]) -> str:
    codes = resolve(ctx, text, notes)
    if len(codes) > 1:
        raise ToolError(f"'{text}' covers several airports; pick one.", codes)
    return codes[0]


def resolve_many(ctx: ToolContext, items: list[str], notes: list[str], lo: int, hi: int) -> list[str]:
    codes: list[str] = []
    for item in items:
        for code in resolve(ctx, item, notes):
            if code not in codes:
                codes.append(code)
    if not lo <= len(codes) <= hi:
        raise ToolError(f"Give between {lo} and {hi} airports; got {len(codes)}: {', '.join(codes)}.")
    return codes


def scoped(ctx: ToolContext, region: str | None, states: list[str] | None) -> tuple[list[str], list[str], str]:
    """Ranked and not-ranked airport codes in a scope (national, a named region or states), and the scope label.

    Not-ranked airports are ordered by size. Nationally only hubs that lack data are listed (nonhubs number hundreds).
    """
    if region is not None and states is not None:
        raise ToolError("Give either `region` or `states`, not both.")
    scope_states: tuple[str, ...] | None = None
    label = "national"
    if region is not None:
        scope_states, label = config.REGIONS[region], region
    elif states is not None:
        all_states = sorted(set(ctx.faa["state"].dropna()))
        scope_states = tuple(s.strip().upper() for s in states)
        if not scope_states:
            raise ToolError("`states` must be a non-empty list of two-letter US state codes.")
        bad = [s for s in scope_states if s not in all_states]
        if bad:
            raise ToolError(f"Unknown state code(s): {', '.join(bad)}. The agent covers US airports only.", all_states)
        label = "states " + ", ".join(scope_states)

    def inside(df: pd.DataFrame) -> pd.Series:
        return df["state"].isin(scope_states) if scope_states else df["state"].notna()

    ranked = list(ctx.universe.index[inside(ctx.universe)])
    not_ranked = ctx.not_ranked[inside(ctx.not_ranked)]
    if scope_states is None:
        not_ranked = not_ranked[not_ranked["reason"] != scoring.REASON_NONHUB]
    return ranked, list(not_ranked.sort_values("enplanements", ascending=False).index), label
