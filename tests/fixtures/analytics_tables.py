"""A tiny hand-built set of processed tables for analytics and tool tests.

Ranked universe (FAA hub L/M/S, 1,000+ departures, both years): AAA, BBB, CCC, DDD.
Every month of an airport has the same values, so 12-month weighted figures equal the monthly ones.

| code | state | hub | pct_dep_15 | excess taxi | enpl 2025 | growth |
| AAA  | MA    | L   | 10         | 1           | 1,050     | +5%    |
| BBB  | MA    | L   | 20         | 3           | 3,960     | -1%    |
| CCC  | CA    | M   | 30         | 2           | 3,060     | +2%    |
| DDD  | CA    | S   | 40         | 4           | 2,200     | +10%   |
Not ranked: NNN (MA, nonhub), HHH (MA, hub S, no On-Time data), EEE (CA, hub M, 3 months, 300 departures).
"""

from datetime import date

import duckdb
import pandas as pd

from sources import build
from sources.tables import read_tables

MONTHS_12 = [date(2025, m, 1) for m in range(9, 13)] + [date(2026, m, 1) for m in range(1, 9)]

# code: (pct_dep_15, mean_taxi_out, p10_taxi_out, mean_dep_delay, months)
OPS = {
    "AAA": (10.0, 11.0, 10.0, 5.0, MONTHS_12),
    "BBB": (20.0, 13.0, 10.0, 8.0, MONTHS_12),
    "CCC": (30.0, 12.0, 10.0, 12.0, MONTHS_12),
    "DDD": (40.0, 14.0, 10.0, 15.0, MONTHS_12),
    "NNN": (25.0, 12.0, 10.0, 9.0, MONTHS_12),
    "EEE": (15.0, 12.0, 10.0, 7.0, MONTHS_12[-3:]),
}
FLOWN_PER_MONTH = 100
CANCELLED_PER_MONTH = 2

# code: (city, state, hub, enpl 2024, enpl 2025)
FAA = {
    "AAA": ("Alpha", "MA", "L", 1000, 1050),
    "BBB": ("Bravo", "MA", "L", 4000, 3960),
    "CCC": ("Charlie", "CA", "M", 3000, 3060),
    "DDD": ("Delta", "CA", "S", 2000, 2200),
    "NNN": ("November", "MA", "N", 500, 600),
    "HHH": ("Hotel", "MA", "S", 800, 900),
    "EEE": ("Echo", "CA", "M", 5000, 5100),
}


def ops_monthly() -> pd.DataFrame:
    rows = []
    for code, (pct15, taxi, p10, delay, months) in OPS.items():
        for m in months:
            rows.append({
                "origin": code, "month": m,
                "scheduled": FLOWN_PER_MONTH + CANCELLED_PER_MONTH, "cancelled": CANCELLED_PER_MONTH,
                "flown": FLOWN_PER_MONTH, "mean_dep_delay_min": delay, "pct_dep_15": pct15,
                "mean_taxi_out_min": taxi, "p10_taxi_out_min": p10,
            })
    return pd.DataFrame(rows)


def distance_monthly() -> pd.DataFrame:
    rows = []
    for m in MONTHS_12:
        rows.append({"origin": "AAA", "month": m, "distance_group": 1, "flights": 70})
        rows.append({"origin": "AAA", "month": m, "distance_group": 11, "flights": 30})
        rows.append({"origin": "BBB", "month": m, "distance_group": 3, "flights": 100})
    for m in MONTHS_12[-3:]:
        rows.append({"origin": "EEE", "month": m, "distance_group": 11, "flights": 100})
    return pd.DataFrame(rows)


def faa_airports() -> pd.DataFrame:
    return pd.DataFrame([
        {"locid": c, "city": city, "airport_name": f"{city} International", "state": st,
         "faa_region": "XX", "service_level": "P", "hub_class": hub}
        for c, (city, st, hub, _, _) in FAA.items()
    ])


def enplanements() -> pd.DataFrame:
    rows = []
    for c, (_, _, _, e24, e25) in FAA.items():
        rows.append({"locid": c, "year": 2024, "enplanements": e24})
        rows.append({"locid": c, "year": 2025, "enplanements": e25})
    return pd.DataFrame(rows)


def airfare_pairs() -> pd.DataFrame:
    """Few pairs: every airport has fewer than MIN_AIRFARE_PAIRS, so fare signals are insufficient."""
    rows = []
    for a, b, d in [("AAA", "BBB", 200), ("AAA", "CCC", 2600), ("BBB", "DDD", 2500)]:
        rows.append({"year": 2026, "quarter": 1, "airport_1": a, "airport_2": b, "nsmiles": d,
                     "passengers": 100.0, "fare": 10 * d ** 0.5, "carrier_lg": "UA", "large_ms": 0.6,
                     "fare_lg": 10 * d ** 0.5, "carrier_low": "WN", "lf_ms": 0.2, "fare_low": 8 * d ** 0.5})
    return pd.DataFrame(rows)


def airports() -> pd.DataFrame:
    return pd.DataFrame([
        {"ident": f"K{c}", "iata_code": c, "name": f"{FAA[c][0]} International", "municipality": FAA[c][0],
         "state": FAA[c][1], "runway_count": 2, "longest_runway_ft": 10000}
        for c in FAA
    ])


def data_sources() -> pd.DataFrame:
    return pd.DataFrame([
        {"dataset": "ontime", "fetched_at": "2026-10-22T07:24:51+00:00", "period_start": "2025-09", "period_end": "2026-08"},
        {"dataset": "faa_enplanements", "fetched_at": "2026-10-22T07:11:26+00:00", "period_start": "2024", "period_end": "2025"},
        {"dataset": "airfare", "fetched_at": "2026-10-22T07:25:16+00:00", "period_start": "2024-Q1", "period_end": "2026-Q1"},
        {"dataset": "ourairports", "fetched_at": "2026-10-22T07:17:25+00:00", "period_start": None, "period_end": None},
    ])


def tiny_tables() -> dict[str, pd.DataFrame]:
    """The fixture tables loaded into an in-memory database and summarized by the real build SQL."""
    raw = {"ops_monthly": ops_monthly(), "ops_distance_monthly": distance_monthly(), "faa_airports": faa_airports(),
           "enplanements": enplanements(), "airfare_pairs": airfare_pairs(), "airports": airports(),
           "data_sources": data_sources()}
    with duckdb.connect() as con:
        build.create_schema(con)
        for name, frame in raw.items():
            con.register("frame", frame)
            con.execute(f"INSERT INTO {name} BY NAME SELECT * FROM frame")
            con.unregister("frame")
        build.create_summaries(con)
        return read_tables(con)
