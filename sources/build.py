"""Build the processed DuckDB database from the raw files.

    uv run python -m sources.build

The database is written to a temporary file next to the final path, validated, and only then
swapped in atomically. If validation fails, the old database stays untouched.
"""

from __future__ import annotations

import contextlib
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

import duckdb
import openpyxl

from sources import config


class BuildError(Exception):
    """The build cannot produce a trustworthy database; the message lists every failure."""


@dataclass(frozen=True)
class Thresholds:
    """Validation thresholds. Parameters so that tests can use tiny fixtures."""

    expected_months: int = config.ONTIME_WINDOW_MONTHS
    required_airports: tuple[str, ...] = ("SFO", "LAX", "SNA", "BOS", "ANC", "JFK")  # the brief's airports
    min_enplanement_airports: int = 300
    min_airfare_rows: int = 5_000


# ---------------------------------------------------------------- schema

SCHEMA = """
CREATE TABLE airports (
    ident VARCHAR, iata_code VARCHAR, name VARCHAR, municipality VARCHAR, state VARCHAR,
    latitude_deg DOUBLE, longitude_deg DOUBLE, type VARCHAR, scheduled_service BOOLEAN,
    runway_count INTEGER, longest_runway_ft INTEGER
);
CREATE TABLE faa_airports (
    locid VARCHAR, city VARCHAR, airport_name VARCHAR, state VARCHAR,
    faa_region VARCHAR, service_level VARCHAR, hub_class VARCHAR
);
CREATE TABLE enplanements (locid VARCHAR, year INTEGER, enplanements BIGINT);
CREATE TABLE ops_monthly (
    origin VARCHAR, month DATE, scheduled INTEGER, cancelled INTEGER, flown INTEGER,
    mean_dep_delay_min DOUBLE, pct_dep_15 DOUBLE, mean_taxi_out_min DOUBLE, p10_taxi_out_min DOUBLE
);
CREATE TABLE ops_distance_monthly (
    origin VARCHAR, month DATE, distance_group INTEGER, flights INTEGER
);
CREATE TABLE airfare_pairs (
    year INTEGER, quarter INTEGER, airport_1 VARCHAR, airport_2 VARCHAR, nsmiles DOUBLE,
    passengers DOUBLE, fare DOUBLE, carrier_lg VARCHAR, large_ms DOUBLE, fare_lg DOUBLE,
    carrier_low VARCHAR, lf_ms DOUBLE, fare_low DOUBLE
);
CREATE TABLE data_sources (
    dataset VARCHAR, description VARCHAR, url VARCHAR, fetched_at VARCHAR,
    period_start VARCHAR, period_end VARCHAR, row_count BIGINT
);
"""

TABLES = ("airports", "faa_airports", "enplanements", "ops_monthly", "ops_distance_monthly", "airfare_pairs",
          "data_sources", "airport_ops", "airport_enplanements")


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA)


# ---------------------------------------------------------------- raw file lookup

_ONTIME_ZIP = re.compile(r"^ontime_(\d{4})_(\d{2})\.zip$")  # ignores *.part files
_FAA_XLSX = re.compile(r"^arp-cy(\d{4})-commercial-service-enplanements\.xlsx$")


def find_ontime_zips(ontime_dir: Path, window: int = config.ONTIME_WINDOW_MONTHS) -> list[tuple[date, Path]]:
    """The newest `window` complete On-Time zips as (first day of month, path), oldest first.

    Older months left on disk by earlier downloads are ignored.
    """
    found = []
    for path in ontime_dir.glob("ontime_*.zip"):
        m = _ONTIME_ZIP.match(path.name)
        if m:
            found.append((date(int(m[1]), int(m[2]), 1), path))
    return sorted(found)[-window:]


def find_faa_workbook(faa_dir: Path) -> Path | None:
    """The workbook of the latest calendar year, or None."""
    found = sorted(
        (int(m[1]), p) for p in faa_dir.glob("arp-cy*.xlsx") if (m := _FAA_XLSX.match(p.name))
    )
    return found[-1][1] if found else None


# ---------------------------------------------------------------- On-Time

ONTIME_COLUMNS = ("FlightDate", "Origin", "Cancelled", "DepDelayMinutes", "DepDel15", "TaxiOut", "DistanceGroup")


@contextlib.contextmanager
def extracted_csv(zip_path: Path, tmp_dir: Path) -> Iterator[Path]:
    """Extract the zip's CSV into tmp_dir and delete it afterwards (the CSV is ~276 MB)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise BuildError(f"{zip_path.name}: expected exactly one CSV, found {members}")
        target = tmp_dir / f"{zip_path.stem}.csv"
        with zf.open(members[0]) as src, open(target, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    try:
        yield target
    finally:
        target.unlink(missing_ok=True)


def stage_flights(con: duckdb.DuckDBPyConnection, csv_path: Path) -> int:
    """Load one month's CSV into the temp table `flights` with typed columns; return row count.

    Everything is read as text and cast with TRY_CAST, so one bad value becomes NULL instead
    of failing the build. The file has a trailing comma on every line; the extra empty column
    is simply not selected.
    """
    con.execute("DROP TABLE IF EXISTS flights")
    cols = ", ".join(ONTIME_COLUMNS)
    con.execute(
        f"""
        CREATE TEMP TABLE flights AS
        SELECT
            DATE_TRUNC('month', TRY_CAST(FlightDate AS DATE))::DATE AS month,
            Origin AS origin,
            TRY_CAST(Cancelled AS DOUBLE) AS cancelled,
            TRY_CAST(DepDelayMinutes AS DOUBLE) AS dep_delay,
            TRY_CAST(DepDel15 AS DOUBLE) AS dep_del15,
            TRY_CAST(TaxiOut AS DOUBLE) AS taxi_out,
            TRY_CAST(DistanceGroup AS DOUBLE)::INTEGER AS distance_group
        FROM (
            SELECT {cols}
            FROM read_csv(?, header=true, all_varchar=true, ignore_errors=true)
        )
        WHERE TRY_CAST(FlightDate AS DATE) IS NOT NULL AND COALESCE(Origin, '') <> ''
        """,
        [str(csv_path)],
    )
    return con.execute("SELECT COUNT(*) FROM flights").fetchone()[0]


def aggregate_flights(con: duckdb.DuckDBPyConnection) -> None:
    """Aggregate the staged `flights` table into ops_monthly and ops_distance_monthly.

    Flown = Cancelled is 0. Delay and taxi statistics use flown flights with a value
    (cancelled flights have no delay or taxi time).
    """
    con.execute(
        """
        INSERT INTO ops_monthly
        SELECT
            origin, month,
            COUNT(*) AS scheduled,
            COUNT(*) FILTER (WHERE cancelled = 1) AS cancelled,
            COUNT(*) FILTER (WHERE cancelled = 0) AS flown,
            AVG(dep_delay) FILTER (WHERE cancelled = 0) AS mean_dep_delay_min,
            100.0 * COUNT(*) FILTER (WHERE cancelled = 0 AND dep_delay IS NOT NULL AND dep_del15 = 1)
                / NULLIF(COUNT(*) FILTER (WHERE cancelled = 0 AND dep_delay IS NOT NULL), 0)
                AS pct_dep_15,
            AVG(taxi_out) FILTER (WHERE cancelled = 0) AS mean_taxi_out_min,
            QUANTILE_CONT(taxi_out, 0.1) FILTER (WHERE cancelled = 0) AS p10_taxi_out_min
        FROM flights
        GROUP BY origin, month
        """
    )
    con.execute(
        """
        INSERT INTO ops_distance_monthly
        SELECT origin, month, distance_group, COUNT(*)
        FROM flights WHERE cancelled = 0
        GROUP BY origin, month, distance_group
        """
    )


def load_ontime(con: duckdb.DuckDBPyConnection, zips: list[tuple[date, Path]], tmp_dir: Path) -> None:
    """Process the zips one at a time: extract, stage, aggregate, delete the extract."""
    for _month, zip_path in zips:
        with extracted_csv(zip_path, tmp_dir) as csv_path:
            stage_flights(con, csv_path)
            aggregate_flights(con)
    con.execute("DROP TABLE IF EXISTS flights")


# ---------------------------------------------------------------- FAA

_ENPLANEMENTS_HEADER = re.compile(r"^CY\s*(\d{2})\s*Enplanements$", re.IGNORECASE)
_FAA_TEXT_COLUMNS = {
    "locid": "Locid", "city": "City", "airport_name": "Airport Name", "state": "ST",
    "faa_region": "RO", "service_level": "S/L", "hub_class": "Hub",
}


def parse_faa_workbook(path: Path) -> tuple[list[tuple], list[tuple]]:
    """Read the first sheet by position (its name has a trailing space).

    Returns (faa_airports rows, enplanements rows). Only rows with a 3-character text Locid are
    airports: the summary rows at the bottom ("Total Primary Airports", counts) have a number or
    no Locid. Locids may contain digits (for example 1G4, W33). Year columns are parsed from
    their headers ("CY 25 Enplanements" is 2025). Empty enplanement cells produce no row.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(wb.worksheets[0].iter_rows(values_only=True))
    finally:
        wb.close()
    header_idx = next(
        (i for i, r in enumerate(rows) if r and any(str(c).strip() == "Locid" for c in r if c)), None
    )
    if header_idx is None:
        raise BuildError(f"{path.name}: header row with 'Locid' not found")
    header = [str(c).strip() if c is not None else "" for c in rows[header_idx]]
    col = {name: i for i, name in enumerate(header)}
    missing = [h for h in _FAA_TEXT_COLUMNS.values() if h not in col]
    year_cols = {
        2000 + int(m[1]): i for i, h in enumerate(header) if (m := _ENPLANEMENTS_HEADER.match(h))
    }
    if missing or not year_cols:
        raise BuildError(f"{path.name}: unexpected header {header}; missing {missing}")

    def text(row, name):
        v = row[col[name]]
        return None if v is None else str(v).strip()

    airports, enplanements = [], []
    for row in rows[header_idx + 1:]:
        locid = row[col["Locid"]]
        if not (isinstance(locid, str) and re.fullmatch(r"[A-Z0-9]{3}", locid.strip())):
            continue
        locid = locid.strip()
        airports.append((
            locid, text(row, "City"), text(row, "Airport Name"), text(row, "ST"),
            text(row, "RO"), text(row, "S/L"), text(row, "Hub"),
        ))
        for year, i in sorted(year_cols.items()):
            value = row[i]
            if isinstance(value, (int, float)):
                enplanements.append((locid, year, int(value)))
    return airports, enplanements


def load_faa(con: duckdb.DuckDBPyConnection, workbook: Path) -> None:
    airports, enplanements = parse_faa_workbook(workbook)
    con.executemany("INSERT INTO faa_airports VALUES (?, ?, ?, ?, ?, ?, ?)", airports)
    con.executemany("INSERT INTO enplanements VALUES (?, ?, ?)", enplanements)


# ---------------------------------------------------------------- OurAirports

def load_ourairports(con: duckdb.DuckDBPyConnection, airports_csv: Path, runways_csv: Path) -> None:
    """US airports with an IATA code, plus open-runway count and longest open runway.

    An airport with no runway record at all gets NULL (unknown), not 0. If it has runway
    records but all are closed, runway_count is 0 and longest_runway_ft is NULL.
    """
    con.execute(
        """
        CREATE TEMP TABLE oa_runways AS
        SELECT airport_ident AS ident,
               COUNT(*) AS n_records,
               COUNT(*) FILTER (WHERE COALESCE(closed, '') <> '1') AS n_open,
               MAX(TRY_CAST(length_ft AS DOUBLE)::INTEGER) FILTER (WHERE COALESCE(closed, '') <> '1')
                   AS longest
        FROM read_csv(?, header=true, all_varchar=true)
        GROUP BY airport_ident
        """,
        [str(runways_csv)],
    )
    con.execute(
        """
        INSERT INTO airports
        SELECT a.ident, a.iata_code, a.name, NULLIF(a.municipality, ''),
               NULLIF(REGEXP_REPLACE(a.iso_region, '^US-', ''), ''),
               TRY_CAST(a.latitude_deg AS DOUBLE), TRY_CAST(a.longitude_deg AS DOUBLE),
               a.type, a.scheduled_service = 'yes',
               CASE WHEN r.n_records IS NULL THEN NULL ELSE r.n_open END,
               r.longest
        FROM read_csv(?, header=true, all_varchar=true) a
        LEFT JOIN oa_runways r ON r.ident = a.ident
        WHERE a.iso_country = 'US' AND COALESCE(a.iata_code, '') <> ''
        """,
        [str(airports_csv)],
    )
    con.execute("DROP TABLE oa_runways")


# ---------------------------------------------------------------- DOT airfare

AIRFARE_FIELDS = (
    "year", "quarter", "airport_1", "airport_2", "nsmiles", "passengers", "fare", "carrier_lg",
    "large_ms", "fare_lg", "carrier_low", "lf_ms", "fare_low",
)


def load_airfare(con: duckdb.DuckDBPyConnection, json_path: Path) -> None:
    """Socrata returns every value as text (and omits empty fields), so read all as VARCHAR and cast."""
    cols = ", ".join(f"'{f}': 'VARCHAR'" for f in AIRFARE_FIELDS)
    con.execute(
        f"""
        INSERT INTO airfare_pairs
        SELECT TRY_CAST(year AS DOUBLE)::INTEGER, TRY_CAST(quarter AS DOUBLE)::INTEGER,
               airport_1, airport_2, TRY_CAST(nsmiles AS DOUBLE), TRY_CAST(passengers AS DOUBLE),
               TRY_CAST(fare AS DOUBLE), carrier_lg, TRY_CAST(large_ms AS DOUBLE),
               TRY_CAST(fare_lg AS DOUBLE), carrier_low, TRY_CAST(lf_ms AS DOUBLE),
               TRY_CAST(fare_low AS DOUBLE)
        FROM read_json(?, format='array', columns={{{cols}}})
        """,
        [str(json_path)],
    )


# ---------------------------------------------------------------- per-airport summaries

SUMMARIES = """
-- One row per origin airport over the whole On-Time window. Monthly values are weighted by operated flights.
CREATE TABLE airport_ops AS
SELECT
    origin AS airport,
    COUNT(DISTINCT month) AS months,
    STRFTIME(MIN(month), '%Y-%m') AS first_month,
    STRFTIME(MAX(month), '%Y-%m') AS last_month,
    SUM(scheduled)::BIGINT AS scheduled, SUM(cancelled)::BIGINT AS cancelled, SUM(flown)::BIGINT AS flown,
    SUM(pct_dep_15 * flown) / NULLIF(SUM(flown), 0) AS pct_dep_15,
    SUM((mean_taxi_out_min - p10_taxi_out_min) * flown) / NULLIF(SUM(flown), 0) AS excess_taxi_out_min,
    SUM(mean_taxi_out_min * flown) / NULLIF(SUM(flown), 0) AS mean_taxi_out_min,
    SUM(mean_dep_delay_min * flown) / NULLIF(SUM(flown), 0) AS mean_dep_delay_min,
    100.0 * SUM(cancelled) / NULLIF(SUM(scheduled), 0) AS cancellation_rate_pct
FROM ops_monthly
GROUP BY origin;

-- One row per airport: the latest calendar year and the change against the year before.
-- Growth is NULL (never 0) when the prior year is missing or zero.
CREATE TABLE airport_enplanements AS
WITH latest AS (SELECT MAX(year) AS year FROM enplanements),
wide AS (
    SELECT e.locid AS airport, l.year AS latest_year,
           SUM(e.enplanements) FILTER (WHERE e.year = l.year)::BIGINT AS enplanements,
           SUM(e.enplanements) FILTER (WHERE e.year = l.year - 1)::BIGINT AS enplanements_prior
    FROM enplanements e, latest l
    GROUP BY e.locid, l.year
)
SELECT *, 100.0 * (enplanements / NULLIF(enplanements_prior, 0) - 1) AS growth_pct
FROM wide;
"""


def create_summaries(con: duckdb.DuckDBPyConnection) -> None:
    """Per-airport summary tables that the tools read, computed once here instead of at question time."""
    con.execute(SUMMARIES)


# ---------------------------------------------------------------- data_sources

DATASETS = {  # key: (description, url)
    "ontime": ("BTS On-Time Reporting Carrier On-Time Performance: monthly flight-level data, aggregated per "
               "origin airport (reporting carriers only)", config.ONTIME_URL),
    "faa_enplanements": ("FAA commercial service airports: calendar-year enplanements (passenger boardings)",
                         config.FAA_ENPLANEMENTS_URL),
    "airfare": ("DOT Consumer Airfare Report, table 1a: average fare and passengers per airport-pair market and "
                "quarter", f"{config.SOCRATA_BASE}/{config.AIRFARE_RESOURCE}.json"),
    "ourairports": ("OurAirports community data: airport names, codes, locations and runways (not an official "
                    "source)", config.OURAIRPORTS_BASE),
}


def _ym(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def dataset_periods(con: duckdb.DuckDBPyConnection) -> dict[str, tuple[str | None, str | None, int]]:
    """(period_start, period_end, row_count) per dataset, computed from the loaded tables.

    Period formats: ontime "YYYY-MM", airfare "YYYY-Qn", FAA "YYYY" (calendar year).
    row_count for ontime is the number of raw flight rows (sum of `scheduled`).
    """
    lo, hi, n = con.execute("SELECT MIN(month), MAX(month), COALESCE(SUM(scheduled), 0) FROM ops_monthly").fetchone()
    out = {"ontime": (lo and _ym(lo), hi and _ym(hi), int(n))}

    lo, hi, n = con.execute("SELECT MIN(year), MAX(year), COUNT(*) FROM enplanements").fetchone()
    out["faa_enplanements"] = (lo and str(lo), hi and str(hi), n)

    lo, hi, n = con.execute(
        "SELECT MIN(year * 10 + quarter), MAX(year * 10 + quarter), COUNT(*) FROM airfare_pairs"
    ).fetchone()
    quarter = lambda v: None if v is None else f"{v // 10}-Q{v % 10}"
    out["airfare"] = (quarter(lo), quarter(hi), n)

    out["ourairports"] = (None, None, con.execute("SELECT COUNT(*) FROM airports").fetchone()[0])  # a snapshot
    return out


def fetched_at(files: list[Path]) -> str | None:
    """The fetch date of a dataset: the newest modification time of its raw files (set by sources.download)."""
    mtimes = [f.stat().st_mtime for f in files if f.exists()]
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat(timespec="seconds")


def dataset_files(raw_dir: Path) -> dict[str, list[Path]]:
    """Raw files behind each dataset."""
    faa = find_faa_workbook(raw_dir / "faa")
    return {
        "ontime": [p for _, p in find_ontime_zips(raw_dir / "ontime")],
        "faa_enplanements": [faa] if faa else [],
        "airfare": [raw_dir / "socrata" / f"airfare_{config.AIRFARE_RESOURCE}.json"],
        "ourairports": [raw_dir / "ourairports" / name for name in config.OURAIRPORTS_FILES],
    }


def fill_data_sources(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    """One row per dataset: description, URL, fetch date, period and row count."""
    periods = dataset_periods(con)
    files = dataset_files(raw_dir)
    rows = [(key, description, url, fetched_at(files[key]), *periods[key])
            for key, (description, url) in DATASETS.items()]
    con.executemany("INSERT INTO data_sources VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


# ---------------------------------------------------------------- validation

def validate(con: duckdb.DuckDBPyConnection, t: Thresholds) -> list[str]:
    """Sanity checks; returns every failure (empty list means the data is fit to answer questions)."""
    failures: list[str] = []

    def count(sql: str) -> int:
        return con.execute(sql).fetchone()[0]

    months = count("SELECT COUNT(DISTINCT month) FROM ops_monthly")
    if months != t.expected_months:
        failures.append(f"On-Time: expected {t.expected_months} months, found {months}")
    present = {r[0] for r in con.execute("SELECT DISTINCT origin FROM ops_monthly").fetchall()}
    missing = [a for a in t.required_airports if a not in present]
    if missing:
        failures.append(f"On-Time: required airports missing: {', '.join(missing)}")
    n = count("SELECT COUNT(DISTINCT locid) FROM enplanements")
    if n < t.min_enplanement_airports:
        failures.append(f"FAA: {n} airports in enplanements, need at least {t.min_enplanement_airports}")
    n = count("SELECT COUNT(*) FROM airfare_pairs")
    if n < t.min_airfare_rows:
        failures.append(f"Airfare: {n} rows, need at least {t.min_airfare_rows}")
    if count("SELECT COUNT(*) FROM airports") == 0:
        failures.append("OurAirports: no US airports loaded")
    return failures


# ---------------------------------------------------------------- orchestration

def check_inputs(raw_dir: Path) -> list[str]:
    """List the raw inputs that are missing (all at once, so one run shows everything)."""
    files = dataset_files(raw_dir)
    missing = []
    if not files["ontime"]:
        missing.append(f"{raw_dir / 'ontime'}: no ontime_YYYY_MM.zip files")
    if not files["faa_enplanements"]:
        missing.append(f"{raw_dir / 'faa'}: no arp-cyYYYY-commercial-service-enplanements.xlsx")
    missing += [f"{p}: file not found" for p in files["airfare"] + files["ourairports"] if not p.exists()]
    return missing


def build(
    raw_dir: Path = config.RAW,
    db_path: Path = config.DB_PATH,
    tmp_dir: Path = config.TMP,
    thresholds: Thresholds = Thresholds(),
) -> dict[str, int]:
    """Build and validate the database, then atomically replace db_path. Returns rows per table.

    Raises BuildError (listing every failure) and leaves any existing db_path untouched.
    """
    missing = check_inputs(raw_dir)
    if missing:
        raise BuildError("missing raw inputs (run sources.download first):\n  " + "\n  ".join(missing))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    new_path = db_path.with_name(db_path.name + ".new")
    for stale in (new_path, new_path.with_name(new_path.name + ".wal")):
        stale.unlink(missing_ok=True)

    swapped = False
    con = duckdb.connect(str(new_path))
    try:
        create_schema(con)
        load_faa(con, find_faa_workbook(raw_dir / "faa"))
        load_ourairports(con, *dataset_files(raw_dir)["ourairports"])
        load_airfare(con, dataset_files(raw_dir)["airfare"][0])
        load_ontime(con, find_ontime_zips(raw_dir / "ontime"), tmp_dir)
        create_summaries(con)
        fill_data_sources(con, raw_dir)

        failures = validate(con, thresholds)
        if failures:
            raise BuildError(
                "validation failed, the existing database was left untouched:\n  " + "\n  ".join(failures)
            )
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
        con.close()
        new_path.replace(db_path)
        swapped = True
        return counts
    finally:
        if not swapped:
            with contextlib.suppress(Exception):
                con.close()
            for leftover in (new_path, new_path.with_name(new_path.name + ".wal")):
                leftover.unlink(missing_ok=True)


def main() -> int:
    started = time.monotonic()
    try:
        counts = build()
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started
    with duckdb.connect(str(config.DB_PATH), read_only=True) as con:
        months = con.execute("SELECT MIN(month), MAX(month), COUNT(DISTINCT month) FROM ops_monthly").fetchone()
    print(f"Built {config.DB_PATH} ({config.DB_PATH.stat().st_size / 1e6:.1f} MB) in {elapsed:.0f}s")
    for table, n in counts.items():
        print(f"  {table:22s}{n:>12,}")
    print(f"  On-Time months: {months[2]} ({months[0]:%Y-%m} .. {months[1]:%Y-%m})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
