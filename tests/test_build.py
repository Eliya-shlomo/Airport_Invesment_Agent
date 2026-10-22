"""Offline tests for sources/build.py with tiny hand-built fixtures (no network, no real data)."""

from __future__ import annotations

import json
import os
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import openpyxl
import pytest

from sources import build, config

# ---------------------------------------------------------------- fixtures

HEADER = [
    "FlightDate", "Reporting_Airline", "Origin", "OriginState", "Dest", "Cancelled", "CancellationCode",
    "DepDelayMinutes", "DepDel15", "TaxiOut", "Distance", "DistanceGroup", "CarrierDelay",
    "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]


def flight(day, carrier, origin, dest, cancelled=0, code="", delay="", del15="", taxi="",
           dist=500, group=2, causes=("", "", "", "", "")):
    """One CSV row as text, like BTS: floats written as '0.00', cancelled rows have empty delay/taxi."""
    return [day, carrier, origin, "XX", dest, f"{cancelled}.00", code, delay, del15, taxi,
            f"{dist}.00", str(group), *causes]


JAN = [
    # AAA, January: 7 scheduled, 2 cancelled, 5 flown (one flown row has no delay/taxi value).
    flight("2026-01-05", "UA", "AAA", "BBB", delay="0.00", del15="0.00", taxi="10.00"),
    flight("2026-01-06", "UA", "AAA", "BBB", delay="20.00", del15="1.00", taxi="20.00",
           causes=("15.00", "0.00", "5.00", "0.00", "0.00")),
    flight("2026-01-07", "DL", "AAA", "CCC", delay="40.00", del15="1.00", taxi="30.00",
           dist=2600, group=11, causes=("0.00", "30.00", "0.00", "0.00", "10.00")),
    flight("2026-01-08", "DL", "AAA", "CCC", delay="0.00", del15="0.00", taxi="40.00", dist=2600, group=11),
    flight("2026-01-09", "UA", "AAA", "BBB", cancelled=1, code="B"),
    flight("2026-01-10", "DL", "AAA", "CCC", cancelled=1, code="A", dist=2600, group=11),
    flight("2026-01-11", "UA", "AAA", "BBB"),
    # BBB, January: 3 scheduled, 2 cancelled, 1 flown.
    flight("2026-01-05", "UA", "BBB", "AAA", delay="5.00", del15="0.00", taxi="8.00"),
    flight("2026-01-06", "UA", "BBB", "AAA", cancelled=1, code="C"),
    flight("2026-01-07", "UA", "BBB", "AAA", cancelled=1, code="D"),
]
FEB = [
    flight("2026-02-02", "UA", "AAA", "BBB", delay="30.00", del15="1.00", taxi="12.00",
           causes=("20.00", "10.00", "0.00", "0.00", "0.00")),
    flight("2026-02-03", "UA", "AAA", "BBB", delay="10.00", del15="0.00", taxi="14.00"),
]


def write_ontime_zip(path: Path, rows: list[list[str]]) -> None:
    """A zip like BTS's: one CSV (header and every line end with a trailing comma) plus readme.html."""
    lines = [",".join(HEADER) + ","] + [",".join(r) + "," for r in rows]
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2026_1.csv", "\n".join(lines) + "\n")
        zf.writestr("readme.html", "<html></html>")


def write_faa(path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "CY 25 Primary Airports "  # trailing space, like the real file
    ws.append(["Rank", "RO", "ST", "Locid", "City", "Airport Name", "S/L", "Hub",
               "CY 25 Enplanements", "CY 24 Enplanements", "% Change"])
    ws.append([1, "WP", "CA", "AAA", "Aville", "Aville International", "P", "L", 1000, 800, 0.25])
    ws.append([2, "NE", "MA", "BBB", "Bville", "Bville Regional", "P", "S", 500, None, None])
    ws.append([3, "AL", "AK", "1G4", "Peach", "Peach Springs", "P", "N", 40, 50, -0.2])
    ws.append([None, None, None, 114, "Nonprimary Commercial Service", None, None, None, None, None, None])
    ws.append([None, None, None, 402, "Total Primary Airports", None, None, None, 1540, 850, None])
    wb.save(path)


OA_AIRPORTS = """ident,type,name,latitude_deg,longitude_deg,elevation_ft,iso_country,iso_region,municipality,scheduled_service,iata_code
KAAA,large_airport,Aville International,37.6,-122.4,13,US,US-CA,Aville,yes,AAA
KBBB,small_airport,Bville Regional,42.3,-71.0,20,US,US-MA,Bville,no,BBB
KCCC,small_airport,No IATA Field,30.0,-97.0,5,US,US-TX,Cville,no,
KDDD,medium_airport,Closed Runways Field,40.0,-74.0,5,US,US-NY,Dville,yes,DDD
EGLL,large_airport,London Heathrow,51.4,-0.4,83,GB,GB-ENG,London,yes,LHR
"""
OA_RUNWAYS = """id,airport_ref,airport_ident,length_ft,width_ft,surface,lighted,closed
1,1,KAAA,11870,200,ASP,1,0
2,1,KAAA,7500,150,ASP,1,0
3,1,KAAA,12000,150,ASP,1,1
4,4,KDDD,5000,100,ASP,1,1
"""

AIRFARE = [
    {"year": "2025", "quarter": "4", "airport_1": "AAA", "airport_2": "BBB", "nsmiles": "500",
     "passengers": "85.3", "fare": "199.5", "carrier_lg": "UA", "large_ms": "0.61", "fare_lg": "205.1",
     "carrier_low": "DL", "lf_ms": "0.2", "fare_low": "150"},
    # carrier_low, lf_ms and fare_low are missing
    {"year": "2026", "quarter": "1", "airport_1": "CCC", "airport_2": "AAA", "nsmiles": "1200",
     "passengers": "10", "fare": "300", "carrier_lg": "AS", "large_ms": "0.9", "fare_lg": "310"},
]
SMALL = build.Thresholds(
    expected_months=2, required_airports=("AAA", "BBB"), min_enplanement_airports=2, min_airfare_rows=2,
)


def write_raw(raw: Path) -> None:
    for sub in ("ontime", "faa", "socrata", "ourairports"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    write_ontime_zip(raw / "ontime" / "ontime_2026_01.zip", JAN)
    write_ontime_zip(raw / "ontime" / "ontime_2026_02.zip", FEB)
    (raw / "ontime" / "ontime_2026_03.zip.part").write_bytes(b"partial download")
    write_faa(raw / "faa" / "arp-cy2025-commercial-service-enplanements.xlsx")
    (raw / "ourairports" / "airports.csv").write_text(OA_AIRPORTS)
    (raw / "ourairports" / "runways.csv").write_text(OA_RUNWAYS)
    (raw / "socrata" / "airfare_tfrh-tu9e.json").write_text(json.dumps(AIRFARE))


@pytest.fixture
def con():
    c = duckdb.connect()
    build.create_schema(c)
    yield c
    c.close()


@pytest.fixture
def raw(tmp_path):
    write_raw(tmp_path / "raw")
    return tmp_path / "raw"


def row(con, sql, params=()):
    return con.execute(sql, list(params)).fetchone()


# ---------------------------------------------------------------- On-Time

def load_two_months(con, tmp_path):
    raw = tmp_path / "raw"
    write_raw(raw)
    zips = build.find_ontime_zips(raw / "ontime")
    build.load_ontime(con, zips, tmp_path / "tmp")
    return zips


def test_find_ontime_zips_keeps_only_the_newest_window(tmp_path):
    for name in ("ontime_2025_12.zip", "ontime_2026_01.zip", "ontime_2026_02.zip"):
        (tmp_path / name).write_bytes(b"")
    assert [m for m, _ in build.find_ontime_zips(tmp_path, window=2)] == [date(2026, 1, 1), date(2026, 2, 1)]


def test_ops_monthly_counts_and_delays_for_aaa_january(con, tmp_path):
    load_two_months(con, tmp_path)
    r = con.execute(
        "SELECT scheduled, cancelled, flown, mean_dep_delay_min, pct_dep_15, mean_taxi_out_min, "
        "p10_taxi_out_min FROM ops_monthly WHERE origin='AAA' AND month=DATE '2026-01-01'"
    ).fetchone()
    # 7 rows, 2 cancelled, 5 flown. Delays over the 4 flown flights with a value: 0,20,40,0.
    assert r[:3] == (7, 2, 5)
    assert r[3] == pytest.approx(15.0)
    assert r[4] == pytest.approx(50.0)  # 2 of 4 have DepDel15 = 1
    assert r[5] == pytest.approx(25.0)  # taxi 10,20,30,40
    assert r[6] == pytest.approx(13.0)  # p10: 10 + 0.3 * (20 - 10)
    b = con.execute(
        "SELECT scheduled, cancelled, flown, mean_dep_delay_min, pct_dep_15, mean_taxi_out_min, "
        "p10_taxi_out_min FROM ops_monthly WHERE origin='BBB'"
    ).fetchone()
    assert b == (3, 2, 1, 5.0, 0.0, 8.0, 8.0)


def test_ops_monthly_second_month_and_month_is_first_day(con, tmp_path):
    load_two_months(con, tmp_path)
    r = con.execute(
        "SELECT scheduled, mean_dep_delay_min, pct_dep_15, mean_taxi_out_min, p10_taxi_out_min "
        "FROM ops_monthly WHERE origin='AAA' AND month=DATE '2026-02-01'"
    ).fetchone()
    assert r[0] == 2
    assert r[1:] == pytest.approx((20.0, 50.0, 13.0, 12.2))
    months = [m for (m,) in con.execute("SELECT DISTINCT month FROM ops_monthly ORDER BY 1").fetchall()]
    assert months == [date(2026, 1, 1), date(2026, 2, 1)]
    assert row(con, "SELECT COUNT(*) FROM ops_monthly")[0] == 3  # AAA x2 months + BBB x1


def test_distance_table_counts_flown_flights_only(con, tmp_path):
    load_two_months(con, tmp_path)
    dist = dict(con.execute(
        "SELECT distance_group, flights FROM ops_distance_monthly WHERE origin='AAA' AND month=DATE '2026-01-01'"
    ).fetchall())
    assert dist == {2: 3, 11: 2}  # the two cancelled flights are not counted


def test_airport_ops_weights_months_by_flown_flights(con):
    con.execute("""INSERT INTO ops_monthly VALUES
        ('AAA', DATE '2026-01-01', 105, 5, 100, 10.0, 20.0, 15.0, 10.0),
        ('AAA', DATE '2026-02-01', 300, 0, 300, 20.0, 40.0, 20.0, 12.0)""")
    build.create_summaries(con)
    r = con.execute("SELECT months, first_month, last_month, flown, pct_dep_15, excess_taxi_out_min, "
                    "mean_dep_delay_min, cancellation_rate_pct FROM airport_ops WHERE airport = 'AAA'").fetchone()
    assert r[:4] == (2, "2026-01", "2026-02", 400)
    assert r[4] == pytest.approx(35.0)  # (20*100 + 40*300) / 400
    assert r[5] == pytest.approx(7.25)  # ((15-10)*100 + (20-12)*300) / 400
    assert r[6] == pytest.approx(17.5)  # (10*100 + 20*300) / 400
    assert r[7] == pytest.approx(5 / 405 * 100)


def test_airport_enplanements_growth_and_missing_prior_year(con):
    con.execute("""INSERT INTO enplanements VALUES ('AAA', 2024, 100), ('AAA', 2025, 110), ('BBB', 2025, 50),
                   ('CCC', 2024, 0), ('CCC', 2025, 20)""")
    build.create_summaries(con)
    rows = {r[0]: r[1:] for r in con.execute(
        "SELECT airport, latest_year, enplanements, growth_pct FROM airport_enplanements").fetchall()}
    assert rows["AAA"][:2] == (2025, 110) and rows["AAA"][2] == pytest.approx(10.0)
    assert rows["BBB"][2] is None  # missing prior year: NULL, never 0
    assert rows["CCC"][2] is None  # zero base has no defined growth


# ---------------------------------------------------------------- FAA

def test_parse_faa_workbook_long_format_and_row_filter(tmp_path):
    write_faa(tmp_path / "arp-cy2025-commercial-service-enplanements.xlsx")
    airports, enp = build.parse_faa_workbook(tmp_path / "arp-cy2025-commercial-service-enplanements.xlsx")
    # Summary rows (numeric Locid) are skipped; an alphanumeric Locid like 1G4 is a real airport.
    assert [a[0] for a in airports] == ["AAA", "BBB", "1G4"]
    assert airports[0] == ("AAA", "Aville", "Aville International", "CA", "WP", "P", "L")
    # CY 25 -> 2025, CY 24 -> 2024; BBB has no 2024 value, so no row (not a zero).
    assert sorted(enp) == sorted([
        ("AAA", 2025, 1000), ("AAA", 2024, 800), ("BBB", 2025, 500), ("1G4", 2025, 40), ("1G4", 2024, 50),
    ])


# ---------------------------------------------------------------- OurAirports

def test_airports_filter_state_and_runways(con, raw):
    build.load_ourairports(con, raw / "ourairports" / "airports.csv", raw / "ourairports" / "runways.csv")
    rows = {r[0]: r for r in con.execute(
        "SELECT iata_code, ident, state, scheduled_service, runway_count, longest_runway_ft, municipality "
        "FROM airports ORDER BY 1"
    ).fetchall()}
    assert set(rows) == {"AAA", "BBB", "DDD"}  # no non-US, no airport without an IATA code
    assert rows["AAA"] == ("AAA", "KAAA", "CA", True, 2, 11870, "Aville")  # the closed 12000 ft runway is ignored
    assert rows["BBB"][2:6] == ("MA", False, None, None)  # no runway records: unknown, not zero
    assert rows["DDD"][4:6] == (0, None)  # only closed runways


# ---------------------------------------------------------------- Socrata

def test_airfare_numeric_types_and_missing_fields(con, raw):
    build.load_airfare(con, raw / "socrata" / "airfare_tfrh-tu9e.json")
    first = row(con, "SELECT year, quarter, nsmiles, passengers, fare, carrier_lg, large_ms, carrier_low, fare_low "
                     "FROM airfare_pairs WHERE airport_1='AAA'")
    assert first == (2025, 4, 500.0, 85.3, 199.5, "UA", 0.61, "DL", 150.0)
    second = row(con, "SELECT carrier_low, lf_ms, fare_low, fare_lg FROM airfare_pairs WHERE airport_1='CCC'")
    assert second == (None, None, None, 310.0)
    types = dict(con.execute("SELECT column_name, data_type FROM information_schema.columns "
                             "WHERE table_name='airfare_pairs'").fetchall())
    assert types["year"] == "INTEGER" and types["fare"] == "DOUBLE" and types["carrier_lg"] == "VARCHAR"


# ---------------------------------------------------------------- data_sources

def test_data_sources_periods_counts_and_fetched_at(tmp_path):
    raw = tmp_path / "raw"
    write_raw(raw)
    old = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc).timestamp()
    for f in (raw / "socrata").iterdir():
        os.utime(f, (old, old))
    db = tmp_path / "out" / "airports.duckdb"
    build.build(raw, db, tmp_path / "tmp", SMALL)
    with duckdb.connect(str(db), read_only=True) as con:
        ds = {r[0]: r for r in con.execute("SELECT * FROM data_sources").fetchall()}
    assert set(ds) == {"ontime", "faa_enplanements", "airfare", "ourairports"}
    # (dataset, description, url, fetched_at, period_start, period_end, row_count)
    assert ds["ontime"][2] == config.ONTIME_URL
    assert ds["ontime"][4:] == ("2026-01", "2026-02", 12)
    assert ds["faa_enplanements"][4:] == ("2024", "2025", 5)
    assert ds["airfare"][3:] == ("2026-10-01T12:00:00+00:00", "2025-Q4", "2026-Q1", 2)  # fetched = file mtime
    assert ds["ourairports"][4:] == (None, None, 3)


# ---------------------------------------------------------------- full build and validation

def test_full_build_creates_all_tables_atomically(tmp_path):
    raw = tmp_path / "raw"
    write_raw(raw)
    db = tmp_path / "out" / "airports.duckdb"
    counts = build.build(raw, db, tmp_path / "tmp", SMALL)
    assert set(counts) == set(build.TABLES)
    assert counts["ops_monthly"] == 3 and counts["enplanements"] == 5 and counts["data_sources"] == 4
    assert sorted(p.name for p in db.parent.iterdir()) == ["airports.duckdb"]  # no .new or .wal left
    assert list((tmp_path / "tmp").glob("*")) == []
    with duckdb.connect(str(db), read_only=True) as con:
        cols = [c for (c,) in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='ops_monthly' "
            "ORDER BY ordinal_position").fetchall()]
    assert cols[:3] == ["origin", "month", "scheduled"] and len(cols) == 9


def test_validation_failure_lists_all_problems_and_keeps_old_database(tmp_path):
    raw = tmp_path / "raw"
    write_raw(raw)
    db = tmp_path / "out" / "airports.duckdb"
    build.build(raw, db, tmp_path / "tmp", SMALL)
    before = db.read_bytes()

    strict = build.Thresholds(
        expected_months=12, required_airports=("AAA", "ZZZ"), min_enplanement_airports=300, min_airfare_rows=5000,
    )
    with pytest.raises(build.BuildError) as exc:
        build.build(raw, db, tmp_path / "tmp", strict)
    msg = str(exc.value)
    for expected in ("expected 12 months, found 2", "ZZZ", "FAA: 3 airports", "Airfare: 2 rows", "left untouched"):
        assert expected in msg
    assert db.read_bytes() == before
    assert sorted(p.name for p in db.parent.iterdir()) == ["airports.duckdb"]
    with duckdb.connect(str(db), read_only=True) as con:
        assert row(con, "SELECT COUNT(*) FROM ops_monthly")[0] == 3
