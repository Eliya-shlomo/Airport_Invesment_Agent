"""Paths, URLs and the data window. The single place that decides what we download and keep."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TMP = ROOT / "data" / "tmp"
DB_PATH = PROCESSED / "airports.duckdb"

USER_AGENT = "Mozilla/5.0 (airport-agent data refresh)"

# On-Time: a rolling window of whole months ending at the latest published month.
# 12 months cancels seasonality (summer thunderstorms, winter storms).
ONTIME_WINDOW_MONTHS = 12
ONTIME_URL = (
    "https://www.transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
# BTS publishes with a lag of roughly two months; probe back this far for the latest file.
ONTIME_MAX_LAG_MONTHS = 6

# FAA enplanements: one workbook per calendar year; it also holds the prior year.
FAA_ENPLANEMENTS_URL = (
    "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/"
    "arp-cy{year}-commercial-service-enplanements.xlsx"
)

# DOT Socrata dataset (no key). A row filter keeps the download small.
SOCRATA_BASE = "https://data.transportation.gov/resource"
AIRFARE_RESOURCE = "tfrh-tu9e"  # Consumer Airfare Report, Table 1a: all airport-pair markets
AIRFARE_YEARS_BACK = 2  # fetch from (current year - 2): about nine quarters
SOCRATA_PAGE_SIZE = 50_000

OURAIRPORTS_BASE = "https://davidmegginson.github.io/ourairports-data"
OURAIRPORTS_FILES = ("airports.csv", "runways.csv")
