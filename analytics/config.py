"""Every assumption behind a score, threshold or definition, in one place.

Scoring code reads these constants and never hard-codes its own. Tools return the relevant ones in `assumptions`.
"""

# --- Ranked universe ----------------------------------------------------------------------
RANKED_HUB_CLASSES: tuple[str, ...] = ("L", "M", "S")  # FAA hub classes: large, medium, small
MIN_FLOWN_DEPARTURES: int = 1_000  # operated departures over the On-Time window
FULL_WINDOW_MONTHS: int = 12  # fewer months than this is partial coverage

# --- Expansion opportunity score ----------------------------------------------------------
SCORE_WEIGHTS: dict[str, float] = {"congestion": 0.40, "size": 0.30, "growth": 0.30}
# Congestion pillar: equal-weight mean of the percentiles of these two indicators.
CONGESTION_INDICATORS: tuple[str, ...] = ("pct_dep_15", "excess_taxi_out_min")
LATE_DEPARTURE_MINUTES: int = 15  # BTS DepDel15: departure 15+ minutes after schedule
TAXI_BASELINE_PERCENTILE: int = 10  # monthly p10 taxi-out stands in for FAA's unimpeded taxi time
# Sensitivity: every combination of these factors on each weight, renormalized (rank range).
WEIGHT_SENSITIVITY_FACTORS: tuple[float, ...] = (0.8, 1.0, 1.2)
PERCENTILE_MAX: float = 100.0  # percentile = average rank / n * 100

# Congestion comparison: pillar gaps below this many percentile points are reported as "similar".
SIMILAR_CONGESTION_POINTS: float = 5.0

# --- Unmet demand signals -----------------------------------------------------------------
SIGNAL_HIGH_PERCENTILE: float = 200 / 3  # top third of the ranked universe (66.7)
AIRFARE_QUARTERS: int = 4  # latest quarters of DOT airfare table 1a
MIN_AIRFARE_PAIRS: int = 20  # fewer airport pairs: fare signals are "insufficient data"
UNMET_STRONG_MIN_SIGNALS: int = 3
UNMET_MODERATE_MIN_SIGNALS: int = 2
FAA_SLOT_CONTROLLED: tuple[str, ...] = ("JFK", "LGA", "DCA")
FAA_SCHEDULE_REVIEW: tuple[str, ...] = ("ORD", "LAX", "EWR", "SFO")
FAA_CONSTRAINT_SOURCE: dict[str, str] = {
    "dataset": "FAA ATO Slot Administration",
    "url": "https://www.faa.gov/about/office_org/headquarters_offices/ato/service_units/systemops/perf_analysis/slot_administration",
    "fetched": "2026-10-22",
}

# --- Long haul ----------------------------------------------------------------------------
LONG_HAUL_MIN_MILES: int = 2_500
LONG_HAUL_DISTANCE_GROUP: int = 11  # BTS DistanceGroup 11 = 2,500 miles or more

# --- Single-metric rankings ---------------------------------------------------------------
# metric name -> (what is ranked, unit). Ranked within the same airports as the expansion score.
RANK_METRICS: dict[str, tuple[str, str]] = {
    "growth": ("Enplanement growth, latest calendar year against the year before", "percent"),
    "size": ("Enplanements (passengers boarding) in the latest calendar year", "passengers boarding"),
    "late_departures": (f"Share of operated departures {LATE_DEPARTURE_MINUTES}+ minutes late", "percent"),
    "excess_taxi_out": ("Excess taxi-out time: monthly mean taxi-out minus the monthly "
                        f"{TAXI_BASELINE_PERCENTILE}th percentile", "minutes"),
    "mean_departure_delay": ("Mean departure delay of operated departures (early counts as 0)", "minutes"),
    "cancellation_rate": ("Share of scheduled flights cancelled", "percent"),
    "long_haul_share": (f"Share of operated departures of {LONG_HAUL_MIN_MILES:,} miles or more", "percent"),
}

# --- Places -------------------------------------------------------------------------------
REGIONS: dict[str, tuple[str, ...]] = {
    "new_england": ("CT", "ME", "MA", "NH", "RI", "VT"),
}
PLACE_ALIASES: dict[str, tuple[str, ...]] = {
    "la": ("LAX",),
    "los angeles": ("LAX",),
    "la area": ("LAX", "BUR", "LGB", "ONT"),
    "santa ana": ("SNA",),
    "orange county": ("SNA",),
    "anchorage": ("ANC",),
    # Cities whose main airport is listed by the FAA under another town (Hartford's is in Windsor Locks).
    "hartford": ("BDL",),
    "washington": ("DCA", "IAD"),
    "washington dc": ("DCA", "IAD"),
    "baltimore": ("BWI",),
    "cincinnati": ("CVG",),
    "new orleans": ("MSY",),
    "providence": ("PVD",),
    "san francisco": ("SFO",),
}
