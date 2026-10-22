"""Confidence levels set by fixed rules from data coverage, never by the LLM."""

from __future__ import annotations

from analytics import config

LEVELS = ("low", "medium", "high")
RANKING = "medium"  # proxy congestion and one year of growth cap every ranking at medium
UNMET_DEMAND = "low"  # an indirect estimate, always


def lowest(levels: list[str]) -> str:
    """The weakest of several levels."""
    return min(levels, key=LEVELS.index)


def from_coverage(months: int, flown: int, proxy: bool = False) -> str:
    """A measured count or share is high with a full On-Time window, else medium. A proxy (congestion) is one
    level lower: medium with a full window, else low."""
    full = months >= config.FULL_WINDOW_MONTHS and flown >= config.MIN_FLOWN_DEPARTURES
    if proxy:
        return "medium" if full else "low"
    return "high" if full else "medium"


def coverage_warnings(airport: str, months: int, flown: int, first: str, last: str) -> list[str]:
    """Warnings the answer must mention when an airport's On-Time coverage is partial or thin."""
    warnings = []
    if months < config.FULL_WINDOW_MONTHS:
        warnings.append(
            f"{airport}: only {months} of {config.FULL_WINDOW_MONTHS} months of On-Time data ({first}..{last}); "
            "figures may reflect a season, not a full year"
        )
    if flown < config.MIN_FLOWN_DEPARTURES:
        warnings.append(
            f"{airport}: only {flown} reporting-carrier departures in the window "
            f"(fewer than {config.MIN_FLOWN_DEPARTURES:,}); figures are unstable"
        )
    return warnings
