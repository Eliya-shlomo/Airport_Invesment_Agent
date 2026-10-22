"""Download the raw data files into data/raw. Safe to rerun: files that are already there are skipped.

    uv run python -m sources.download            # fetch what is missing
    uv run python -m sources.download --force    # refetch every file

Each file is written to a `.part` file and renamed when complete, so a failed download never leaves a
half-written file under its final name. A file's modification time is its fetch date (sources.build reads it).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import httpx

from sources import config

T = TypeVar("T")

TIMEOUT = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
RETRIES = 3
BACKOFF_SECONDS = 2.0
MIN_SOCRATA_ROWS = 1_000


class DownloadError(RuntimeError):
    """A dataset could not be fetched or failed a sanity check."""


def log(message: str) -> None:
    print(message, flush=True)


def make_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": config.USER_AGENT}, follow_redirects=True, timeout=TIMEOUT)


def with_retries(action: Callable[[], T], tries: int = RETRIES, sleep: Callable[[float], None] = time.sleep) -> T:
    """Run `action`, retrying connection errors and 5xx/429 responses with a growing pause."""
    for attempt in range(1, tries + 1):
        try:
            return action()
        except httpx.HTTPStatusError as err:
            retryable = err.response.status_code >= 500 or err.response.status_code == 429
            if not retryable or attempt == tries:
                raise
        except httpx.TransportError:
            if attempt == tries:
                raise
        sleep(BACKOFF_SECONDS * attempt)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- files


def remote_exists(client: httpx.Client, url: str) -> bool:
    """True if the server has the file. Reads one byte only (the On-Time files are about 30 MB)."""

    def probe() -> bool:
        with client.stream("GET", url, headers={"Range": "bytes=0-0"}) as response:
            if response.status_code == 404:
                return False
            response.raise_for_status()
            return True

    return with_retries(probe)


def download_file(client: httpx.Client, url: str, dest: Path) -> None:
    """Stream `url` to `dest` through a `.part` file; on failure `dest` is untouched and no `.part` is left."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    def attempt() -> None:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with part.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)

    try:
        with_retries(attempt)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)


def fetch(client: httpx.Client, url: str, dest: Path, force: bool) -> None:
    """Download `url` to `dest` unless the file is already there (or `force`)."""
    if dest.exists() and not force:
        log(f"  {dest.name:<55} skipped (already downloaded)")
        return
    download_file(client, url, dest)
    log(f"  {dest.name:<55} {dest.stat().st_size / 1e6:8.1f} MB  downloaded")


# ---------------------------------------------------------------- On-Time

Month = tuple[int, int]  # (year, month)


def previous_month(month: Month) -> Month:
    year, mon = month
    return (year - 1, 12) if mon == 1 else (year, mon - 1)


def month_window(latest: Month, count: int) -> list[Month]:
    """The `count` months ending at `latest` (inclusive), oldest first."""
    months = [latest]
    while len(months) < count:
        months.append(previous_month(months[-1]))
    return months[::-1]


def ontime_url(month: Month) -> str:
    """BTS URLs do not zero-pad the month."""
    return config.ONTIME_URL.format(year=month[0], month=month[1])


def ontime_filename(month: Month) -> str:
    """Local names zero-pad the month so they sort chronologically."""
    return f"ontime_{month[0]}_{month[1]:02d}.zip"


def latest_ontime_month(client: httpx.Client, today: datetime) -> Month:
    """The newest published month: try last month, then go back (BTS publishes about two months late)."""
    month = previous_month((today.year, today.month))
    for _ in range(config.ONTIME_MAX_LAG_MONTHS):
        if remote_exists(client, ontime_url(month)):
            return month
        month = previous_month(month)
    raise DownloadError(f"no On-Time file found in the last {config.ONTIME_MAX_LAG_MONTHS} months")


def download_ontime(client: httpx.Client, today: datetime, force: bool) -> None:
    latest = latest_ontime_month(client, today)
    window = month_window(latest, config.ONTIME_WINDOW_MONTHS)
    log(f"On-Time: {len(window)} months, {window[0][0]}-{window[0][1]:02d} to {latest[0]}-{latest[1]:02d}")
    for month in window:
        fetch(client, ontime_url(month), config.RAW / "ontime" / ontime_filename(month), force)


# ---------------------------------------------------------------- FAA and OurAirports


def download_faa(client: httpx.Client, today: datetime, force: bool) -> None:
    """The latest FAA enplanements workbook (last year's, else the year before). It also holds the prior year."""
    for year in (today.year - 1, today.year - 2):
        url = config.FAA_ENPLANEMENTS_URL.format(year=year)
        if remote_exists(client, url):
            log(f"FAA enplanements: calendar year {year}")
            fetch(client, url, config.RAW / "faa" / url.rsplit("/", 1)[1], force)
            return
    raise DownloadError(f"no FAA enplanements workbook for {today.year - 1} or {today.year - 2}")


def download_ourairports(client: httpx.Client, force: bool) -> None:
    log("OurAirports:")
    for name in config.OURAIRPORTS_FILES:
        fetch(client, f"{config.OURAIRPORTS_BASE}/{name}", config.RAW / "ourairports" / name, force)


# ---------------------------------------------------------------- DOT airfare (Socrata)


def fetch_socrata_rows(client: httpx.Client, url: str, where: str) -> list[dict[str, Any]]:
    """All rows matching `where`. Socrata returns 1,000 rows by default, so page with a stable `:id` order."""
    rows: list[dict[str, Any]] = []
    page = config.SOCRATA_PAGE_SIZE
    while True:
        params = {"$where": where, "$order": ":id", "$limit": page, "$offset": len(rows)}

        def get_page(params: dict[str, Any] = params) -> list[dict[str, Any]]:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()

        batch = with_retries(get_page)
        rows.extend(batch)
        if len(batch) < page:
            return rows


def download_airfare(client: httpx.Client, today: datetime, force: bool) -> None:
    dest = config.RAW / "socrata" / f"airfare_{config.AIRFARE_RESOURCE}.json"
    log("DOT airfare:")
    if dest.exists() and not force:
        log(f"  {dest.name:<55} skipped (already downloaded)")
        return
    where = f"year >= '{today.year - config.AIRFARE_YEARS_BACK}'"  # `year` is text in this dataset
    rows = fetch_socrata_rows(client, f"{config.SOCRATA_BASE}/{config.AIRFARE_RESOURCE}.json", where)
    if len(rows) < MIN_SOCRATA_ROWS:
        raise DownloadError(f"airfare: only {len(rows)} rows returned (expected at least {MIN_SOCRATA_ROWS})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_text(json.dumps(rows))
    part.replace(dest)
    log(f"  {dest.name:<55} {len(rows):8,} rows  downloaded")


# ---------------------------------------------------------------- main


def run(client: httpx.Client, today: datetime, force: bool = False) -> None:
    """Fetch every dataset. One failing dataset does not stop the others; raises DownloadError at the end."""
    steps: dict[str, Callable[[], None]] = {
        "ontime": lambda: download_ontime(client, today, force),
        "faa_enplanements": lambda: download_faa(client, today, force),
        "airfare": lambda: download_airfare(client, today, force),
        "ourairports": lambda: download_ourairports(client, force),
    }
    failures: dict[str, str] = {}
    for key, step in steps.items():
        try:
            step()
        except (DownloadError, httpx.HTTPError, OSError, ValueError) as err:
            failures[key] = f"{type(err).__name__}: {err}"
            log(f"  FAILED {key}: {failures[key]}")
    if failures:
        raise DownloadError(f"{len(failures)} dataset(s) failed: " + "; ".join(f"{k} ({v})" for k, v in failures.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the raw data files into data/raw.")
    parser.add_argument("--force", action="store_true", help="refetch every file")
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        with make_client() as client:
            run(client, datetime.now(timezone.utc), force=args.force)
    except DownloadError as err:
        print(f"download failed: {err}", file=sys.stderr)
        return 1
    log(f"done in {time.monotonic() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
