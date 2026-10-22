"""Offline tests for sources/download.py: no network, HTTP is mocked with httpx.MockTransport."""

import json
from datetime import datetime, timezone

import httpx
import pytest

from sources import config, download


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def files_handler(files: dict[str, bytes]):
    """Serves `files` (url -> bytes); 404 for anything else."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = files.get(str(request.url))
        return httpx.Response(404) if body is None else httpx.Response(200, content=body)

    return handler


# ------------------------------------------------------------ latest published month


def test_latest_ontime_month_goes_back_until_a_file_exists():
    today = datetime(2026, 10, 22, tzinfo=timezone.utc)
    client = client_for(files_handler({download.ontime_url((2026, 8)): b"zip"}))
    assert download.latest_ontime_month(client, today) == (2026, 8)  # 2026-09 is not published yet


# ------------------------------------------------------------ files


def test_fetch_skips_an_existing_file_unless_forced(tmp_path):
    url = "https://x.test/a.zip"
    client = client_for(files_handler({url: b"new-content"}))
    dest = tmp_path / "a.zip"
    dest.write_bytes(b"old")
    download.fetch(client, url, dest, force=False)
    assert dest.read_bytes() == b"old"
    download.fetch(client, url, dest, force=True)
    assert dest.read_bytes() == b"new-content"
    assert not dest.with_name("a.zip.part").exists()


def test_failed_download_keeps_existing_file_and_leaves_no_part(tmp_path):
    dest = tmp_path / "a.zip"
    dest.write_bytes(b"keep me")
    with pytest.raises(httpx.HTTPStatusError):
        download.download_file(client_for(files_handler({})), "https://x.test/a.zip", dest)
    assert dest.read_bytes() == b"keep me"
    assert list(tmp_path.glob("*.part")) == []


def test_retries_server_errors_then_succeeds():
    calls = []

    def action():
        calls.append(1)
        if len(calls) < 3:
            raise httpx.HTTPStatusError(
                "boom", request=httpx.Request("GET", "https://x.test"), response=httpx.Response(503)
            )
        return "ok"

    sleeps = []
    assert download.with_retries(action, sleep=sleeps.append) == "ok"
    assert len(calls) == 3 and len(sleeps) == 2


def test_does_not_retry_404():
    calls = []

    def action():
        calls.append(1)
        raise httpx.HTTPStatusError(
            "nope", request=httpx.Request("GET", "https://x.test"), response=httpx.Response(404)
        )

    with pytest.raises(httpx.HTTPStatusError):
        download.with_retries(action, sleep=lambda s: None)
    assert len(calls) == 1


# ------------------------------------------------------------ DOT airfare (Socrata)


def test_fetch_socrata_rows_pages_with_stable_order(monkeypatch):
    monkeypatch.setattr(config, "SOCRATA_PAGE_SIZE", 2)
    data = [{"i": str(i)} for i in range(5)]
    seen = []

    def handler(request):
        params = request.url.params
        seen.append((params["$order"], params["$limit"], params["$offset"]))
        offset, limit = int(params["$offset"]), int(params["$limit"])
        return httpx.Response(200, json=data[offset : offset + limit])

    rows = download.fetch_socrata_rows(client_for(handler), "https://x.test/r.json", "year >= '2024'")
    assert rows == data
    assert seen == [(":id", "2", "0"), (":id", "2", "2"), (":id", "2", "4")]


def test_airfare_downloads_with_a_quoted_year_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW", tmp_path)
    rows = [{"year": "2025", "n": str(i)} for i in range(1200)]
    wheres = []

    def handler(request):
        wheres.append(request.url.params["$where"])
        return httpx.Response(200, json=rows)

    download.download_airfare(client_for(handler), datetime(2026, 10, 22, tzinfo=timezone.utc), force=False)
    assert json.loads((tmp_path / "socrata" / "airfare_tfrh-tu9e.json").read_text()) == rows
    assert wheres == ["year >= '2024'"]  # `year` is text in this dataset, so the value is quoted


def test_airfare_with_too_few_rows_aborts_and_keeps_old_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW", tmp_path)
    dest = tmp_path / "socrata" / "airfare_tfrh-tu9e.json"
    dest.parent.mkdir()
    dest.write_text("[]")
    client = client_for(lambda r: httpx.Response(200, json=[{"a": "1"}] * 5))
    with pytest.raises(download.DownloadError, match="only 5 rows"):
        download.download_airfare(client, datetime(2026, 10, 22, tzinfo=timezone.utc), force=True)
    assert dest.read_text() == "[]"


# ------------------------------------------------------------ end to end (mocked)


def test_run_reports_every_failed_dataset_and_keeps_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW", tmp_path)
    keep = tmp_path / "ourairports" / "airports.csv"
    keep.parent.mkdir()
    keep.write_bytes(b"precious")
    with pytest.raises(download.DownloadError, match="ontime") as exc:
        download.run(client_for(lambda r: httpx.Response(404)), datetime(2026, 10, 22, tzinfo=timezone.utc))
    assert "faa_enplanements" in str(exc.value)
    assert keep.read_bytes() == b"precious"
