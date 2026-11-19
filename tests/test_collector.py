import gzip
import json
from datetime import UTC, datetime

import pytest
import requests

import collector
import config

# A tiny fake station_status feed with one station
STATUS = {
    "last_updated": 1763303700,
    "data": {
        "stations": [
            {
                "station_id": "1",
                "num_bikes_available": 3,
                "num_docks_available": 17,
                "is_installed": True,
                "is_renting": True,
                "is_returning": True,
            },
        ]
    },
}
FEED_URLS = {"station_status": "status-url", "station_information": "info-url"}
NOW = datetime(2026, 11, 16, 14, 35, tzinfo=UTC)


class FakeResponse:
    """Pretends to be what requests.get() returns, so tests don't need the internet."""

    def __init__(self, data=None, fail=False):
        self.data = data
        self.fail = fail

    def raise_for_status(self):
        if self.fail:
            raise requests.HTTPError("503 Server Error")

    def json(self):
        return self.data


def test_get_json_retries_after_a_failure(monkeypatch):
    # First request fails, second one works
    responses = [FakeResponse(fail=True), FakeResponse({"ok": True})]
    monkeypatch.setattr(collector.requests, "get", lambda url, timeout: responses.pop(0))
    monkeypatch.setattr(collector.time, "sleep", lambda seconds: None)  # don't actually wait

    assert collector.get_json("some-url") == {"ok": True}


def test_get_json_gives_up_eventually(monkeypatch):
    monkeypatch.setattr(collector.requests, "get", lambda url, timeout: FakeResponse(fail=True))
    monkeypatch.setattr(collector.time, "sleep", lambda seconds: None)

    with pytest.raises(requests.HTTPError):
        collector.get_json("some-url")


def test_snapshot_is_saved_in_the_right_place(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path)  # save into a temp folder
    monkeypatch.setattr(collector, "get_json", lambda url: STATUS)

    collector.save_station_status(FEED_URLS, NOW, previous_update=None)

    day = "date=2026-11-16"
    parquet = tmp_path / "station_status" / day / "station_status_20261116T143500Z.parquet"
    raw = tmp_path / "json" / "station_status" / day / "station_status_20261116T143500Z.json.gz"
    assert parquet.exists()
    with gzip.open(raw, "rt") as f:
        assert json.load(f) == STATUS


def test_unchanged_feed_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(collector, "get_json", lambda url: STATUS)

    # Pretend we already saved this exact version of the feed
    collector.save_station_status(FEED_URLS, NOW, previous_update=STATUS["last_updated"])

    assert not (tmp_path / "station_status").exists()
