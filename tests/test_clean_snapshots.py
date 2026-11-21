import pandas as pd

import clean_snapshots
from clean_snapshots import clean_snapshots as clean, load_stations

NOW = pd.Timestamp("2026-11-19 15:00", tz="UTC")

STATIONS = pd.DataFrame({
    "station_id": ["good", "empty", "full", "closed", "clogged", "silent", "broken"],
    "capacity": [20, 20, 20, 20, 20, 20, 20],
})


def status_row(
    station_id, bikes=5, docks=5, bikes_disabled=0, renting=True, last_reported=NOW
):
    return {
        "snapshot_time": NOW,
        "station_id": station_id,
        "bikes_available": bikes,
        "docks_available": docks,
        "bikes_disabled": bikes_disabled,
        "docks_disabled": 0,
        "is_installed": True,
        "is_renting": renting,
        "is_returning": True,
        "last_reported": last_reported,
    }


def make_status():
    return pd.DataFrame([
        status_row("good"),
        status_row("empty", bikes=0, docks=20),
        status_row("full", bikes=20, docks=0),
        status_row("closed", renting=False),
        status_row("clogged", bikes=0, docks=0, bikes_disabled=18),  # 18 of 20 bikes broken
        status_row("silent", last_reported=NOW - pd.Timedelta(hours=3)),
        status_row("broken", bikes=-1),
        status_row("mystery"),  # not in the station list
    ])


def test_bad_rows_are_removed_and_counted():
    df, report = clean(make_status(), STATIONS)

    assert sorted(df["station_id"]) == ["empty", "full", "good"]
    assert report == {
        "rows": 8,
        "unknown station": 1,
        "station closed": 1,
        "mostly broken": 1,
        "missing counts": 0,
        "negative counts": 1,
        "stopped reporting": 1,
    }


def test_empty_and_full_flags_and_capacity_are_added():
    df, _ = clean(make_status(), STATIONS)
    df = df.set_index("station_id")

    assert df.loc["empty", "is_empty"] and not df.loc["empty", "is_full"]
    assert df.loc["full", "is_full"] and not df.loc["full", "is_empty"]
    assert not df.loc["good", "is_empty"] and not df.loc["good", "is_full"]
    assert df.loc["good", "capacity"] == 20


def test_load_stations_keeps_the_latest_details(monkeypatch, tmp_path):
    for day, capacity in [("2026-11-18", 15), ("2026-11-19", 25)]:
        folder = tmp_path / f"date={day}"
        folder.mkdir()
        pd.DataFrame({
            "snapshot_time": [pd.Timestamp(day, tz="UTC")],
            "station_id": ["a"],
            "capacity": [capacity],
        }).to_parquet(folder / "info.parquet")
    monkeypatch.setattr(clean_snapshots, "INFO_DIR", tmp_path)

    stations = load_stations()

    assert len(stations) == 1
    assert stations["capacity"].iloc[0] == 25  # the newer value wins


def test_a_few_broken_bikes_are_fine():
    # 3 broken bikes out of 20: the station still works, so it stays
    status = pd.DataFrame([status_row("full", bikes=17, docks=0, bikes_disabled=3)])
    df, report = clean(status, STATIONS)

    assert len(df) == 1
    assert report["mostly broken"] == 0