import pandas as pd

import compact_snapshots
from compact_snapshots import compact_day, finished_days


def snapshot(time, station_ids):
    """A tiny fake snapshot: a few stations at one moment."""
    return pd.DataFrame({
        "snapshot_time": pd.Timestamp(time, tz="UTC"),
        "station_id": station_ids,
        "bikes_available": 5,
    })


def test_finished_days_leaves_out_today(monkeypatch, tmp_path):
    for day in ["2026-11-19", "2026-11-20", "2026-11-21"]:
        (tmp_path / f"date={day}").mkdir()
    monkeypatch.setattr(compact_snapshots, "RAW_DIR", tmp_path)

    assert finished_days(today="2026-11-21") == ["2026-11-19", "2026-11-20"]


def test_compact_day_combines_files_and_drops_duplicates(monkeypatch, tmp_path):
    day_folder = tmp_path / "date=2026-11-19"
    day_folder.mkdir()
    snapshot("2026-11-19 10:05", ["B", "A"]).to_parquet(day_folder / "b.parquet")
    snapshot("2026-11-19 10:00", ["A", "B"]).to_parquet(day_folder / "a.parquet")
    snapshot("2026-11-19 10:00", ["A"]).to_parquet(day_folder / "a_again.parquet")  # saved twice
    monkeypatch.setattr(compact_snapshots, "RAW_DIR", tmp_path)

    df = compact_day("2026-11-19")

    assert len(df) == 4  # 2 snapshots x 2 stations, with the duplicate removed
    assert list(df["station_id"]) == ["A", "B", "A", "B"]  # sorted by time, then station
    assert df["snapshot_time"].is_monotonic_increasing