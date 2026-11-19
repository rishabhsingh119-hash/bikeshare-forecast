import pandas as pd

from check_data import print_report, summarize


def make_snapshots(times, bikes, docks):
    """Build a small fake table of snapshots for one open station."""
    return pd.DataFrame(
        {
            "snapshot_time": pd.to_datetime(times, utc=True),
            "station_id": "1",
            "bikes_available": pd.array(bikes, dtype="Int64"),
            "docks_available": pd.array(docks, dtype="Int64"),
            "is_installed": True,
            "is_renting": True,
        }
    )


def test_finds_missing_snapshots():
    # 10:10 to 10:25 is missing
    df = make_snapshots(
        ["2026-11-16 10:00", "2026-11-16 10:05", "2026-11-16 10:30", "2026-11-16 10:35"],
        bikes=[5, 5, 5, 5],
        docks=[5, 5, 5, 5],
    )
    s = summarize(df, poll_interval=300)

    assert s["collected"] == 4
    assert s["expected"] == 8
    assert len(s["gaps"]) == 1


def test_snapshot_taken_at_startup_still_counts():
    # The collector takes one snapshot as soon as it starts, before the round times
    df = make_snapshots(
        ["2026-11-16 10:02:30", "2026-11-16 10:05:00", "2026-11-16 10:10:00"],
        bikes=[5, 5, 5],
        docks=[5, 5, 5],
    )
    s = summarize(df, poll_interval=300)

    assert s["collected"] == s["expected"] == 3
    assert s["gaps"] == []


def test_empty_and_full_rates():
    df = make_snapshots(
        ["2026-11-16 10:00", "2026-11-16 10:05", "2026-11-16 10:10", "2026-11-16 10:15"],
        bikes=[0, 2, 5, 0],
        docks=[10, 8, 0, 10],
    )
    s = summarize(df, poll_interval=300)

    assert s["empty_pct"] == 50  # 2 of 4 snapshots had no bikes
    assert s["full_pct"] == 25  # 1 of 4 snapshots had no docks


def test_closed_stations_are_ignored():
    df = pd.DataFrame(
        {
            "snapshot_time": pd.to_datetime(["2026-11-16 10:00"] * 2, utc=True),
            "station_id": ["open", "closed"],
            "bikes_available": pd.array([5, 0], dtype="Int64"),
            "docks_available": pd.array([5, 0], dtype="Int64"),
            "is_installed": [True, True],
            "is_renting": [True, False],
        }
    )
    s = summarize(df, poll_interval=300)

    assert s["open_stations"] == 1
    assert s["empty_pct"] == 0  # the closed station's 0 bikes doesn't count
    assert s["full_pct"] == 0


def test_report_prints_without_errors(capsys):
    df = make_snapshots(
        ["2026-11-16 10:00", "2026-11-16 10:05"],
        bikes=[5, 5],
        docks=[5, 5],
    )
    print_report(summarize(df, poll_interval=300))

    output = capsys.readouterr().out
    assert "2 of 2 expected" in output
    assert "America/New_York" in output
