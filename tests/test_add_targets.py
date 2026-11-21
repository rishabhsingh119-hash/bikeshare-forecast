import pandas as pd

from add_targets import add_targets, to_slots


def snapshots(station_id, start, end, empty_at=(), skip=()):
    """Snapshots every 5 minutes for one station. It's empty at the times in
    `empty_at`, and the times in `skip` are missing."""
    rows = []
    for time in pd.date_range(start, end, freq="5min", tz="UTC"):
        clock = time.strftime("%H:%M")
        if clock in skip:
            continue
        rows.append({
            "snapshot_time": time,
            "station_id": station_id,
            "is_empty": clock in empty_at,
            "is_full": False,
        })
    return pd.DataFrame(rows)


def answer(df, station_id, clock, column="empty_next_hour"):
    """The answer for one station at one time, or None if that row was dropped."""
    rows = df[(df["station_id"] == station_id) & (df["time"].dt.strftime("%H:%M") == clock)]
    return None if rows.empty else rows[column].iloc[0]


def test_becomes_empty_within_the_hour():
    df = add_targets(snapshots("A", "2026-11-19 10:00", "2026-11-19 11:30", empty_at={"10:50"}))

    assert answer(df, "A", "10:00") == 1  # 10:50 is within the next hour
    assert answer(df, "A", "10:45") == 1  # 10:50 is the very next snapshot


def test_quiet_hour_gives_a_no():
    df = add_targets(snapshots("B", "2026-11-19 10:00", "2026-11-19 11:30"))

    assert answer(df, "B", "10:00") == 0  # a full hour seen, never empty
    assert answer(df, "B", "10:30") == 0  # the last row with a full hour after it
    assert answer(df, "B", "10:35") is None  # the data ends before the hour is up


def test_missing_snapshot_means_we_cant_say_no():
    df = add_targets(snapshots("C", "2026-11-19 10:00", "2026-11-19 11:30", skip={"10:20"}))

    assert answer(df, "C", "10:00") is None  # the hour after 10:00 has a hole in it
    assert answer(df, "C", "10:20") is None  # no snapshot at 10:20 at all


def test_right_now_doesnt_count():
    df = add_targets(snapshots("D", "2026-11-19 10:00", "2026-11-19 11:30", empty_at={"10:00"}))

    assert answer(df, "D", "10:00") == 0  # empty now, but not in the next hour
    assert answer(df, "D", "10:00", column="is_empty") == 1


def test_two_snapshots_in_one_slot_keeps_the_later():
    df = pd.DataFrame({
        "snapshot_time": pd.to_datetime(["2026-11-19 10:05:00", "2026-11-19 10:07:56"], utc=True),
        "station_id": ["A", "A"],
        "is_empty": [False, True],
        "is_full": [False, False],
    })
    slots = to_slots(df)

    assert len(slots) == 1
    assert slots["time"][0] == pd.Timestamp("2026-11-19 10:05", tz="UTC")
    assert slots["is_empty"][0] == 1  # from the 10:07:56 snapshot