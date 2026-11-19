import zipfile

import pandas as pd

from backfill_trips import hourly_counts, month_list, read_trips, to_utc

# The header of Citi Bike's trip history CSVs
HEADER = (
    "ride_id,rideable_type,started_at,ended_at,start_station_name,start_station_id,"
    "end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual\n"
)


def trip_row(ride_id, start, end, from_station, to_station):
    """One line of a trip CSV. Only the columns we use have real values."""
    return (
        f"{ride_id},classic_bike,{start},{end},Start St,{from_station},"
        f"End St,{to_station},40.7,-74.0,40.7,-74.0,member\n"
    )


def make_zip(path, files):
    """Build a fake monthly zip file from {file name: [rows]}."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, rows in files.items():
            zf.writestr(name, HEADER + "".join(rows))
    return path


def test_month_list_crosses_years():
    assert month_list("202511", "202602") == ["202511", "202512", "202601", "202602"]


def test_read_trips_combines_files_and_drops_duplicates(tmp_path):
    zip_path = make_zip(
        tmp_path / "trips.zip",
        {
            "202607-citibike-tripdata_1.csv": [
                trip_row("r1", "2026-07-01 08:10:00", "2026-07-01 08:25:00", "6140.10", "5329.03"),
            ],
            "202607-citibike-tripdata_2.csv": [
                trip_row("r1", "2026-07-01 08:10:00", "2026-07-01 08:25:00", "6140.10", "5329.03"),
                trip_row("r2", "2026-07-01 09:00:00", "2026-07-01 09:20:00", "5329.03", "6140.10"),
            ],
            "__MACOSX/._202607-citibike-tripdata_1.csv": [],
        },
    )
    trips = read_trips(zip_path)

    assert sorted(trips["ride_id"]) == ["r1", "r2"]  # r1 only counted once
    assert "6140.10" in set(trips["start_station_id"])  # not turned into 6140.1


def test_to_utc_handles_daylight_saving():
    times = pd.Series(["2026-07-01 08:00:00", "2026-11-01 01:30:00"])
    utc = to_utc(times)

    assert utc[0] == pd.Timestamp("2026-07-01 12:00", tz="UTC")  # New York is UTC-4 in July
    assert pd.isna(utc[1])  # 1:30 AM happened twice that night, so it's dropped


def test_hourly_counts():
    trips = pd.DataFrame(
        [
            # ride_id, started_at, ended_at, start_station_id, end_station_id
            ["r1", "2026-07-01 08:10:00", "2026-07-01 08:25:00", "A", "B"],
            ["r2", "2026-07-01 08:40:00", "2026-07-01 09:05:00", "A", "A"],
            ["r3", "2026-07-01 08:50:00", "2026-07-01 09:10:00", "B", None],  # never docked
        ],
        columns=["ride_id", "started_at", "ended_at", "start_station_id", "end_station_id"],
    )

    counts = hourly_counts(trips).set_index(["station_id", "hour"])
    at_8am = pd.Timestamp("2026-07-01 12:00", tz="UTC")  # 8 AM New York time
    at_9am = pd.Timestamp("2026-07-01 13:00", tz="UTC")

    assert counts.loc[("A", at_8am), "departures"] == 2
    assert counts.loc[("A", at_8am), "arrivals"] == 0
    assert counts.loc[("B", at_8am), "arrivals"] == 1
    assert counts.loc[("A", at_9am), "arrivals"] == 1
    assert counts["arrivals"].sum() == 2  # r3 has no end station, so no arrival
