import numpy as np
import pandas as pd

from build_features import (
    FEATURES,
    add_recent_changes,
    add_station_profiles,
    add_time_features,
    add_weather,
    build_features,
)


def utc(text):
    return pd.Timestamp(text, tz="UTC")


def test_time_features():
    df = add_time_features(pd.DataFrame({"time": [utc("2026-11-26 13:30")]}))

    # 13:30 UTC is 8:30 AM in New York, on Thanksgiving (a Thursday)
    assert df["time_of_day"][0] == 8.5
    assert df["day_of_week"][0] == 3
    assert df["day_off"][0]


def test_recent_changes_only_look_back():
    times = pd.date_range("2026-11-19 10:00", periods=7, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "time": times,
        "station_id": "A",
        "bikes_available": [5, 5, 5, 8, 8, 8, 2],
    })
    df = add_recent_changes(df).set_index("time")

    assert np.isnan(df.loc[utc("2026-11-19 10:00"), "bikes_change_15min"])  # no past yet
    assert df.loc[utc("2026-11-19 10:15"), "bikes_change_15min"] == 3  # 5 -> 8
    assert df.loc[utc("2026-11-19 10:30"), "bikes_change_15min"] == -6  # 8 -> 2
    assert df["bikes_change_60min"].isna().all()  # never an hour of history


def test_station_profiles():
    df = pd.DataFrame({
        "station_id": ["a", "b", "c"],
        "hour_of_day": [8, 8, 8],
        "day_off": [False, False, False],
    })
    stations = pd.DataFrame({"station_id": ["a", "b", "c"], "short_name": ["1.01", "2.02", "3.03"]})
    types = pd.DataFrame({"short_name": ["1.01"], "station_type": [3]})
    demand = pd.DataFrame({
        "short_name": ["1.01", "1.01", "2.02"],
        "day_off": [False, False, False],
        "hour_of_day": [8, 9, 14],
        "usual_departures": [6.0, 2.0, 1.0],
        "usual_arrivals": [1.0, 4.0, 1.0],
    })
    df = add_station_profiles(df, stations, types, demand).set_index("station_id")

    assert df.loc["a", "station_type"] == 3
    assert df.loc["a", "usual_departures_this_hour"] == 6  # 8 AM
    assert df.loc["a", "usual_arrivals_next_hour"] == 4  # 9 AM
    assert df.loc["b", "usual_departures_this_hour"] == 0  # has history, but no trips at 8 AM
    assert np.isnan(df.loc["c", "usual_departures_this_hour"])  # no history at all
    assert np.isnan(df.loc["b", "station_type"])  # too quiet to have a type


def test_weather_lines_up_with_the_right_hours():
    weather = pd.DataFrame({
        "hour": pd.date_range("2026-11-19 10:00", periods=3, freq="h", tz="UTC"),
        "temperature_2m": [5.0, 6.0, 7.0],
        "wind_speed_10m": [10.0, 12.0, 14.0],
        "precipitation": [0.0, 0.4, 2.0],  # rain from 9-10, 10-11 and 11-12
    })
    df = add_weather(pd.DataFrame({"time": [utc("2026-11-19 10:20")]}), weather)

    assert df["temperature"][0] == 5  # at 10:00
    assert df["rain_this_hour"][0] == 0.4  # 10:00-11:00
    assert df["rain_next_hour"][0] == 2.0  # 11:00-12:00


def test_build_features_gives_plain_numbers():
    times = pd.date_range("2026-11-19 10:00", periods=3, freq="5min", tz="UTC")
    labeled = pd.DataFrame({
        "time": times,
        "station_id": "a",
        "bikes_available": pd.array([3, 2, 0], dtype="Int64"),
        "ebikes_available": pd.array([1, 1, 0], dtype="Int64"),
        "docks_available": pd.array([7, 8, 10], dtype="Int64"),
        "bikes_disabled": pd.array([0, 0, None], dtype="Int64"),
        "capacity": pd.array([10, 10, 10], dtype="Int64"),
        "is_empty": [0, 0, 1],
        "is_full": [0, 0, 0],
        "empty_next_hour": pd.array([1, 1, 0], dtype="Int64"),
        "full_next_hour": pd.array([0, 0, 0], dtype="Int64"),
        "empty_or_full_next_hour": pd.array([1, 1, 0], dtype="Int64"),
    })
    weather = pd.DataFrame({
        "hour": [utc("2026-11-19 10:00")],
        "temperature_2m": [5.0],
        "wind_speed_10m": [10.0],
        "precipitation": [0.0],
    })
    df = build_features(
        labeled,
        stations=pd.DataFrame({"station_id": ["a"], "short_name": ["1.01"]}),
        types=pd.DataFrame({"short_name": ["1.01"], "station_type": [0]}),
        demand=pd.DataFrame(columns=["short_name", "day_off", "hour_of_day",
                                     "usual_departures", "usual_arrivals"]),
        weather=weather,
    )

    assert len(df) == 3
    assert all(df[feature].dtype == "float64" for feature in FEATURES)
    assert df["fill_ratio"].tolist() == [0.3, 0.2, 0.0]