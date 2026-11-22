# Leakage tests: prove that no feature secretly uses information from the future.
#
# The idea: build the features, then change everything that happens after some
# moment and build them again. Features for rows before that moment must not
# change at all. If one does, it was peeking ahead.

import numpy as np
import pandas as pd
import pytest

import build_features
from build_features import FEATURES, check_profiles_come_first

CUTOFF = pd.Timestamp("2026-11-19 11:00", tz="UTC")


def make_labeled(seed):
    """Three hours of snapshots for two stations, with random numbers."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-11-19 10:00", "2026-11-19 12:55", freq="5min", tz="UTC")
    df = pd.DataFrame({
        "time": np.repeat(times, 2),
        "station_id": np.tile(["a", "b"], len(times)),
    })
    for column in ["bikes_available", "ebikes_available", "docks_available", "bikes_disabled"]:
        df[column] = pd.array(rng.integers(0, 10, len(df)), dtype="Int64")
    df["capacity"] = pd.array([20] * len(df), dtype="Int64")
    df["is_empty"] = (df["bikes_available"] == 0).astype(int)
    df["is_full"] = (df["docks_available"] == 0).astype(int)
    for column in ["empty_next_hour", "full_next_hour", "empty_or_full_next_hour"]:
        df[column] = pd.array(rng.integers(0, 2, len(df)), dtype="Int64")
    return df


def build(labeled):
    weather = pd.DataFrame({
        "hour": pd.date_range("2026-11-19 09:00", periods=8, freq="h", tz="UTC"),
        "temperature_2m": 5.0,
        "wind_speed_10m": 10.0,
        "precipitation": 0.0,
    })
    demand = pd.DataFrame({
        "short_name": ["1.01"] * 24,
        "day_off": [False] * 24,
        "hour_of_day": range(24),
        "usual_departures": 2.0,
        "usual_arrivals": 1.0,
    })
    return build_features.build_features(
        labeled,
        stations=pd.DataFrame({"station_id": ["a", "b"], "short_name": ["1.01", "2.02"]}),
        types=pd.DataFrame({"short_name": ["1.01"], "station_type": [2]}),
        demand=demand,
        weather=weather,
    )


def test_features_never_use_the_future():
    original = make_labeled(seed=1)

    # Same data up to the cutoff, completely different numbers after it
    changed = original.copy()
    different = make_labeled(seed=2)
    after = changed["time"] > CUTOFF
    changed.loc[after] = different.loc[after]

    before_original = build(original).query("time <= @CUTOFF").reset_index(drop=True)
    before_changed = build(changed).query("time <= @CUTOFF").reset_index(drop=True)

    pd.testing.assert_frame_equal(before_original[FEATURES], before_changed[FEATURES])


def test_the_check_would_catch_a_feature_that_peeks():
    # A feature that looks 15 minutes ahead must change when the future changes.
    # If this ever passes, the test above couldn't catch a leak either.
    original = make_labeled(seed=1)
    changed = original.copy()
    after = changed["time"] > CUTOFF
    changed.loc[after, "bikes_available"] = changed.loc[after, "bikes_available"] + 5

    def peek(df):
        future = df.groupby("station_id")["bikes_available"].shift(-3)  # 3 snapshots ahead
        return future[df["time"] <= CUTOFF].reset_index(drop=True)

    assert not peek(original).equals(peek(changed))


def fake_trip_months(folder, months):
    for month in months:
        (folder / f"trips_hourly_{month}.parquet").touch()


def test_profiles_from_before_the_live_data_are_fine(monkeypatch, tmp_path):
    fake_trip_months(tmp_path, ["202609", "202610"])
    monkeypatch.setattr(build_features, "TRIPS_DIR", tmp_path)

    check_profiles_come_first(pd.Timestamp("2026-11-18 23:25", tz="UTC"))  # no error


def test_profiles_overlapping_the_live_data_are_stopped(monkeypatch, tmp_path):
    fake_trip_months(tmp_path, ["202610", "202611"])
    monkeypatch.setattr(build_features, "TRIPS_DIR", tmp_path)

    with pytest.raises(SystemExit):
        check_profiles_come_first(pd.Timestamp("2026-11-18 23:25", tz="UTC"))