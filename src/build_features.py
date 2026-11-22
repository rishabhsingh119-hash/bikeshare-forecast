# Builds the table the model learns from: one row per station per 5-minute
# snapshot, with everything we know at that moment (the features) and whether
# the station became empty or full in the next hour (the answer).
#
# Every feature only uses what we'd know at that moment. The weather is a
# forecast, which we'd also have when making a live prediction.
#
# Usage:
#   uv run python src/build_features.py
#
# Output:
#   data/processed/features.parquet

import pandas as pd

import config
from add_targets import TARGETS
from build_station_profiles import TRIPS_DIR
from clean_snapshots import load_stations
from days import is_day_off

LABELED_PATH = config.PROCESSED_DATA_DIR / "labeled_snapshots.parquet"
DEMAND_PATH = config.PROCESSED_DATA_DIR / "usual_demand.parquet"
TYPES_PATH = config.PROCESSED_DATA_DIR / "station_types.parquet"
WEATHER_PATH = config.PROCESSED_DATA_DIR / "weather_hourly.parquet"
OUTPUT_PATH = config.PROCESSED_DATA_DIR / "features.parquet"

SLOT = pd.Timedelta(seconds=config.POLL_INTERVAL)  # one snapshot every 5 minutes

# The columns the model is allowed to look at
FEATURES = [
    # Time
    "time_of_day",
    "day_of_week",
    "day_off",
    # The station right now
    "bikes_available",
    "ebikes_available",
    "docks_available",
    "bikes_disabled",
    "capacity",
    "fill_ratio",
    "is_empty",
    "is_full",
    # How it has been changing
    "bikes_change_15min",
    "bikes_change_30min",
    "bikes_change_60min",
    # What's normal for this station
    "station_type",
    "usual_departures_this_hour",
    "usual_arrivals_this_hour",
    "usual_departures_next_hour",
    "usual_arrivals_next_hour",
    # Weather
    "temperature",
    "wind_speed",
    "rain_this_hour",
    "rain_next_hour",
]
TARGET = "empty_or_full_next_hour"


def add_time_features(df):
    local = df["time"].dt.tz_convert(config.TIMEZONE)
    df["hour_of_day"] = local.dt.hour  # used to look up the usual demand
    df["time_of_day"] = local.dt.hour + local.dt.minute / 60  # 8:30 AM -> 8.5
    df["day_of_week"] = local.dt.day_of_week  # Monday is 0
    df["day_off"] = is_day_off(local)
    return df


def add_state_features(df):
    # How full the station is, from 0 (no bikes) to 1 (no free docks)
    df["fill_ratio"] = df["bikes_available"] / (df["bikes_available"] + df["docks_available"])
    return df


def add_recent_changes(df):
    """How many bikes each station gained (+) or lost (-) over the last 15, 30 and 60 minutes.

    Only looks backwards, so it never uses anything from the future.
    """
    # A grid with one row per 5-minute slot and one column per station, so a
    # missing snapshot is a gap instead of being skipped over
    wide = df.pivot(index="time", columns="station_id", values="bikes_available")
    all_slots = pd.date_range(wide.index.min(), wide.index.max(), freq=SLOT, name="time")
    wide = wide.reindex(all_slots)

    changes = []
    for minutes in [15, 30, 60]:
        steps = pd.Timedelta(minutes=minutes) // SLOT  # 3, 6 or 12 snapshots back
        change = wide - wide.shift(steps)
        changes.append(change.stack().dropna().rename(f"bikes_change_{minutes}min"))

    changes = pd.concat(changes, axis=1).reset_index()
    return df.merge(changes, on=["time", "station_id"], how="left")


def add_station_profiles(df, stations, types, demand):
    """Add each station's type and its usual departures and arrivals this hour and next."""
    # The live data uses long station IDs; the profiles use short names
    df = df.merge(stations[["station_id", "short_name"]], on="station_id", how="left")
    df = df.merge(types, on="short_name", how="left")

    this_hour = demand.rename(columns={
        "usual_departures": "usual_departures_this_hour",
        "usual_arrivals": "usual_arrivals_this_hour",
    })
    df = df.merge(this_hour, on=["short_name", "day_off", "hour_of_day"], how="left")

    next_hour = demand.rename(columns={
        "hour_of_day": "next_hour_of_day",
        "usual_departures": "usual_departures_next_hour",
        "usual_arrivals": "usual_arrivals_next_hour",
    })
    df["next_hour_of_day"] = (df["hour_of_day"] + 1) % 24
    df = df.merge(next_hour, on=["short_name", "day_off", "next_hour_of_day"], how="left")

    # A station with trip history but no trips in that hour usually has 0.
    # A station with no trip history at all stays blank: we just don't know.
    known = df["short_name"].isin(demand["short_name"])
    for column in [
        "usual_departures_this_hour",
        "usual_arrivals_this_hour",
        "usual_departures_next_hour",
        "usual_arrivals_next_hour",
    ]:
        df.loc[known, column] = df.loc[known, column].fillna(0)
    return df.drop(columns="next_hour_of_day")


def add_weather(df, weather):
    """Temperature and wind at the start of this hour, and the rain forecast.

    Open-Meteo's rain number for 11:00 is the total from 10:00 to 11:00. So for a
    snapshot at 10:20, "rain this hour" is the 11:00 number and "rain next hour"
    is the 12:00 one.
    """
    weather = weather.set_index("hour")
    hour = df["time"].dt.floor("h")
    one_hour = pd.Timedelta(hours=1)

    df["temperature"] = hour.map(weather["temperature_2m"])
    df["wind_speed"] = hour.map(weather["wind_speed_10m"])
    df["rain_this_hour"] = (hour + one_hour).map(weather["precipitation"])
    df["rain_next_hour"] = (hour + 2 * one_hour).map(weather["precipitation"])
    return df


def check_profiles_come_first(first_time):
    """Stop if the station profiles were built from trips during the live data period.

    The profiles must only know about the past. If they included trips from the
    same days the model is tested on, they'd quietly give away the answers.
    """
    months = sorted(path.stem.removeprefix("trips_hourly_") for path in TRIPS_DIR.glob("*.parquet"))
    last_month = pd.Period(f"{months[-1][:4]}-{months[-1][4:]}", freq="M")
    trips_end = last_month.end_time.tz_localize(config.TIMEZONE)

    if trips_end >= first_time:
        raise SystemExit(
            f"Trip history runs until {last_month}, but the live data starts on "
            f"{first_time:%Y-%m-%d}. Remove the overlapping trip months and rebuild "
            "the station profiles before building features."
        )


def build_features(labeled, stations, types, demand, weather):
    df = labeled.copy()
    df = add_time_features(df)
    df = add_state_features(df)
    df = add_recent_changes(df)
    df = add_station_profiles(df, stations, types, demand)
    df = add_weather(df, weather)

    # Plain numbers for every feature (True/False become 1/0, missing becomes NaN),
    # which is what the models expect
    df[FEATURES] = df[FEATURES].astype("float64")
    return df[["time", "station_id", "short_name"] + FEATURES + list(TARGETS.values())]


def main():
    labeled = pd.read_parquet(LABELED_PATH)
    check_profiles_come_first(labeled["time"].min())

    df = build_features(
        labeled=labeled,
        stations=load_stations(),
        types=pd.read_parquet(TYPES_PATH),
        demand=pd.read_parquet(DEMAND_PATH),
        weather=pd.read_parquet(WEATHER_PATH),
    )
    df.to_parquet(OUTPUT_PATH, index=False)

    local = df["time"].dt.tz_convert(config.TIMEZONE)
    print(f"{len(df):,} rows, {len(FEATURES)} features")
    print(f"From {local.min():%a %b %d %H:%M} to {local.max():%a %b %d %H:%M} (New York time)")

    missing = df[FEATURES].isna().mean().sort_values(ascending=False)
    print("\nFeatures with missing values:")
    print((missing[missing > 0] * 100).round(1).astype(str).add("%").to_string())
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()