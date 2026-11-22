# Builds two things from a year of trip history, for the model to use:
#
# 1. Each station's usual demand: how many bikes usually leave and arrive in
#    every hour, on working days and on days off (weekends and holidays).
# 2. Each station's type, found by grouping stations with a similar daily
#    rhythm, as explored in notebooks/02_trips_and_weather.ipynb.
#
# Stations are identified by their short name (like "6140.05"), because that's
# the ID the trip files use.
#
# Usage:
#   uv run python src/build_station_profiles.py
#
# Output:
#   data/processed/usual_demand.parquet
#   data/processed/station_types.parquet

import pandas as pd
from sklearn.cluster import KMeans

import config
from days import is_day_off

TRIPS_DIR = config.PROCESSED_DATA_DIR / "trips_hourly"
DEMAND_PATH = config.PROCESSED_DATA_DIR / "usual_demand.parquet"
TYPES_PATH = config.PROCESSED_DATA_DIR / "station_types.parquet"

NUMBER_OF_TYPES = 4
MIN_TRIPS_FOR_TYPE = 1000  # quieter stations have too few trips for a reliable rhythm


def load_trips():
    """All the hourly trip counts, with the local date, hour of day, and kind of day."""
    trips = pd.read_parquet(TRIPS_DIR).rename(columns={"station_id": "short_name"})

    # A trip that crosses midnight at the end of a month can be in two monthly
    # files, so add up any station-hour that appears twice
    trips = trips.groupby(["short_name", "hour"], as_index=False)[["departures", "arrivals"]].sum()

    local = trips["hour"].dt.tz_convert(config.TIMEZONE)
    trips["date"] = local.dt.normalize()
    trips["hour_of_day"] = local.dt.hour
    trips["day_off"] = is_day_off(local)
    return trips


def usual_demand(trips):
    """Average departures and arrivals for each station, kind of day, and hour of day.

    Hours with no trips aren't in the trip files at all, so averaging the rows
    would overstate demand. Instead we divide the total by the number of days
    the station was in use.
    """
    keys = ["short_name", "day_off", "hour_of_day"]
    totals = trips.groupby(keys, as_index=False)[["departures", "arrivals"]].sum()
    days_in_use = (
        trips.groupby(["short_name", "day_off"])["date"].nunique().rename("days").reset_index()
    )

    demand = totals.merge(days_in_use, on=["short_name", "day_off"])
    demand["usual_departures"] = demand["departures"] / demand["days"]
    demand["usual_arrivals"] = demand["arrivals"] / demand["days"]
    return demand[keys + ["usual_departures", "usual_arrivals"]]


def station_types(trips):
    """Group stations by their working-day rhythm. Returns the types and a summary of each."""
    working_days = trips[~trips["day_off"]]
    by_hour = working_days.groupby(["short_name", "hour_of_day"])[["departures", "arrivals"]].sum()
    totals = by_hour.groupby("short_name").sum().sum(axis=1)
    busy = totals[totals >= MIN_TRIPS_FOR_TYPE].index

    # Each station's share of its own trips in each hour, so big and small
    # stations with the same rhythm look the same
    profiles = by_hour.loc[busy].unstack(fill_value=0)
    every_hour = pd.MultiIndex.from_product([["departures", "arrivals"], range(24)])
    profiles = profiles.reindex(columns=every_hour, fill_value=0)  # even hours with no trips
    shares = profiles.div(profiles.sum(axis=1), axis=0)

    kmeans = KMeans(n_clusters=NUMBER_OF_TYPES, n_init=10, random_state=42)
    groups = kmeans.fit_predict(shares.to_numpy())

    # KMeans numbers its groups in no particular order. Renumber them so the type
    # goes up with how many bikes a station gains in the morning rush (7-9 AM):
    # the lowest type loses the most (residential), the highest gains the most (workplace).
    net = shares["arrivals"] - shares["departures"]
    morning = net[[7, 8, 9]].sum(axis=1)
    evening = net[[16, 17, 18]].sum(axis=1)
    new_numbers = morning.groupby(groups).mean().rank(method="first").astype(int) - 1

    types = pd.DataFrame({
        "short_name": shares.index,
        "station_type": new_numbers.loc[groups].to_numpy(),
    })
    summary = pd.DataFrame({
        "stations": types["station_type"].value_counts(),
        "morning_gain": morning.groupby(types["station_type"].values).mean(),
        "evening_gain": evening.groupby(types["station_type"].values).mean(),
    }).sort_index()
    return types, summary


def main():
    trips = load_trips()
    print(f"Trips from {trips['date'].min():%Y-%m-%d} to {trips['date'].max():%Y-%m-%d}")

    demand = usual_demand(trips)
    demand.to_parquet(DEMAND_PATH, index=False)
    print(f"Usual demand: {demand['short_name'].nunique():,} stations, {len(demand):,} rows")

    types, summary = station_types(trips)
    types.to_parquet(TYPES_PATH, index=False)
    print(f"\nStation types for {len(types):,} stations:")
    print("(gain = share of the station's trips that are extra arrivals; negative = more leaving)")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()