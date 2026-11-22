import pandas as pd

import build_station_profiles
from build_station_profiles import station_types, usual_demand


def trip_rows(short_name, rows):
    """rows: (local time, departures, arrivals)"""
    df = pd.DataFrame(rows, columns=["local_time", "departures", "arrivals"])
    local = pd.to_datetime(df["local_time"]).dt.tz_localize("America/New_York")
    df["short_name"] = short_name
    df["date"] = local.dt.normalize()
    df["hour_of_day"] = local.dt.hour
    df["day_off"] = local.dt.day_of_week >= 5
    return df.drop(columns="local_time")


def test_usual_demand_counts_quiet_days_too():
    # Two working days in use. 8 AM had 4 departures on the first day and none on
    # the second, so the usual number at 8 AM is 2 a day, not 4.
    trips = trip_rows("A", [
        ("2026-11-16 08:00", 4, 0),  # Monday
        ("2026-11-17 09:00", 1, 1),  # Tuesday: no trips at 8 AM
    ])
    demand = usual_demand(trips).set_index(["short_name", "day_off", "hour_of_day"])

    assert demand.loc[("A", False, 8), "usual_departures"] == 2
    assert demand.loc[("A", False, 9), "usual_arrivals"] == 0.5


def commuter_day(short_name, morning, evening):
    """One working day: `morning` and `evening` are (departures, arrivals) at 8 AM and 6 PM."""
    return trip_rows(short_name, [
        ("2026-11-16 08:00", *morning),
        ("2026-11-16 12:00", 3, 3),
        ("2026-11-16 18:00", *evening),
    ])


def test_station_types_go_from_residential_to_workplace(monkeypatch):
    monkeypatch.setattr(build_station_profiles, "NUMBER_OF_TYPES", 3)
    monkeypatch.setattr(build_station_profiles, "MIN_TRIPS_FOR_TYPE", 1)
    trips = pd.concat([
        commuter_day("home1", morning=(10, 1), evening=(1, 10)),  # bikes leave in the morning
        commuter_day("home2", morning=(12, 1), evening=(1, 11)),
        commuter_day("mixed1", morning=(5, 5), evening=(5, 5)),
        commuter_day("mixed2", morning=(6, 6), evening=(4, 4)),
        commuter_day("work1", morning=(1, 10), evening=(10, 1)),  # bikes arrive in the morning
        commuter_day("work2", morning=(1, 12), evening=(11, 1)),
    ])

    types, summary = station_types(trips)
    types = types.set_index("short_name")["station_type"]

    assert types["home1"] == types["home2"] == 0  # loses the most bikes in the morning
    assert types["mixed1"] == types["mixed2"] == 1
    assert types["work1"] == types["work2"] == 2  # gains the most bikes in the morning
    assert list(summary["stations"]) == [2, 2, 2]