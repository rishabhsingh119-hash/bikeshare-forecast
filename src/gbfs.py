# Functions for reading GBFS bike-share feeds.
#
# GBFS comes in a few versions (1.x, 2.x and 3.x) that name some fields
# differently, so all of those differences are handled here in one place.
#
# Run this file directly to try it on the live feed:
#   uv run python src/gbfs.py

from datetime import UTC, datetime

import pandas as pd
import requests

import config


def get_feed_urls(gbfs_json, language="en"):
    """Return a dict like {"station_status": "https://...", ...} from gbfs.json."""
    data = gbfs_json["data"]

    if "feeds" in data:
        # Version 3 lists the feeds directly
        feeds = data["feeds"]
    elif language in data:
        # Versions 1 and 2 group the feeds by language
        feeds = data[language]["feeds"]
    else:
        # Our language isn't there, so use the first one
        first_language = list(data)[0]
        feeds = data[first_language]["feeds"]

    return {feed["name"]: feed["url"] for feed in feeds}


def parse_station_status(status_json, snapshot_time):
    """Turn a station_status feed into a table with one row per station."""
    rows = []
    for station in status_json["data"]["stations"]:
        rows.append(
            {
                "snapshot_time": snapshot_time,
                "station_id": str(station["station_id"]),
                # Version 3 renamed "bikes" to "vehicles"
                "bikes_available": station.get(
                    "num_bikes_available", station.get("num_vehicles_available")
                ),
                "ebikes_available": station.get("num_ebikes_available"),
                "bikes_disabled": station.get(
                    "num_bikes_disabled", station.get("num_vehicles_disabled")
                ),
                "docks_available": station.get("num_docks_available"),
                "docks_disabled": station.get("num_docks_disabled"),
                "is_installed": to_bool(station.get("is_installed")),
                "is_renting": to_bool(station.get("is_renting")),
                "is_returning": to_bool(station.get("is_returning")),
                "last_reported": station.get("last_reported"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)
    df["last_reported"] = to_timestamps(df["last_reported"])

    # Whole numbers that can be missing
    for col in [
        "bikes_available",
        "ebikes_available",
        "bikes_disabled",
        "docks_available",
        "docks_disabled",
    ]:
        df[col] = df[col].astype("Int64")

    # True/False values that can be missing
    for col in ["is_installed", "is_renting", "is_returning"]:
        df[col] = df[col].astype("boolean")

    return df


def parse_station_info(info_json, snapshot_time, language="en"):
    """Turn a station_information feed into a table of names, locations and capacity."""
    rows = []
    for station in info_json["data"]["stations"]:
        rows.append(
            {
                "snapshot_time": snapshot_time,
                "station_id": str(station["station_id"]),
                "name": get_text(station.get("name"), language),
                # The station ID used in Citi Bike's trip history files
                "short_name": get_text(station.get("short_name"), language),
                "lat": station.get("lat"),
                "lon": station.get("lon"),
                "capacity": station.get("capacity"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)
        df["capacity"] = df["capacity"].astype("Int64")
    return df


def to_bool(value):
    # Version 1 uses 1/0 instead of true/false
    if value is None:
        return None
    return bool(value)


def to_timestamps(column):
    # Versions 1 and 2 send Unix seconds (1763303700),
    # version 3 sends text ("2026-11-16T14:35:00Z")
    if pd.api.types.is_numeric_dtype(column):
        return pd.to_datetime(column, unit="s", utc=True)
    return pd.to_datetime(column, utc=True, errors="coerce")


def get_text(value, language="en"):
    # Version 3 stores names as a list of translations:
    # [{"text": "W 52 St", "language": "en"}, ...]
    if not isinstance(value, list):
        return value
    for item in value:
        if item.get("language", "").startswith(language):
            return item["text"]
    return value[0]["text"] if value else None


def preview():
    """Download the live feed and print a quick look at it."""
    feed_urls = get_feed_urls(requests.get(config.GBFS_URL, timeout=20).json())
    print("Feeds available:", ", ".join(feed_urls))

    now = datetime.now(UTC)
    info_json = requests.get(feed_urls["station_information"], timeout=20).json()
    status_json = requests.get(feed_urls["station_status"], timeout=20).json()
    info = parse_station_info(info_json, now)
    status = parse_station_status(status_json, now)

    print(f"\n{len(info)} stations. A few of them:")
    print(info[["station_id", "name", "capacity"]].head())

    # Closed stations show 0 bikes AND 0 docks, which would make them count
    # as both empty and full, so only look at stations that are open
    open_stations = status[status["is_installed"] & status["is_renting"]]
    print(f"\n{len(open_stations)} of {len(status)} stations are open right now:")
    print(
        open_stations[
            ["station_id", "bikes_available", "ebikes_available", "docks_available"]
        ].head()
    )

    print(f"\nEmpty right now: {(open_stations['bikes_available'] == 0).sum()} stations")
    print(f"Full right now:  {(open_stations['docks_available'] == 0).sum()} stations")


if __name__ == "__main__":
    preview()
