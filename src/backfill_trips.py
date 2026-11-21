# Downloads Citi Bike's trip history and turns it into hourly counts per station:
# how many bikes left each station, and how many arrived, in every hour.
#
# Usage:
#   uv run python src/backfill_trips.py --start 202510 --end 202510   # one month
#   uv run python src/backfill_trips.py --start 202511 --end 202610   # a year
#
# Output, one file per month:
#   data/processed/trips_hourly/trips_hourly_202510.parquet
#
# Each month's zip is deleted once it's processed, since it can be downloaded again.

import argparse
import io
import time
import zipfile

import pandas as pd
import requests

import config
from gbfs import get_feed_urls, parse_station_info

TRIPDATA_URL = "https://s3.amazonaws.com/tripdata"
TRIPS_DIR = config.RAW_DATA_DIR / "trips"
OUTPUT_DIR = config.PROCESSED_DATA_DIR / "trips_hourly"

# We only need these columns, which keeps memory use down
COLUMNS = ["ride_id", "started_at", "ended_at", "start_station_id", "end_station_id"]


def month_list(start, end):
    """month_list("202511", "202602") -> ["202511", "202512", "202601", "202602"]"""
    months = pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M")
    return [month.strftime("%Y%m") for month in months]


def download_month(month, retries=3):
    """Download one month's zip file, unless we already have it. Returns its path."""
    path = TRIPS_DIR / f"{month}-citibike-tripdata.zip"
    if path.exists():
        print(f"{month}: already downloaded")
        return path

    url = f"{TRIPDATA_URL}/{month}-citibike-tripdata.zip"
    for attempt in range(1, retries + 1):
        try:
            print(f"{month}: downloading {url}")
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code in (403, 404):
                print(f"{month}: not published yet, skipping")
                return None
            response.raise_for_status()

            # Download in pieces (the files are hundreds of MB), into a temp file first
            TRIPS_DIR.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            temp_path.replace(path)
            return path

        except requests.RequestException as error:
            # Big downloads sometimes stall. Wait a bit and start that file again.
            if attempt == retries:
                raise
            wait = 10 * attempt
            print(f"{month}: download failed ({error}). Trying again in {wait}s")
            time.sleep(wait)


def read_trips(zip_path):
    """Read every CSV inside a month's zip file into one table of trips."""
    tables = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            # Skip folders and the hidden files Macs add to zips
            if not name.endswith(".csv") or name.startswith("__MACOSX"):
                continue
            with zf.open(name) as f:
                # Read station IDs as text: as numbers, "6140.10" would become 6140.1
                text = io.TextIOWrapper(f, encoding="utf-8")
                tables.append(pd.read_csv(text, usecols=COLUMNS, dtype=str))

    if not tables:
        raise ValueError(f"No CSV files found in {zip_path}")

    trips = pd.concat(tables, ignore_index=True)
    # Some zips contain the same trips twice, so keep each ride only once
    return trips.drop_duplicates("ride_id")


def to_utc(local_times):
    """Trip times are New York local time. Convert them to UTC like the rest of our data.

    When clocks go back in November, 1:00-2:00 AM happens twice, so those times are
    ambiguous. We drop them (marked NaT) rather than guess.
    """
    times = pd.to_datetime(local_times, format="ISO8601")
    times = times.dt.tz_localize(config.TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")
    return times.dt.tz_convert("UTC")


def hourly_counts(trips):
    """Count departures and arrivals for every station in every hour."""
    departures = (
        pd.DataFrame(
            {
                "station_id": trips["start_station_id"],
                "hour": to_utc(trips["started_at"]).dt.floor("h"),
            }
        )
        .dropna()
        .value_counts()
        .rename("departures")
    )
    arrivals = (
        pd.DataFrame(
            {
                "station_id": trips["end_station_id"],
                "hour": to_utc(trips["ended_at"]).dt.floor("h"),
            }
        )
        .dropna()
        .value_counts()
        .rename("arrivals")
    )

    # Put both counts side by side. A station-hour with departures but no
    # arrivals gets 0 arrivals, and the other way round.
    counts = pd.concat([departures, arrivals], axis=1).fillna(0).astype(int)
    return counts.reset_index().sort_values(["hour", "station_id"], ignore_index=True)


def check_station_ids(counts):
    """Check how many trip station IDs match the live feed's station short names."""
    feed_urls = get_feed_urls(requests.get(config.GBFS_URL, timeout=20).json())
    info_json = requests.get(feed_urls["station_information"], timeout=20).json()
    live_ids = set(parse_station_info(info_json, pd.Timestamp.now(tz="UTC"))["short_name"])

    trip_ids = set(counts["station_id"])
    matched = len(trip_ids & live_ids)
    print(
        f"\n{matched} of {len(trip_ids)} trip stations match a live station "
        f"({matched / len(trip_ids):.0%})"
    )


def main():
    parser = argparse.ArgumentParser(description="Turn trip history into hourly counts.")
    parser.add_argument("--start", required=True, help="first month, like 202511")
    parser.add_argument("--end", required=True, help="last month, like 202610")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    last_output = None

    for month in month_list(args.start, args.end):
        output_path = OUTPUT_DIR / f"trips_hourly_{month}.parquet"
        if output_path.exists():
            # Already done on an earlier run, so don't redo the work
            print(f"{month}: already processed")
            last_output = output_path
            continue

        zip_path = download_month(month)
        if zip_path is None:
            continue

        trips = read_trips(zip_path)
        counts = hourly_counts(trips)
        counts.to_parquet(output_path, index=False)
        print(f"{month}: {len(trips):,} trips -> {len(counts):,} station-hours")
        last_output = output_path

        # The zip can be downloaded again any time, so free up the disk space
        zip_path.unlink()

    if last_output is not None:
        check_station_ids(pd.read_parquet(last_output))


if __name__ == "__main__":
    main()
