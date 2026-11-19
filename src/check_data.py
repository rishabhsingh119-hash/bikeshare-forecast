# Quick health check on the snapshots the collector has saved.
#
# Usage:
#   uv run python src/check_data.py            # check the latest day
#   uv run python src/check_data.py --days 7   # check the last 7 days

import argparse

import pandas as pd

import config


def load_snapshots(days):
    """Load every station_status file from the last few days into one table."""
    folder = config.RAW_DATA_DIR / "station_status"
    day_folders = sorted(folder.glob("date=*"))[-days:]
    files = [file for day in day_folders for file in sorted(day.glob("*.parquet"))]
    if not files:
        raise SystemExit(f"No data found in {folder}. Is the collector running?")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def summarize(df, poll_interval):
    """Work out how complete the data is and how often stations are empty or full."""
    times = df["snapshot_time"].drop_duplicates().sort_values().reset_index(drop=True)

    # Round each snapshot down to its 5-minute slot (10:07:56 -> 10:05), then
    # compare with how many slots there are between the first and last one
    slots = times.dt.floor(f"{poll_interval}s")
    expected = int((slots.iloc[-1] - slots.iloc[0]).total_seconds() // poll_interval) + 1

    # A gap is any time we went more than two polls without a snapshot
    gaps = []
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > pd.Timedelta(seconds=poll_interval * 2):
            gaps.append((times[i - 1], times[i]))

    # Closed stations show 0 bikes AND 0 docks, so only count open ones
    is_open = df["is_installed"].fillna(True) & df["is_renting"].fillna(True)
    open_stations = df[is_open]

    return {
        "start": times.iloc[0],
        "end": times.iloc[-1],
        "collected": slots.nunique(),
        "expected": expected,
        "stations": df["station_id"].nunique(),
        "open_stations": open_stations["station_id"].nunique(),
        "empty_pct": (open_stations["bikes_available"] == 0).mean() * 100,
        "full_pct": (open_stations["docks_available"] == 0).mean() * 100,
        "gaps": gaps,
    }


def local(timestamp):
    """Show a UTC time in the city's own time zone, e.g. 'Nov 18 15:15'."""
    return f"{timestamp.tz_convert(config.TIMEZONE):%b %d %H:%M}"


def print_report(s):
    coverage = s["collected"] / s["expected"] * 100
    print(f"Period:     {local(s['start'])} to {local(s['end'])} ({config.TIMEZONE} time)")
    print(f"Snapshots:  {s['collected']} of {s['expected']} expected ({coverage:.0f}%)")
    print(f"Stations:   {s['stations']} total, {s['open_stations']} open")
    print(f"Empty:      {s['empty_pct']:.1f}% of the time (open stations only)")
    print(f"Full:       {s['full_pct']:.1f}% of the time (open stations only)")

    if not s["gaps"]:
        print("Gaps:       none")
        return
    print(f"Gaps:       {len(s['gaps'])}")
    for start, end in s["gaps"][:10]:
        print(f"  no data from {local(start)} to {local(end)}")


def main():
    parser = argparse.ArgumentParser(description="Check the collected data.")
    parser.add_argument("--days", type=int, default=1, help="how many days to check")
    args = parser.parse_args()

    df = load_snapshots(args.days)
    print_report(summarize(df, config.POLL_INTERVAL))


if __name__ == "__main__":
    main()
