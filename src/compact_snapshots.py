# Combines each day's 5-minute snapshot files into one file per day.
#
# The collector saves 288 small files a day. Reading weeks of data would mean
# opening thousands of files, which is slow. One file per day is much faster to
# read. The original small files are left as they are.
#
# Only finished days are compacted (not today, which is still being collected),
# and days that are already done are skipped, so it's safe to run any time.
#
# Usage:
#   uv run python src/compact_snapshots.py
#
# Output, one file per day:
#   data/processed/station_status/station_status_2026-11-19.parquet

from datetime import UTC, datetime

import pandas as pd

import config

RAW_DIR = config.RAW_DATA_DIR / "station_status"
OUTPUT_DIR = config.PROCESSED_DATA_DIR / "station_status"


def finished_days(today):
    """Days that have snapshots and are already over, like ["2026-11-18", "2026-11-19"].

    Dates are in UTC, the same as the collector's date folders.
    """
    days = []
    for folder in sorted(RAW_DIR.glob("date=*")):
        day = folder.name.removeprefix("date=")
        if day < today:  # "2026-11-19" < "2026-11-21" works for dates written this way
            days.append(day)
    return days


def compact_day(day):
    """Read all of one day's snapshot files and combine them into one table."""
    files = sorted((RAW_DIR / f"date={day}").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)

    # Each station should appear once per snapshot. If the same snapshot was
    # ever saved twice, keep just one copy.
    df = df.drop_duplicates(["snapshot_time", "station_id"])
    return df.sort_values(["snapshot_time", "station_id"], ignore_index=True)


def main():
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for day in finished_days(today):
        output_path = OUTPUT_DIR / f"station_status_{day}.parquet"
        if output_path.exists():
            print(f"{day}: already compacted")
            continue

        df = compact_day(day)
        df.to_parquet(output_path, index=False)

        size_mb = output_path.stat().st_size / 1_000_000
        snapshots = df["snapshot_time"].nunique()
        print(f"{day}: {snapshots} snapshots, {len(df):,} rows, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
