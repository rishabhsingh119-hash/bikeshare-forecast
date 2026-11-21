# Cleans the compacted station snapshots so they only contain rows we can trust.
#
# Rows are removed when:
#   - the station isn't in the station list (no name, location or capacity)
#   - the station is closed (not installed, or not renting or returning bikes)
#   - most of the station is taken up by broken bikes or docks
#   - the bike or dock count is missing or negative
#   - the station hasn't reported in over an hour, so its numbers are probably frozen
#
# It also adds the station's capacity, and whether it's empty or full right now.
#
# Usage:
#   uv run python src/clean_snapshots.py
#
# Output, one file per day:
#   data/processed/station_status_clean/station_status_clean_2026-11-19.parquet

from collections import Counter

import pandas as pd

import config

INFO_DIR = config.RAW_DATA_DIR / "station_information"
INPUT_DIR = config.PROCESSED_DATA_DIR / "station_status"
OUTPUT_DIR = config.PROCESSED_DATA_DIR / "station_status_clean"

# A working station checks in every few minutes. Silence for longer than this
# usually means it lost its connection and its numbers stopped updating.
MAX_REPORT_AGE = pd.Timedelta(hours=1)


def load_stations():
    """The latest known details (name, location, capacity) for every station.

    Uses all saved station_information files, so stations that have since been
    removed are still included.
    """
    files = sorted(INFO_DIR.glob("date=*/*.parquet"))
    info = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    info = info.sort_values("snapshot_time")
    return info.drop_duplicates("station_id", keep="last").drop(columns="snapshot_time")


def drop_rows(df, keep, reason, report):
    """Keep the rows where `keep` is True, and note how many were removed and why."""
    report[reason] = int((~keep).sum())
    return df[keep]


def clean_snapshots(status, stations):
    """Remove rows we can't trust. Returns the clean table and a count of what was removed."""
    report = {"rows": len(status)}
    df = status

    known = df["station_id"].isin(stations["station_id"])
    df = drop_rows(df, known, "unknown station", report)
    df = df.merge(stations[["station_id", "capacity"]], on="station_id", how="left")

    is_open = (
        df["is_installed"].fillna(True)
        & df["is_renting"].fillna(True)
        & df["is_returning"].fillna(True)
    )
    df = drop_rows(df, is_open, "station closed", report)

    # A station filled up with broken bikes (or broken docks) can't be used, even
    # though it says it's open. That's a repair problem, not a riding pattern.
    broken = df["bikes_disabled"].fillna(0) + df["docks_disabled"].fillna(0)
    mostly_broken = (broken > df["capacity"] / 2).fillna(False)
    df = drop_rows(df, ~mostly_broken, "mostly broken", report)

    has_counts = df["bikes_available"].notna() & df["docks_available"].notna()
    df = drop_rows(df, has_counts, "missing counts", report)

    not_negative = (df["bikes_available"] >= 0) & (df["docks_available"] >= 0)
    df = drop_rows(df, not_negative, "negative counts", report)

    report_age = df["snapshot_time"] - df["last_reported"]
    still_reporting = ~(report_age > MAX_REPORT_AGE)
    df = drop_rows(df, still_reporting, "stopped reporting", report)

    df["is_empty"] = df["bikes_available"] == 0
    df["is_full"] = df["docks_available"] == 0
    return df.reset_index(drop=True), report


def print_report(totals):
    rows = totals.pop("rows")
    print(f"\nChecked {rows:,} rows")
    for reason, removed in totals.items():
        print(f"  {reason:<18} {removed:>10,} removed ({removed / rows:.1%})")
    kept = rows - sum(totals.values())
    print(f"  {'kept':<18} {kept:>10,}         ({kept / rows:.1%})")


def main():
    stations = load_stations()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    totals = Counter()

    # Cleaning is quick, so every day is redone. That way a change to the
    # rules above is applied to all the data.
    for path in sorted(INPUT_DIR.glob("station_status_*.parquet")):
        day = path.stem.removeprefix("station_status_")
        clean, report = clean_snapshots(pd.read_parquet(path), stations)
        clean.to_parquet(OUTPUT_DIR / f"station_status_clean_{day}.parquet", index=False)

        totals.update(report)
        print(f"{day}: kept {len(clean):,} of {report['rows']:,} rows")

    print_report(totals)


if __name__ == "__main__":
    main()