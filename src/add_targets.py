# Adds the answer the model will learn to predict: for every station at every
# snapshot, does it become empty or full at some point in the next hour?
#
# Usage:
#   uv run python src/add_targets.py
#
# Output:
#   data/processed/labeled_snapshots.parquet

import pandas as pd

import config

INPUT_DIR = config.PROCESSED_DATA_DIR / "station_status_clean"
OUTPUT_PATH = config.PROCESSED_DATA_DIR / "labeled_snapshots.parquet"

HORIZON = pd.Timedelta(minutes=60)  # how far ahead we predict
SLOT = pd.Timedelta(seconds=config.POLL_INTERVAL)  # one snapshot every 5 minutes

# The station's state right now -> the answer column: does it happen in the next hour?
TARGETS = {
    "is_empty": "empty_next_hour",
    "is_full": "full_next_hour",
    "is_empty_or_full": "empty_or_full_next_hour",
}


def to_slots(df):
    """Line snapshots up on the 5-minute grid (10:07:56 -> 10:05), one row per station per slot."""
    df = df.copy()
    df["time"] = df["snapshot_time"].dt.floor(SLOT)

    # If two snapshots land in the same slot (this happens when the collector
    # restarts), keep the later one
    df = df.sort_values("snapshot_time").drop_duplicates(["time", "station_id"], keep="last")

    df["is_empty_or_full"] = df["is_empty"] | df["is_full"]
    for flag in TARGETS:
        df[flag] = df[flag].astype(int)  # 1/0 instead of True/False, easier to do maths on
    return df.drop(columns="snapshot_time").reset_index(drop=True)


def look_ahead(df, flag):
    """For every station and time slot: is `flag` true at any point in the next hour?

    Returns 1 (yes), 0 (no), or nothing when we can't tell because snapshots are
    missing from that hour.
    """
    steps = HORIZON // SLOT  # 12 snapshots in an hour

    # A grid with one row per time slot and one column per station.
    # Missing snapshots become gaps (NaN) instead of being skipped over.
    wide = df.pivot(index="time", columns="station_id", values=flag)
    all_slots = pd.date_range(wide.index.min(), wide.index.max(), freq=SLOT, name="time")
    wide = wide.reindex(all_slots)

    # Rolling windows look backwards in time, so flip the grid upside down to
    # look forwards instead. Then shift by one slot so "the next hour" doesn't
    # include right now.
    flipped = wide[::-1]
    any_true = flipped.rolling(steps, min_periods=1).max()[::-1].shift(-1)
    seen = flipped.rolling(steps, min_periods=1).count()[::-1].shift(-1)

    # A "yes" counts as soon as we see it. A "no" only counts if we saw
    # every snapshot in that hour.
    answer = any_true.where((any_true == 1) | (seen == steps))
    return answer.stack().dropna().rename(TARGETS[flag])


def add_targets(df):
    """Add the three answer columns to the cleaned snapshots."""
    df = to_slots(df)
    answers = pd.concat([look_ahead(df, flag) for flag in TARGETS], axis=1).reset_index()
    df = df.merge(answers, on=["time", "station_id"], how="left")

    for column in TARGETS.values():
        df[column] = df[column].astype("Int64")

    # Put the time and station first, so the table is easy to read
    first = ["time", "station_id"]
    df = df[first + [column for column in df.columns if column not in first]]

    # Without the main answer, a row is no use for training
    return df.dropna(subset=["empty_or_full_next_hour"]).reset_index(drop=True)


def print_summary(df):
    print(f"Labeled {len(df):,} rows")

    # The real question: of the stations that are fine right now,
    # how many run into trouble within the hour?
    fine_now = df[df["is_empty_or_full"] == 0]
    print(f"\nStations that are fine right now: {len(fine_now):,} rows")
    print(f"  become empty within the hour:       {fine_now['empty_next_hour'].mean():.1%}")
    print(f"  become full within the hour:        {fine_now['full_next_hour'].mean():.1%}")
    print(f"  become empty or full within the hour: "
          f"{fine_now['empty_or_full_next_hour'].mean():.1%}")


def main():
    files = sorted(INPUT_DIR.glob("station_status_clean_*.parquet"))
    snapshots = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)

    labeled = add_targets(snapshots)
    labeled.to_parquet(OUTPUT_PATH, index=False)

    print_summary(labeled)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()