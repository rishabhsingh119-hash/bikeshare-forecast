# Scores a few simple ways of predicting trouble, to set the bar a real model
# has to beat.
#
# Usage:
#   uv run python src/baselines.py

import numpy as np
import pandas as pd

import config
from build_features import OUTPUT_PATH as FEATURES_PATH
from build_features import TARGET
from evaluate import only_fine_now, print_scores, score, split_by_time


def few_bikes_or_docks(df):
    """The simple rule: the fewer bikes or free docks a station has, the riskier it is."""
    return -np.minimum(df["bikes_available"], df["docks_available"])


def few_bikes_or_docks_after_usual_demand(df):
    """Like the simple rule, but first moves the counts by what usually happens next hour."""
    usual_change = (df["usual_arrivals_next_hour"] - df["usual_departures_next_hour"]).fillna(0)
    bikes_later = df["bikes_available"] + usual_change
    docks_later = df["docks_available"] - usual_change
    return -np.minimum(bikes_later, docks_later)


def describe(name, df):
    local = df["time"].dt.tz_convert(config.TIMEZONE)
    print(
        f"{name}: {len(df):,} rows, {local.min():%a %H:%M} to {local.max():%a %H:%M}, "
        f"{df[TARGET].mean():.1%} ran into trouble"
    )


def main():
    df = pd.read_parquet(FEATURES_PATH)
    train, test = split_by_time(df)
    train, test = only_fine_now(train), only_fine_now(test)
    describe("Train", train)
    describe("Test ", test)

    # The baselines don't learn anything, so they only need the test data
    answers = test[TARGET].astype(int)
    rng = np.random.default_rng(42)
    results = {
        "Random guess": score(answers, rng.random(len(test))),
        "Rule: few bikes or docks": score(answers, few_bikes_or_docks(test)),
        "Rule + usual demand": score(answers, few_bikes_or_docks_after_usual_demand(test)),
    }
    print()
    print_scores(results)


if __name__ == "__main__":
    main()