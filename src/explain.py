# Explains the model's predictions: which features pushed a station's risk up or down.
#
# It uses SHAP values, which LightGBM can work out by itself. For every prediction,
# each feature gets a number: positive means it made trouble more likely, negative
# means less likely, and bigger means a stronger push. Added together with a
# starting value, they give exactly the model's score.
#
# Usage:
#   uv run python src/explain.py

import joblib
import lightgbm as lgb
import pandas as pd

import config
from build_features import FEATURES, TARGET
from build_features import OUTPUT_PATH as FEATURES_PATH
from clean_snapshots import load_stations
from evaluate import only_fine_now, split_by_time
from train_model import CALIBRATOR_PATH, MODEL_PATH


def shap_values(booster, rows):
    """One SHAP value for every feature of every row."""
    contributions = booster.predict(rows[FEATURES], pred_contrib=True)
    # LightGBM adds one extra last column: the starting value. We only want the features.
    return pd.DataFrame(contributions[:, :-1], columns=FEATURES, index=rows.index)


def top_reasons(row_shap, row_features, top=3):
    """The features that pushed this prediction up the most: (feature, its value, the push)."""
    biggest = row_shap.sort_values(ascending=False).head(top)
    return [(feature, row_features[feature], push) for feature, push in biggest.items()]


def main():
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    calibrator = joblib.load(CALIBRATOR_PATH)
    _, test = split_by_time(pd.read_parquet(FEATURES_PATH))
    test = only_fine_now(test)

    # The whole picture: on average, how much does each feature move the predictions?
    # Working out SHAP values is slow, so a random sample of rows is plenty.
    sample = test.sample(min(len(test), 50_000), random_state=42)
    overall = shap_values(booster, sample).abs().mean().sort_values(ascending=False)
    print("Average push on predictions (mean absolute SHAP value):")
    print(overall.head(10).round(3).to_string())

    # Single predictions: the riskiest stations at the last snapshot of the test
    # day, as if it were right now
    latest = test[test["time"] == test["time"].max()].copy()
    latest["risk"] = calibrator.predict(booster.predict(latest[FEATURES]))
    riskiest = latest.nlargest(5, "risk")
    shap = shap_values(booster, riskiest)
    names = load_stations().set_index("station_id")["name"]

    when = riskiest["time"].iloc[0].tz_convert(config.TIMEZONE)
    print(f"\nRiskiest stations at {when:%a %b %d, %H:%M}:")
    for index, row in riskiest.iterrows():
        what_happened = "and it did" if row[TARGET] == 1 else "but it didn't"
        print(f"\n{names.get(row['station_id'], row['station_id'])}")
        print(f"  {row['risk']:.0%} chance of trouble within the hour ({what_happened})")
        for feature, value, push in top_reasons(shap.loc[index], row):
            print(f"  because {feature} = {value:.2f}  (push +{push:.2f})")


if __name__ == "__main__":
    main()