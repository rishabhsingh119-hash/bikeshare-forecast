# Trains a LightGBM model to predict which stations that are fine right now will
# become empty or full within the hour, compares it with the baselines, and
# calibrates it so its predictions can be read as real probabilities.
#
# Usage:
#   uv run python src/train_model.py
#
# Output:
#   data/models/lightgbm.txt          (the trained model)
#   data/models/calibrator.joblib     (turns its scores into honest probabilities)

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import config
from baselines import describe, few_bikes_or_docks, few_bikes_or_docks_after_usual_demand
from build_features import FEATURES, TARGET
from build_features import OUTPUT_PATH as FEATURES_PATH
from evaluate import calibration_table, only_fine_now, print_scores, score, split_by_time

MODEL_DIR = config.PROJECT_ROOT / "data" / "models"
MODEL_PATH = MODEL_DIR / "lightgbm.txt"
CALIBRATOR_PATH = MODEL_DIR / "calibrator.joblib"


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=1000,  # an upper limit; early stopping picks the real number
        learning_rate=0.05,  # how big a step each new tree takes
        num_leaves=31,  # how detailed each tree can be
        min_child_samples=100,  # every rule must apply to at least 100 rows
        subsample=0.8,  # each tree sees a random 80% of the rows...
        subsample_freq=1,
        colsample_bytree=0.8,  # ...and a random 80% of the features
        metric="average_precision",  # the same idea as PR-AUC
        importance_type="gain",
        random_state=42,
        verbose=-1,
    )


def train(train_rows, validation_rows):
    """Add trees until the score on the validation rows stops improving."""
    model = make_model()
    model.fit(
        train_rows[FEATURES],
        train_rows[TARGET].astype(int),
        eval_X=validation_rows[FEATURES],
        eval_y=validation_rows[TARGET].astype(int),
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model


def calibrate(model, validation_rows):
    """Learn how to turn the model's scores into honest probabilities.

    Isotonic regression looks at the validation rows and learns a mapping like
    "when the model says 0.30, trouble really happened 45% of the time". It only
    stretches or squeezes the scores, never reorders them. It must never see the
    test rows, or the test would no longer be a fair one.
    """
    scores = model.predict_proba(validation_rows[FEATURES])[:, 1]
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(scores, validation_rows[TARGET].astype(int))
    return calibrator


def importance(model):
    """Each feature's share of the model's total gain, biggest first."""
    gain = pd.Series(model.feature_importances_, index=FEATURES)
    return (gain / gain.sum() * 100).sort_values(ascending=False)


def main():
    df = pd.read_parquet(FEATURES_PATH)

    # Three parts, in time order: train on the oldest, use the next half day to
    # decide when to stop adding trees (and to calibrate), and keep the last day
    # for the final test
    rest, test = split_by_time(df, test_days=1)
    train_rows, validation_rows = split_by_time(rest, test_days=0.5)
    train_rows = only_fine_now(train_rows)
    validation_rows = only_fine_now(validation_rows)
    test = only_fine_now(test)
    describe("Train     ", train_rows)
    describe("Validation", validation_rows)
    describe("Test      ", test)

    model = train(train_rows, validation_rows)
    calibrator = calibrate(model, validation_rows)
    print(f"\nLightGBM stopped after {model.best_iteration_} trees")

    answers = test[TARGET].astype(int)
    model_scores = model.predict_proba(test[FEATURES])[:, 1]
    probabilities = calibrator.predict(model_scores)

    rng = np.random.default_rng(42)
    results = {
        "Random guess": score(answers, rng.random(len(test))),
        "Rule: few bikes or docks": score(answers, few_bikes_or_docks(test)),
        "Rule + usual demand": score(answers, few_bikes_or_docks_after_usual_demand(test)),
        "LightGBM": score(answers, model_scores),
        "LightGBM, calibrated": score(answers, probabilities),
    }
    print()
    print_scores(results)

    print("\nAre the probabilities honest? Test day, 10 groups from safest to riskiest (%):")
    table = pd.concat(
        {
            "before": calibration_table(answers, model_scores),
            "after": calibration_table(answers, probabilities),
        },
        axis=1,
    )
    print((table * 100).round(1).to_string())
    print(
        f"Brier score (lower is better): {brier_score_loss(answers, model_scores):.4f} before, "
        f"{brier_score_loss(answers, probabilities):.4f} after calibration"
    )

    print("\nWhat the model relies on most (share of total gain):")
    print(importance(model).head(10).round(1).astype(str).add("%").to_string())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH))
    joblib.dump(calibrator, CALIBRATOR_PATH)
    print(f"\nSaved the model and calibrator to {MODEL_DIR}")


if __name__ == "__main__":
    main()