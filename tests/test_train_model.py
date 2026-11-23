import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from build_features import FEATURES, TARGET
from evaluate import score
from train_model import calibrate, importance, train


def test_model_learns_a_simple_pattern():
    rng = np.random.default_rng(0)
    rows = 3000
    df = pd.DataFrame({feature: rng.random(rows) for feature in FEATURES})
    # Trouble is much more likely when only a few bikes are left
    df["bikes_available"] = rng.integers(0, 15, rows).astype(float)
    chance = np.where(df["bikes_available"] < 3, 0.7, 0.05)
    df[TARGET] = (rng.random(rows) < chance).astype(int)

    train_rows, validation_rows, test_rows = df[:2000], df[2000:2500], df[2500:]
    model = train(train_rows, validation_rows)
    risk = model.predict_proba(test_rows[FEATURES])[:, 1]

    assert score(test_rows[TARGET], risk)["PR-AUC"] > 0.5  # random would be about 0.18
    assert importance(model).index[0] == "bikes_available"  # it found the right feature


class TimidModel:
    """A fake model that ranks well but squeezes every score towards 0.5.

    For this test, fill_ratio holds each row's true chance of trouble.
    """

    def predict_proba(self, rows):
        timid = 0.4 + 0.2 * rows["fill_ratio"].to_numpy()
        return np.column_stack([1 - timid, timid])


def test_calibration_turns_timid_scores_into_honest_probabilities():
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({feature: 0.0 for feature in FEATURES}, index=range(20_000))
    rows["fill_ratio"] = rng.random(20_000)
    rows[TARGET] = (rng.random(20_000) < rows["fill_ratio"]).astype(int)
    calibrate_on, test_on = rows[:10_000], rows[10_000:]

    calibrator = calibrate(TimidModel(), calibrate_on)
    timid = TimidModel().predict_proba(test_on)[:, 1]
    honest = calibrator.predict(timid)

    assert brier_score_loss(test_on[TARGET], honest) < brier_score_loss(test_on[TARGET], timid)
    assert abs(honest.max() - 1) < 0.1  # it now dares to say "very likely"