import numpy as np
import pandas as pd

from build_features import FEATURES, TARGET
from evaluate import score
from train_model import importance, train


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