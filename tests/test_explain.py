import numpy as np
import pandas as pd

from build_features import FEATURES, TARGET
from explain import shap_values, top_reasons
from train_model import train


def test_shap_values_add_up_to_the_model_score():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({feature: rng.random(3000) for feature in FEATURES})
    df[TARGET] = (rng.random(3000) < df["fill_ratio"] / 2).astype(int)
    booster = train(df[:2000], df[2000:2500]).booster_
    rows = df[2500:2600]

    shap = shap_values(booster, rows)
    starting_value = booster.predict(rows[FEATURES], pred_contrib=True)[:, -1]
    model_score = booster.predict(rows[FEATURES], raw_score=True)

    assert list(shap.columns) == FEATURES
    assert np.allclose(shap.sum(axis=1) + starting_value, model_score)


def test_top_reasons_are_the_biggest_pushes_upwards():
    row_shap = pd.Series({"fill_ratio": 1.2, "temperature": -0.5, "time_of_day": 0.3})
    row_features = pd.Series({"fill_ratio": 0.05, "temperature": 2.0, "time_of_day": 17.5})

    reasons = top_reasons(row_shap, row_features, top=2)

    assert reasons == [("fill_ratio", 0.05, 1.2), ("time_of_day", 17.5, 0.3)]