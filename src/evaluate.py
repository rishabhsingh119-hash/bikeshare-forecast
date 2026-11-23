# Shared tools for testing models fairly: splitting the data by time, and
# scoring how well a model ranks the risky stations.

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from add_targets import HORIZON


def split_by_time(df, test_days=1):
    """Train on the earlier data and test on the last `test_days` days.

    Training rows from the hour just before the test period are left out. Their
    answers ("did it go empty in the next hour?") depend on what happened during
    the test period, so keeping them would let a little of the test leak into training.
    """
    test_start = df["time"].max() - pd.Timedelta(days=test_days)
    train = df[df["time"] < test_start - HORIZON]
    test = df[df["time"] >= test_start]
    return train, test


def only_fine_now(df):
    """Keep the stations that are neither empty nor full right now.

    Predicting that an empty station is still empty is easy and not useful. The
    real question is which stations that are fine now will run into trouble.
    """
    return df[(df["is_empty"] == 0) & (df["is_full"] == 0)]


def score(answers, risk):
    """How well `risk` ranks the stations that really did become empty or full.

    PR-AUC:   our main score. Random guessing scores about the share of "yes"
              answers (around 0.1); a perfect ranking scores 1.
    ROC-AUC:  how often a real "yes" is ranked above a real "no". Random = 0.5.
    hit rate: of the 10% of rows ranked riskiest, the share that really ran into trouble.
    """
    answers = np.asarray(answers)
    risk = np.asarray(risk, dtype=float)
    top_10_percent = np.argsort(-risk, kind="stable")[: len(risk) // 10]
    return {
        "PR-AUC": average_precision_score(answers, risk),
        "ROC-AUC": roc_auc_score(answers, risk),
        "hit rate (top 10%)": answers[top_10_percent].mean(),
    }


def print_scores(results):
    """results: {model name: scores}. Prints one row per model."""
    table = pd.DataFrame(results).T
    print(table.round(3).to_string())

def calibration_table(answers, risk, groups=10):
    """Sort predictions from safest to riskiest, split them into equal-sized groups,
    and compare each group's average predicted risk with how often trouble really happened.

    For honest probabilities, the two columns should be close in every group.
    """
    df = pd.DataFrame({"predicted": np.asarray(risk, dtype=float), "actual": np.asarray(answers)})
    df["group"] = pd.qcut(df["predicted"].rank(method="first"), groups, labels=False) + 1
    return df.groupby("group")[["predicted", "actual"]].mean()