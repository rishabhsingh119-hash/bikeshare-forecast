import numpy as np
import pandas as pd

from baselines import few_bikes_or_docks, few_bikes_or_docks_after_usual_demand
from evaluate import only_fine_now, score, split_by_time


def test_split_by_time_keeps_a_gap_before_the_test_days():
    times = pd.date_range("2026-11-18 00:00", "2026-11-21 00:00", freq="5min", tz="UTC")
    df = pd.DataFrame({"time": times})

    train, test = split_by_time(df, test_days=1)

    assert test["time"].min() == pd.Timestamp("2026-11-20 00:00", tz="UTC")
    assert train["time"].max() < pd.Timestamp("2026-11-19 23:00", tz="UTC")  # an hour's gap
    assert len(train) + len(test) < len(df)  # the gap rows are in neither


def test_only_fine_now():
    df = pd.DataFrame({"is_empty": [0, 1, 0], "is_full": [0, 0, 1]})
    assert len(only_fine_now(df)) == 1


def test_perfect_ranking_scores_one():
    answers = [0] * 18 + [1, 1]
    risk = list(range(20))  # the two "yes" rows get the highest risk

    scores = score(answers, risk)

    assert scores["PR-AUC"] == 1
    assert scores["ROC-AUC"] == 1
    assert scores["hit rate (top 10%)"] == 1


def test_random_ranking_scores_about_the_base_rate():
    rng = np.random.default_rng(0)
    answers = rng.random(20_000) < 0.1

    scores = score(answers, rng.random(20_000))

    assert abs(scores["PR-AUC"] - 0.1) < 0.02
    assert abs(scores["ROC-AUC"] - 0.5) < 0.02


def test_rules_rank_nearly_empty_and_nearly_full_stations_as_risky():
    df = pd.DataFrame({
        "bikes_available": [1, 10, 19],
        "docks_available": [19, 10, 1],
        "usual_arrivals_next_hour": [0.0, 0.0, 0.0],
        "usual_departures_next_hour": [0.0, 10.0, 0.0],
    })
    simple = few_bikes_or_docks(df)
    with_demand = few_bikes_or_docks_after_usual_demand(df)

    assert simple[0] > simple[1] and simple[2] > simple[1]  # nearly empty, nearly full
    # The middle station usually loses 10 bikes next hour, so it becomes the riskiest
    assert with_demand[1] > with_demand[0]