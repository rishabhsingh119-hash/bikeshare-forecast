import pandas as pd

from days import is_day_off


def test_weekends_and_holidays_are_days_off():
    times = pd.Series(pd.to_datetime([
        "2026-11-21 10:00",  # Saturday
        "2026-11-23 10:00",  # Monday
        "2026-11-26 10:00",  # Thanksgiving, a Thursday
        "2026-11-27 10:00",  # the Friday after, a normal working day
    ])).dt.tz_localize("America/New_York")

    assert list(is_day_off(times)) == [True, False, True, False]