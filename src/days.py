# Working days and days off. People ride very differently on weekends and
# public holidays, so both count as "days off".

import holidays
import pandas as pd

import config

# US public holidays for the years our data covers, and a few years ahead
HOLIDAYS = holidays.country_holidays("US", years=range(2025, 2031))
HOLIDAY_DATES = pd.to_datetime(list(HOLIDAYS)).tz_localize(config.TIMEZONE)


def is_day_off(local_times):
    """True for Saturdays, Sundays and US public holidays. Expects New York local times."""
    weekend = local_times.dt.day_of_week >= 5  # Monday is 0, Saturday is 5
    holiday = local_times.dt.normalize().isin(HOLIDAY_DATES)
    return weekend | holiday