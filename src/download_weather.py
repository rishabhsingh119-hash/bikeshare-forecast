# Downloads hourly weather for New York from Open-Meteo.
#
# We use Open-Meteo's "historical forecast" data: what the weather forecast said
# for each hour, saved over time. When the model makes live predictions later, it
# will only have a forecast to go on, so training on forecasts keeps the two the same.
#
# Usage:
#   uv run python src/download_weather.py                                  # 2025-10-01 to yesterday
#   uv run python src/download_weather.py --start 2026-01-01 --end 2026-01-31
#
# Output:
#   data/processed/weather_hourly.parquet

import argparse
from datetime import date, timedelta

import pandas as pd
import requests

import config

HISTORY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OUTPUT_PATH = config.PROCESSED_DATA_DIR / "weather_hourly.parquet"

# The weather we want for each hour. Units: °C, mm, cm, km/h
VARIABLES = [
    "temperature_2m",  # air temperature
    "apparent_temperature",  # "feels like" temperature, with wind and humidity
    "precipitation",  # rain and snow, in mm of water
    "snowfall",  # cm of snow
    "wind_speed_10m",
    "weather_code",  # 0 = clear, 61 = rain, 71 = snow, 95 = thunderstorm, ...
]


def download_weather(start, end):
    """Download hourly weather between two dates (both included)."""
    params = {
        "latitude": config.WEATHER_LATITUDE,
        "longitude": config.WEATHER_LONGITUDE,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": "GMT",  # GMT is the same as UTC, which all our data uses
    }
    response = requests.get(HISTORY_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def parse_weather(weather_json):
    """Turn Open-Meteo's response into a table with one row per hour."""
    df = pd.DataFrame(weather_json["hourly"])
    df = df.rename(columns={"time": "hour"})
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    return df


def main():
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    parser = argparse.ArgumentParser(description="Download hourly weather for the city.")
    parser.add_argument("--start", default="2025-10-01", help="first day, like 2025-10-01")
    parser.add_argument("--end", default=yesterday, help="last day (default: yesterday)")
    args = parser.parse_args()

    print(f"Downloading weather from {args.start} to {args.end}")
    weather = parse_weather(download_weather(args.start, args.end))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved {len(weather):,} hours to {OUTPUT_PATH}")
    print(f"Missing values:   {weather.isna().sum().sum()}")
    print(f"Rainy/snowy hours: {(weather['precipitation'] > 0).mean():.0%}")
    print(
        f"Temperature:      {weather['temperature_2m'].min():.0f}°C to "
        f"{weather['temperature_2m'].max():.0f}°C"
    )


if __name__ == "__main__":
    main()
