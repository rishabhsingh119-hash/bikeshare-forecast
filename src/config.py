# Loads the project settings from config/settings.toml.
# Values in the .env file override the ones in the settings file.

import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

# The project folder (one level up from src/), so paths work
# no matter which folder you run a script from
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read .env and load its values as environment variables
load_dotenv(PROJECT_ROOT / ".env")

with open(PROJECT_ROOT / "config" / "settings.toml", "rb") as f:
    settings = tomllib.load(f)

# Use the .env value if there is one, otherwise the settings file
GBFS_URL = os.getenv("GBFS_URL") or settings["feed"]["gbfs_url"]
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL") or settings["collector"]["poll_interval"])
SAVE_RAW_JSON = settings["collector"]["save_raw_json"]
TIMEZONE = settings["project"]["timezone"]
WEATHER_LATITUDE = settings["weather"]["latitude"]
WEATHER_LONGITUDE = settings["weather"]["longitude"]

RAW_DATA_DIR = PROJECT_ROOT / settings["paths"]["raw_data"]
PROCESSED_DATA_DIR = PROJECT_ROOT / settings["paths"]["processed_data"]


if __name__ == "__main__":
    # Run "uv run python src/config.py" to check your settings
    print("Feed URL:     ", GBFS_URL or "(not set yet)")
    print("Poll interval:", POLL_INTERVAL, "seconds")
    print("Raw data dir: ", RAW_DATA_DIR)
    print("Time zone:    ", TIMEZONE)
    print("Weather at:   ", WEATHER_LATITUDE, WEATHER_LONGITUDE)
