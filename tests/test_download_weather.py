import pandas as pd

import download_weather
from download_weather import VARIABLES, parse_weather

# A tiny version of Open-Meteo's response, with two hours of weather
WEATHER_JSON = {
    "hourly": {
        "time": ["2026-07-01T12:00", "2026-07-01T13:00"],
        "temperature_2m": [27.5, 28.1],
        "apparent_temperature": [30.2, 31.0],
        "precipitation": [0.0, 2.4],
        "snowfall": [0.0, 0.0],
        "wind_speed_10m": [12.0, 18.5],
        "weather_code": [1, 61],
    }
}


def test_parse_weather():
    df = parse_weather(WEATHER_JSON)

    assert len(df) == 2
    assert df["hour"][0] == pd.Timestamp("2026-07-01 12:00", tz="UTC")
    assert df["precipitation"][1] == 2.4
    assert df["weather_code"][1] == 61


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return WEATHER_JSON


def test_download_asks_for_the_right_data(monkeypatch):
    sent = {}

    def fake_get(url, params, timeout):
        sent.update(params)
        return FakeResponse()

    monkeypatch.setattr(download_weather.requests, "get", fake_get)
    download_weather.download_weather("2026-07-01", "2026-07-01")

    assert sent["timezone"] == "GMT"  # times must come back in UTC
    assert sent["hourly"] == ",".join(VARIABLES)
    assert sent["start_date"] == "2026-07-01"
