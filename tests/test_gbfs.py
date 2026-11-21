from datetime import UTC, datetime

from gbfs import get_feed_urls, parse_station_info, parse_station_status

# A fixed time to use as the snapshot time in every test
NOW = datetime(2026, 11, 16, 14, 35, tzinfo=UTC)


def test_feed_urls_version_2():
    # Versions 1 and 2 group feeds by language
    gbfs_json = {
        "data": {
            "en": {
                "feeds": [
                    {"name": "station_status", "url": "https://example.com/status"},
                ]
            }
        }
    }
    assert get_feed_urls(gbfs_json) == {"station_status": "https://example.com/status"}


def test_feed_urls_version_3():
    # Version 3 lists feeds directly
    gbfs_json = {
        "data": {
            "feeds": [
                {"name": "station_status", "url": "https://example.com/status"},
            ]
        }
    }
    assert get_feed_urls(gbfs_json) == {"station_status": "https://example.com/status"}


def test_feed_urls_falls_back_to_another_language():
    gbfs_json = {
        "data": {
            "fr": {
                "feeds": [
                    {"name": "station_status", "url": "https://example.com/status"},
                ]
            }
        }
    }
    assert get_feed_urls(gbfs_json) == {"station_status": "https://example.com/status"}


def test_station_status_version_2():
    status_json = {
        "data": {
            "stations": [
                {
                    "station_id": 72,
                    "num_bikes_available": 3,
                    "num_docks_available": 20,
                    "is_installed": 1,
                    "is_renting": 1,
                    "is_returning": 1,
                    "last_reported": 1763303700,
                },
            ]
        }
    }
    df = parse_station_status(status_json, NOW)

    assert len(df) == 1
    assert df["station_id"][0] == "72"  # IDs are always stored as text
    assert df["bikes_available"][0] == 3
    assert df["is_renting"][0]  # 1 became True
    assert df["last_reported"][0].year == 2025  # Unix seconds became a date


def test_station_status_version_3():
    status_json = {
        "data": {
            "stations": [
                {
                    "station_id": "a",
                    "num_vehicles_available": 5,
                    "num_docks_available": 7,
                    "is_installed": True,
                    "is_renting": False,
                    "is_returning": True,
                    "last_reported": "2026-11-16T14:30:00Z",
                },
            ]
        }
    }
    df = parse_station_status(status_json, NOW)

    assert df["bikes_available"][0] == 5  # read from "num_vehicles_available"
    assert not df["is_renting"][0]
    assert df["last_reported"][0].hour == 14  # text became a date


def test_missing_values_stay_empty():
    status_json = {
        "data": {
            "stations": [
                {"station_id": "a", "num_bikes_available": 2, "num_docks_available": 5},
            ]
        }
    }
    df = parse_station_status(status_json, NOW)

    assert df["ebikes_available"].isna()[0]
    assert df["bikes_available"][0] == 2  # still a whole number, not 2.0


def test_empty_feed_gives_empty_table():
    df = parse_station_status({"data": {"stations": []}}, NOW)
    assert df.empty


def test_station_info_picks_english_name():
    info_json = {
        "data": {
            "stations": [
                {
                    "station_id": "a",
                    "short_name": "6140.10",
                    "lat": 40.7,
                    "lon": -74.0,
                    "capacity": 31,
                    "name": [
                        {"text": "Rue X", "language": "fr"},
                        {"text": "W 52 St", "language": "en"},
                    ],
                },
            ]
        }
    }
    df = parse_station_info(info_json, NOW)

    assert df["name"][0] == "W 52 St"
    assert df["capacity"][0] == 31
    assert df["short_name"][0] == "6140.10"
