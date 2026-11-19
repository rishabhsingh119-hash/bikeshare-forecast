# Takes a snapshot of every bike-share station every few minutes and saves it.
#
# Usage:
#   uv run python src/collector.py          # run forever (Ctrl+C to stop)
#   uv run python src/collector.py --once   # take one snapshot and stop, good for testing
#
# Files are saved like this:
#   data/raw/station_status/date=2026-11-18/station_status_143500.parquet

import argparse
import gzip
import json
import logging
import time
from datetime import UTC, datetime

import requests

import config
from gbfs import get_feed_urls, parse_station_info, parse_station_status

log = logging.getLogger("collector")


def get_json(url, retries=3):
    """Download a JSON file. If it fails, wait a bit and try again."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            if attempt == retries:
                raise
            wait = 10 * attempt
            log.warning(f"Download failed ({error}). Trying again in {wait}s")
            time.sleep(wait)


def save_table(df, feed_name, now):
    folder = config.RAW_DATA_DIR / feed_name / f"date={now:%Y-%m-%d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{feed_name}_{now:%Y%m%dT%H%M%SZ}.parquet"

    # Write to a temporary file and then rename it, so a crash in the
    # middle of writing never leaves a broken file behind
    temp_path = path.with_suffix(".tmp")
    df.to_parquet(temp_path, index=False)
    temp_path.replace(path)


def save_raw_json(data, feed_name, now):
    folder = config.RAW_DATA_DIR / "json" / feed_name / f"date={now:%Y-%m-%d}"
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / f"{feed_name}_{now:%Y%m%dT%H%M%SZ}.json.gz", "wt") as f:
        json.dump(data, f)


def save_station_info(feed_urls, now):
    data = get_json(feed_urls["station_information"])
    df = parse_station_info(data, now)
    save_table(df, "station_information", now)
    if config.SAVE_RAW_JSON:
        save_raw_json(data, "station_information", now)
    log.info(f"Saved info for {len(df)} stations")


def save_station_status(feed_urls, now, previous_update):
    """Save a status snapshot. Returns the feed's "last_updated" value.

    If the feed hasn't changed since the last poll, nothing is saved.
    """
    data = get_json(feed_urls["station_status"])
    last_updated = data.get("last_updated")

    if previous_update is not None and last_updated == previous_update:
        log.info("Feed hasn't changed since the last poll, skipping")
        return last_updated

    df = parse_station_status(data, now)
    save_table(df, "station_status", now)
    if config.SAVE_RAW_JSON:
        save_raw_json(data, "station_status", now)
    log.info(f"Saved status for {len(df)} stations")
    return last_updated


def main():
    parser = argparse.ArgumentParser(description="Collect bike-share station snapshots.")
    parser.add_argument("--once", action="store_true", help="take one snapshot and stop")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not config.GBFS_URL:
        raise SystemExit("No feed URL. Set gbfs_url in config/settings.toml")

    feed_urls = None
    last_update = None
    info_saved_on = None  # the date we last saved station info

    log.info(f"Collecting every {config.POLL_INTERVAL}s into {config.RAW_DATA_DIR}")
    try:
        while True:
            now = datetime.now(UTC)
            try:
                if feed_urls is None:
                    feed_urls = get_feed_urls(get_json(config.GBFS_URL))

                last_update = save_station_status(feed_urls, now, last_update)

                # Station names and locations rarely change, so once a day is enough
                if info_saved_on != now.date():
                    save_station_info(feed_urls, now)
                    info_saved_on = now.date()

            except Exception as error:
                if args.once:
                    raise
                log.error(f"Snapshot failed: {error}")
                # The feed URLs might have changed, so look them up again next time
                feed_urls = None

            if args.once:
                break

            # Sleep until the next round time (10:05, 10:10, ...) so snapshots
            # from different days line up
            time.sleep(config.POLL_INTERVAL - time.time() % config.POLL_INTERVAL)

    except KeyboardInterrupt:
        log.info("Stopped")


if __name__ == "__main__":
    main()
