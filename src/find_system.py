# Finds a bike-share system's feed URL in the official GBFS catalog.
#
# Usage:
#   uv run python src/find_system.py "citi bike"
#   uv run python src/find_system.py chicago

import csv
import io
import sys

import requests

CATALOG_URL = "https://raw.githubusercontent.com/MobilityData/gbfs/master/systems.csv"


def search(catalog_text, query):
    """Return every system whose name, location or ID contains the query."""
    query = query.lower()
    matches = []
    for row in csv.DictReader(io.StringIO(catalog_text)):
        text = f"{row.get('Name')} {row.get('Location')} {row.get('System ID')}".lower()
        if query in text:
            matches.append(row)
    return matches


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run python src/find_system.py "city or system name"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    response = requests.get(CATALOG_URL, timeout=30)
    response.raise_for_status()

    matches = search(response.text, query)
    if not matches:
        print(f"Nothing found for '{query}'. Try the city name instead.")
        return

    for row in matches:
        print(f"{row.get('Name')} ({row.get('Location')})")
        print(f"  Feed URL: {row.get('Auto-Discovery URL')}")
        print()


if __name__ == "__main__":
    main()
