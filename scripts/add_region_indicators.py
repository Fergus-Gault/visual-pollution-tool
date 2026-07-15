import argparse
import csv
import sys
from pathlib import Path

from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.database import DatabaseManager, Region


INDICATORS = {
    "gdp": {
        "pattern": "API_NY.GDP.MKTP.CD_*.csv",
        "description": "GDP (current US$)",
    },
    "gdppp": {
        "pattern": "API_NY.GDP.PCAP.CD_*.csv",
        "description": "GDP per capita (current US$)",
    },
    "gni": {
        "pattern": "API_NY.GNP.PCAP.CD_*.csv",
        "description": "GNI per capita, Atlas method (current US$)",
    },
    "urb": {
        "pattern": "API_SP.URB.TOTL.IN.ZS_*.csv",
        "description": "Urban population (% of total population)",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="AddRegionIndicators",
        description=(
            "Add the most recent GDP, GDP per capita, GNI, and urban population "
            "indicator values to each region using region.iso3."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the World Bank indicator CSV files.",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Optional country filter for regions to update.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Optional city filter for regions to update.",
    )
    parser.add_argument(
        "--region-id",
        default=None,
        help="Optional single region id to update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without writing to the database.",
    )
    return parser.parse_args()


def normalise(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def find_indicator_file(data_dir, pattern):
    matches = sorted(data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No data file matched {data_dir / pattern}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple data files matched {data_dir / pattern}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def read_world_bank_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    header_index = next(
        index for index, row in enumerate(rows) if row and row[0] == "Country Name"
    )
    header = rows[header_index]
    data_rows = rows[header_index + 1:]

    year_columns = [
        (index, int(column))
        for index, column in enumerate(header)
        if column.isdigit()
    ]
    year_columns.sort(key=lambda item: item[1], reverse=True)

    values_by_iso3 = {}
    for row in data_rows:
        if len(row) < 2:
            continue
        iso3 = row[1].strip().upper()
        if not iso3:
            continue
        for column_index, year in year_columns:
            if column_index >= len(row):
                continue
            value = row[column_index].strip()
            if not value:
                continue
            try:
                values_by_iso3[iso3] = (float(value), year)
                break
            except ValueError:
                continue

    return values_by_iso3


def load_indicator_data(data_dir):
    indicator_data = {}
    for indicator, config in INDICATORS.items():
        path = find_indicator_file(data_dir, config["pattern"])
        indicator_data[indicator] = read_world_bank_csv(path)
        print(
            f"Loaded {indicator}: {config['description']} from {path} "
            f"({len(indicator_data[indicator])} country codes)"
        )
    return indicator_data


def build_region_query(db, args):
    query = db.session.query(Region)
    if args.region_id:
        query = query.filter(Region.id == args.region_id)
    if args.country:
        query = query.filter(
            func.lower(func.coalesce(Region.country, "")) == normalise(args.country)
        )
    if args.city:
        query = query.filter(
            func.lower(func.coalesce(Region.city, "")) == normalise(args.city)
        )
    return query.order_by(Region.country.asc(), Region.city.asc(), Region.name.asc())


def values_for_region(region, indicator_data):
    iso3 = (region.iso3 or "").strip().upper()
    values = {}
    for indicator, data in indicator_data.items():
        value, year = data.get(iso3, (None, None))
        values[indicator] = value
        values[f"{indicator}_year"] = year
    return values


def update_regions(db, regions, indicator_data, dry_run):
    matched = 0
    missing_iso3 = 0
    complete = 0

    for region in regions:
        iso3 = (region.iso3 or "").strip().upper()
        if not iso3:
            missing_iso3 += 1

        values = values_for_region(region, indicator_data)
        if any(values[indicator] is not None for indicator in INDICATORS):
            matched += 1
        if all(values[indicator] is not None for indicator in INDICATORS):
            complete += 1

        if not dry_run:
            for column, value in values.items():
                setattr(region, column, value)

    if not dry_run:
        db.session.commit()

    return {
        "regions": len(regions),
        "matched_regions": matched,
        "complete_regions": complete,
        "missing_iso3_regions": missing_iso3,
    }


def print_preview(regions, indicator_data, dry_run):
    action = "Would update" if dry_run else "Updated"
    print(f"\n{action} {len(regions)} regions")
    for region in regions[:20]:
        values = values_for_region(region, indicator_data)
        parts = []
        for indicator in INDICATORS:
            value = values[indicator]
            year = values[f"{indicator}_year"]
            text = "n/a" if value is None else f"{value:.6g} ({year})"
            parts.append(f"{indicator}={text}")
        print(f"{region.id} | {region.iso3 or '<no iso3>'} | " + " | ".join(parts))
    if len(regions) > 20:
        print(f"... {len(regions) - 20} more regions omitted")


def main():
    args = parse_args()
    indicator_data = load_indicator_data(Path(args.data_dir))

    db = DatabaseManager()
    regions = build_region_query(db, args).all()
    stats = update_regions(db, regions, indicator_data, args.dry_run)

    print_preview(regions, indicator_data, args.dry_run)
    print(
        "\nSummary: "
        f"regions={stats['regions']}, "
        f"matched={stats['matched_regions']}, "
        f"complete={stats['complete_regions']}, "
        f"missing_iso3={stats['missing_iso3_regions']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
