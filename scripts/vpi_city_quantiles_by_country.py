import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))


GRID_CITY_PATTERN = re.compile(r"^(.+?)\s+grid\s+\d+$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="VpiCityQuantilesByCountry",
        description=(
            "Select the top and bottom quantiles of city VPI scores within "
            "each country."
        ),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("data/scores_vpi.csv"),
        help="Input VPI score CSV from score_regions.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/vpi_city_quantiles_by_country.csv"),
        help="CSV output path for selected top and bottom quantile cities.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/vpi_city_quantiles_by_country_summary.csv"),
        help="CSV output path for country-level tested city counts.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.25,
        help="Fraction of cities to keep from each end of the VPI distribution.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=1,
        help="Minimum total images required for a city to be included.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["weighted", "mean"],
        default="weighted",
        help=(
            "How to combine multiple regions for the same city/country. "
            "Weighted uses image_count as the weight."
        ),
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Only include cities with aggregated VPI score greater than zero.",
    )
    parser.add_argument(
        "--include-grid-regions",
        action="store_true",
        help="Include country-grid regions such as '<country> grid 0001'.",
    )
    return parser.parse_args()


def normalize(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def is_header(row):
    lowered = [normalize(value) for value in row]
    return "region_id" in lowered and ("score" in lowered or "vpi_score" in lowered)


def row_to_record(row, header=None):
    if header is not None:
        values = dict(zip(header, row))
        return {
            "region_id": values.get("region_id", ""),
            "city": values.get("city", ""),
            "country": values.get("country", ""),
            "image_count": values.get("image_count", values.get("images", "0")),
            "score": values.get("vpi_score", values.get("score", "")),
        }

    return {
        "region_id": row[0] if len(row) > 0 else "",
        "city": row[1] if len(row) > 1 else "",
        "country": row[2] if len(row) > 2 else "",
        "image_count": row[3] if len(row) > 3 else "0",
        "score": row[4] if len(row) > 4 else "",
    }


def parse_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_country_grid_region(city, country):
    match = GRID_CITY_PATTERN.match((city or "").strip())
    return bool(match and normalize(match.group(1)) == normalize(country))


def load_score_rows(path, include_grid_regions):
    if not path.exists():
        raise SystemExit(f"Input score CSV does not exist: {path}")

    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            first_row = next(reader)
        except StopIteration:
            return rows

        header = first_row if is_header(first_row) else None
        if header is None:
            candidate_rows = [first_row]
        else:
            header = [normalize(column).replace(" ", "_") for column in header]
            candidate_rows = []

        candidate_rows.extend(reader)
        for raw_row in candidate_rows:
            record = row_to_record(raw_row, header)
            city = (record["city"] or "").strip()
            country = (record["country"] or "").strip()
            score = parse_float(record["score"])
            if not city or not country or score is None:
                continue
            if not include_grid_regions and is_country_grid_region(city, country):
                continue
            rows.append(
                {
                    "region_id": record["region_id"],
                    "city": city,
                    "country": country,
                    "image_count": parse_int(record["image_count"]),
                    "score": score,
                }
            )

    return rows


def aggregate_city_scores(rows, aggregation):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["country"], row["city"])].append(row)

    cities = []
    for (country, city), city_rows in grouped.items():
        total_images = sum(row["image_count"] for row in city_rows)
        if aggregation == "weighted" and total_images > 0:
            score = sum(
                row["score"] * row["image_count"] for row in city_rows
            ) / total_images
        else:
            score = sum(row["score"] for row in city_rows) / len(city_rows)

        cities.append(
            {
                "country": country,
                "city": city,
                "vpi_score": score,
                "total_images": total_images,
                "region_count": len(city_rows),
            }
        )

    return cities


def select_country_quantile_rows(cities, quantile):
    if not 0 < quantile <= 0.5:
        raise SystemExit("--quantile must be greater than 0 and no more than 0.5.")
    if not cities:
        return []

    cities_by_country = defaultdict(list)
    for row in cities:
        cities_by_country[row["country"]].append(row)

    selected = []
    for country_rows in cities_by_country.values():
        selected_count = max(1, math.ceil(len(country_rows) * quantile))
        ordered = sorted(
            country_rows,
            key=lambda row: (
                row["vpi_score"],
                normalize(row["city"]),
            ),
        )

        selected.extend(
            dict(row, quantile_group="bottom") for row in ordered[:selected_count]
        )
        selected.extend(
            dict(row, quantile_group="top")
            for row in reversed(ordered[-selected_count:])
        )
    return selected


def country_counts(cities):
    counts = defaultdict(int)
    for row in cities:
        counts[row["country"]] += 1
    return counts


def write_outputs(selected_rows, tested_city_counts, output_path, summary_output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_rows = sorted(
        selected_rows,
        key=lambda row: (
            normalize(row["country"]),
            row["quantile_group"] != "top",
            -row["vpi_score"] if row["quantile_group"] == "top" else row["vpi_score"],
            normalize(row["city"]),
        ),
    )

    selected_count_by_country = defaultdict(lambda: {"top": 0, "bottom": 0})
    for row in selected_rows:
        selected_count_by_country[row["country"]][row["quantile_group"]] += 1

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "quantile_group",
                "country",
                "tested_city_count",
                "city",
                "vpi_score",
                "total_images",
                "region_count",
            ]
        )
        for row in selected_rows:
            writer.writerow(
                [
                    row["quantile_group"],
                    row["country"],
                    tested_city_counts[row["country"]],
                    row["city"],
                    f"{row['vpi_score']:.6f}",
                    row["total_images"],
                    row["region_count"],
                ]
            )

    with summary_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "country",
                "tested_city_count",
                "top_quantile_city_count",
                "bottom_quantile_city_count",
            ]
        )
        for country in sorted(tested_city_counts, key=normalize):
            selected = selected_count_by_country[country]
            writer.writerow(
                [
                    country,
                    tested_city_counts[country],
                    selected["top"],
                    selected["bottom"],
                ]
            )


def print_country_sections(selected_rows, tested_city_counts):
    by_country = defaultdict(list)
    for row in selected_rows:
        by_country[row["country"]].append(row)

    for country in sorted(by_country, key=normalize):
        print(f"\n{country} ({tested_city_counts[country]} cities tested)")
        rows = sorted(
            by_country[country],
            key=lambda row: (
                row["quantile_group"] != "top",
                -row["vpi_score"] if row["quantile_group"] == "top" else row["vpi_score"],
                normalize(row["city"]),
            ),
        )
        for row in rows:
            print(
                f"{row['quantile_group']}: {row['city']} | "
                f"VPI={row['vpi_score']:.6f} | "
                f"images={row['total_images']} | regions={row['region_count']}"
            )


def main():
    args = parse_args()
    if args.min_images < 0:
        raise SystemExit("--min-images must be zero or greater.")

    rows = load_score_rows(args.scores, args.include_grid_regions)
    cities = aggregate_city_scores(rows, args.aggregation)
    cities = [row for row in cities if row["total_images"] >= args.min_images]
    if args.positive_only:
        cities = [row for row in cities if row["vpi_score"] > 0]

    if not cities:
        print("No city VPI scores found.")
        return 0

    selected_rows = select_country_quantile_rows(cities, args.quantile)
    tested_city_counts = country_counts(cities)

    write_outputs(
        selected_rows,
        tested_city_counts,
        args.output,
        args.summary_output,
    )
    print_country_sections(selected_rows, tested_city_counts)
    print(f"\nCities tested: {len(cities)}")
    print(f"Countries tested: {len(tested_city_counts)}")
    print(f"Saved selected city quantiles to {args.output}")
    print(f"Saved country summary to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
