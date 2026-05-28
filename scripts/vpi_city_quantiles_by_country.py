import argparse
import csv
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
            "Find countries represented in the global top/bottom VPI quantiles "
            "and report how many city appearances each country has in those groups."
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
        "--selection-mode",
        choices=["tail_percentile", "quantile_bucket"],
        default="quantile_bucket",
        help=(
            "tail_percentile selects global top/bottom tails like 5%% and 95%%. "
            "quantile_bucket selects broader global quantile buckets like top/bottom decile or quartile."
        ),
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.1,
        help="Tail fraction for --selection-mode tail_percentile, e.g. 0.05 for 5%% and 95%%.",
    )
    parser.add_argument(
        "--quantile-fraction",
        type=float,
        default=0.1,
        help="Bucket fraction for --selection-mode quantile_bucket, e.g. 0.10 for top/bottom decile or 0.25 for quartile.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=300,
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
    parser.add_argument(
        "--min-country-cities",
        type=int,
        default=5,
        help="Minimum tested cities required for a country to appear in the printed rankings.",
    )
    parser.add_argument(
        "--rank-count",
        type=int,
        default=10,
        help="How many countries to show in the printed top/bottom percentile rankings.",
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


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, float(fraction))) * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(len(sorted_values) - 1, lower_index + 1)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    return lower_value + (upper_value - lower_value) * (position - lower_index)


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


def select_global_percentile_rows(cities, percentile_fraction):
    if not 0 < percentile_fraction < 0.5:
        raise SystemExit(
            "--percentile must be greater than 0 and less than 0.5.")
    if not cities:
        return []

    ordered_scores = sorted(row["vpi_score"] for row in cities)
    bottom_cutoff = percentile(ordered_scores, percentile_fraction)
    top_cutoff = percentile(ordered_scores, 1.0 - percentile_fraction)

    selected = []
    for row in cities:
        if row["vpi_score"] <= bottom_cutoff:
            selected.append(dict(row, quantile_group="bottom"))
        elif row["vpi_score"] >= top_cutoff:
            selected.append(dict(row, quantile_group="top"))
    return selected


def select_global_quantile_bucket_rows(cities, quantile_fraction):
    if not 0 < quantile_fraction <= 0.5:
        raise SystemExit(
            "--quantile-fraction must be greater than 0 and no more than 0.5.")
    if not cities:
        return []

    ordered_scores = sorted(row["vpi_score"] for row in cities)
    bottom_cutoff = percentile(ordered_scores, quantile_fraction)
    top_cutoff = percentile(ordered_scores, 1.0 - quantile_fraction)

    selected = []
    for row in cities:
        if row["vpi_score"] <= bottom_cutoff:
            selected.append(dict(row, quantile_group="bottom"))
        elif row["vpi_score"] >= top_cutoff:
            selected.append(dict(row, quantile_group="top"))
    return selected


def country_counts(cities):
    counts = defaultdict(int)
    for row in cities:
        counts[row["country"]] += 1
    return counts


def write_outputs(selected_rows, tested_city_counts, output_path, summary_output_path, mode_label):
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
                "selection_mode",
                "country",
                "tested_city_count",
                "country_tail_share",
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
                    mode_label,
                    row["country"],
                    tested_city_counts[row["country"]],
                    f"{selected_count_by_country[row['country']][row['quantile_group']] / tested_city_counts[row['country']]:.6f}",
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
                "selection_mode",
                "top_percentile_appearances",
                "top_percentile_city_count",
                "top_percentile_share",
                "bottom_percentile_appearances",
                "bottom_percentile_city_count",
                "bottom_percentile_share",
            ]
        )
        for country in sorted(tested_city_counts, key=normalize):
            selected = selected_count_by_country[country]
            writer.writerow(
                [
                    country,
                    tested_city_counts[country],
                    mode_label,
                    selected["top"],
                    selected["top"],
                    f"{selected['top'] / tested_city_counts[country]:.6f}",
                    selected["bottom"],
                    selected["bottom"],
                    f"{selected['bottom'] / tested_city_counts[country]:.6f}",
                ]
            )


def build_country_summary_rows(selected_rows, tested_city_counts):
    selected_count_by_country = defaultdict(lambda: {"top": 0, "bottom": 0})
    for row in selected_rows:
        selected_count_by_country[row["country"]][row["quantile_group"]] += 1

    summary_rows = []
    for country, tested_city_count in tested_city_counts.items():
        selected = selected_count_by_country[country]
        summary_rows.append(
            {
                "country": country,
                "tested_city_count": tested_city_count,
                "top_percentile_appearances": selected["top"],
                "top_percentile_city_count": selected["top"],
                "top_percentile_share": (
                    selected["top"] /
                    tested_city_count if tested_city_count else 0.0
                ),
                "bottom_percentile_appearances": selected["bottom"],
                "bottom_percentile_city_count": selected["bottom"],
                "bottom_percentile_share": (
                    selected["bottom"] /
                    tested_city_count if tested_city_count else 0.0
                ),
            }
        )
    return summary_rows


def selection_label(selection_mode, percentile_fraction, quantile_fraction):
    if selection_mode == "quantile_bucket":
        return f"top/bottom {int(quantile_fraction * 100)}% quantile bucket"
    return f"top/bottom {int(percentile_fraction * 100)}% tail"


def print_rankings(
    summary_rows,
    min_images,
    selection_mode,
    percentile_fraction,
    quantile_fraction,
):
    top_rows = [
        row for row in summary_rows if row["top_percentile_appearances"] > 0
    ]
    bottom_rows = [
        row for row in summary_rows if row["bottom_percentile_appearances"] > 0
    ]
    if not top_rows and not bottom_rows:
        print("No countries appeared in the selected top/bottom groups.")
        return

    top_rows = sorted(
        top_rows,
        key=lambda row: (
            -row["top_percentile_appearances"],
            -row["tested_city_count"],
            normalize(row["country"]),
        ),
    )
    bottom_rows = sorted(
        bottom_rows,
        key=lambda row: (
            -row["bottom_percentile_appearances"],
            -row["tested_city_count"],
            normalize(row["country"]),
        ),
    )

    if selection_mode == "quantile_bucket":
        fraction_text = int(quantile_fraction * 100)
        top_text = f"global top {fraction_text}%"
        bottom_text = f"global bottom {fraction_text}%"
    else:
        fraction_text = int(percentile_fraction * 100)
        top_text = f"global top {fraction_text}% of VPI"
        bottom_text = f"global bottom {fraction_text}% of VPI"

    print(
        f"Using only cities with at least {min_images} images:"
    )
    print()
    print(f"Countries represented in the {top_text}, with appearance counts:")
    for index, row in enumerate(top_rows, start=1):
        print(
            f"{index}. {row['country']}: "
            f"{row['top_percentile_appearances']}/{row['tested_city_count']} "
            f"({row['top_percentile_share'] * 100:.1f}%)"
        )

    print()
    print(
        f"Countries represented in the {bottom_text}, with appearance counts:")
    for index, row in enumerate(bottom_rows, start=1):
        print(
            f"{index}. {row['country']}: "
            f"{row['bottom_percentile_appearances']}/{row['tested_city_count']} "
            f"({row['bottom_percentile_share'] * 100:.1f}%)"
        )


def main():
    args = parse_args()
    if args.min_images < 0:
        raise SystemExit("--min-images must be zero or greater.")
    if args.min_country_cities < 1:
        raise SystemExit("--min-country-cities must be at least 1.")
    if args.rank_count < 1:
        raise SystemExit("--rank-count must be at least 1.")
    if not 0 < args.quantile_fraction <= 0.5:
        raise SystemExit(
            "--quantile-fraction must be greater than 0 and no more than 0.5.")

    rows = load_score_rows(args.scores, args.include_grid_regions)
    cities = aggregate_city_scores(rows, args.aggregation)
    cities = [row for row in cities if row["total_images"] >= args.min_images]
    if args.positive_only:
        cities = [row for row in cities if row["vpi_score"] > 0]

    if not cities:
        print("No city VPI scores found.")
        return 0

    if args.selection_mode == "quantile_bucket":
        selected_rows = select_global_quantile_bucket_rows(
            cities, args.quantile_fraction)
    else:
        selected_rows = select_global_percentile_rows(cities, args.percentile)
    tested_city_counts = country_counts(cities)
    summary_rows = build_country_summary_rows(
        selected_rows, tested_city_counts)
    mode_label = selection_label(
        args.selection_mode, args.percentile, args.quantile_fraction)

    write_outputs(
        selected_rows,
        tested_city_counts,
        args.output,
        args.summary_output,
        mode_label,
    )
    print_rankings(
        summary_rows,
        args.min_images,
        args.selection_mode,
        args.percentile,
        args.quantile_fraction,
    )
    print(f"\nCities tested: {len(cities)}")
    print(f"Countries tested: {len(tested_city_counts)}")
    print(f"Saved selected city quantiles to {args.output}")
    print(f"Saved country summary to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
