import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager, Image, OSMFeature, Region  # noqa: E402


DEFAULT_INCOME_ORDER = [
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "High income",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot imagery and OSM data availability by economic classification."
        )
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=Path("data/CLASS_2025_10_07.csv"),
        help="CSV containing Code and Income group columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maps/data_availability_by_income_group.png"),
        help="Output plot image path.",
    )
    parser.add_argument(
        "--per-region-csv",
        type=Path,
        default=Path("data/data_availability_by_income_group_regions.csv"),
        help="Output per-region availability CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("data/data_availability_by_income_group_summary.csv"),
        help="Output income-group summary CSV.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=1,
        help="Minimum image count for a region to be treated as having imagery.",
    )
    parser.add_argument(
        "--min-osm-features",
        type=int,
        default=1,
        help="Minimum OSM feature count for a region to be treated as having OSM data.",
    )
    parser.add_argument(
        "--osm-available-when-fetched",
        action="store_true",
        help=(
            "Treat regions with osm_fetched=true as having OSM availability, "
            "even when they contain fewer than --min-osm-features features."
        ),
    )
    return parser.parse_args()


def load_classification_table(path):
    if not path.exists():
        raise SystemExit(f"Classification CSV not found: {path}")

    classifications = pd.read_csv(path)
    required = {"Code", "Income group"}
    missing = required - set(classifications.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise SystemExit(
            f"Classification CSV missing required columns: {missing_text}"
        )

    available_columns = [
        column
        for column in ["Code", "Economy", "Region", "Income group"]
        if column in classifications.columns
    ]
    classifications = classifications[available_columns].copy()
    classifications["Code"] = (
        classifications["Code"].astype(str).str.strip().str.upper()
    )
    classifications["Income group"] = (
        classifications["Income group"].astype(str).str.strip()
    )
    return classifications


def normalize_iso3_series(series):
    return series.fillna("").astype(str).str.strip().str.upper()


def order_income_groups(values):
    ordered = [group for group in DEFAULT_INCOME_ORDER if group in values]
    extras = sorted(value for value in values if value not in DEFAULT_INCOME_ORDER)
    return ordered + extras


def load_region_availability(db):
    image_counts = (
        db.session.query(
            Image.region_id.label("region_id"),
            func.count(Image.id).label("image_count"),
        )
        .group_by(Image.region_id)
        .subquery()
    )
    osm_counts = (
        db.session.query(
            OSMFeature.region_id.label("region_id"),
            func.count(OSMFeature.id).label("osm_feature_count"),
        )
        .group_by(OSMFeature.region_id)
        .subquery()
    )

    rows = (
        db.session.query(
            Region.id.label("region_id"),
            Region.name.label("region_name"),
            Region.city.label("city"),
            Region.country.label("country"),
            Region.iso3.label("iso3"),
            Region.population.label("population"),
            Region.dense_scan.label("dense_scan"),
            Region.osm_fetched.label("osm_fetched"),
            func.coalesce(image_counts.c.image_count, 0).label("image_count"),
            func.coalesce(osm_counts.c.osm_feature_count, 0).label(
                "osm_feature_count"
            ),
        )
        .outerjoin(image_counts, image_counts.c.region_id == Region.id)
        .outerjoin(osm_counts, osm_counts.c.region_id == Region.id)
        .all()
    )

    availability = pd.DataFrame(
        rows,
        columns=[
            "region_id",
            "region_name",
            "city",
            "country",
            "iso3",
            "population",
            "dense_scan",
            "osm_fetched",
            "image_count",
            "osm_feature_count",
        ],
    )
    if availability.empty:
        return availability

    availability["iso3"] = normalize_iso3_series(availability["iso3"])
    availability["image_count"] = pd.to_numeric(
        availability["image_count"], errors="coerce"
    ).fillna(0)
    availability["osm_feature_count"] = pd.to_numeric(
        availability["osm_feature_count"], errors="coerce"
    ).fillna(0)
    availability["osm_fetched"] = availability["osm_fetched"].fillna(False).astype(bool)
    return availability


def merge_with_classifications(availability, classifications):
    if availability.empty:
        return availability

    merged = availability.merge(
        classifications,
        left_on="iso3",
        right_on="Code",
        how="left",
    )
    merged = merged[
        merged["Income group"].notna() & (merged["Income group"] != "")
    ].copy()
    return merged


def add_availability_flags(
    availability,
    min_images,
    min_osm_features,
    osm_available_when_fetched,
):
    availability = availability.copy()
    availability["has_imagery"] = availability["image_count"] >= min_images
    if osm_available_when_fetched:
        availability["has_osm"] = (
            availability["osm_feature_count"] >= min_osm_features
        ) | availability["osm_fetched"]
    else:
        availability["has_osm"] = availability["osm_feature_count"] >= min_osm_features
    availability["has_both"] = availability["has_imagery"] & availability["has_osm"]
    return availability


def summarize_by_income_group(availability):
    if availability.empty:
        return pd.DataFrame()

    grouped = availability.groupby("Income group", observed=False)
    summary = grouped.agg(
        region_count=("region_id", "nunique"),
        regions_with_imagery=("has_imagery", "sum"),
        regions_with_osm=("has_osm", "sum"),
        regions_with_both=("has_both", "sum"),
        total_images=("image_count", "sum"),
        total_osm_features=("osm_feature_count", "sum"),
        median_images=("image_count", "median"),
        median_osm_features=("osm_feature_count", "median"),
        mean_images=("image_count", "mean"),
        mean_osm_features=("osm_feature_count", "mean"),
    ).reset_index()

    for column in ["imagery", "osm", "both"]:
        summary[f"{column}_availability_pct"] = (
            summary[f"regions_with_{column}"] / summary["region_count"] * 100
        )

    income_order = order_income_groups(summary["Income group"].unique())
    summary["income_order"] = summary["Income group"].map(
        {group: index for index, group in enumerate(income_order)}
    )
    return summary.sort_values("income_order").drop(columns=["income_order"])


def plot_availability(availability, summary, output_path):
    if availability.empty or summary.empty:
        raise SystemExit("No matched availability data available to plot.")

    income_order = order_income_groups(summary["Income group"].unique())
    x = np.arange(len(income_order), dtype=float)
    bar_width = 0.24

    fig, (coverage_ax, count_ax) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        dpi=180,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )

    summary_by_group = summary.set_index("Income group").reindex(income_order)
    coverage_ax.bar(
        x - bar_width,
        summary_by_group["imagery_availability_pct"],
        width=bar_width,
        color="#0f766e",
        label="Imagery",
    )
    coverage_ax.bar(
        x,
        summary_by_group["osm_availability_pct"],
        width=bar_width,
        color="#7c3aed",
        label="OSM",
    )
    coverage_ax.bar(
        x + bar_width,
        summary_by_group["both_availability_pct"],
        width=bar_width,
        color="#b45309",
        label="Both",
    )
    coverage_ax.set_title("Data Availability by Economic Classification")
    coverage_ax.set_ylabel("Regions with data (%)")
    coverage_ax.set_xticks(x)
    coverage_ax.set_xticklabels(income_order, rotation=20, ha="right")
    coverage_ax.set_ylim(0, 105)
    coverage_ax.legend(frameon=False, loc="upper left")
    coverage_ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)

    rng = np.random.default_rng(42)
    points = availability.copy()
    x_lookup = {label: index for index, label in enumerate(income_order)}
    points["x"] = points["Income group"].map(x_lookup).astype(float)
    points["image_x"] = points["x"] + rng.uniform(-0.20, -0.04, size=len(points))
    points["osm_x"] = points["x"] + rng.uniform(0.04, 0.20, size=len(points))

    count_ax.scatter(
        points["image_x"],
        points["image_count"],
        s=18,
        alpha=0.38,
        color="#0f766e",
        edgecolors="none",
        label="Image count",
    )
    count_ax.scatter(
        points["osm_x"],
        points["osm_feature_count"],
        s=18,
        alpha=0.38,
        color="#7c3aed",
        edgecolors="none",
        label="OSM feature count",
    )

    count_ax.plot(
        x - 0.12,
        summary_by_group["median_images"],
        color="#064e3b",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="Median images",
    )
    count_ax.plot(
        x + 0.12,
        summary_by_group["median_osm_features"],
        color="#4c1d95",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="Median OSM features",
    )

    count_ax.set_xlabel("Economic classification")
    count_ax.set_ylabel("Per-region count, symlog scale")
    count_ax.set_xticks(x)
    count_ax.set_xticklabels(income_order, rotation=20, ha="right")
    count_ax.set_yscale("symlog", linthresh=1)
    count_ax.legend(frameon=False, loc="upper left", ncols=2)
    count_ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
    count_ax.grid(False, axis="x")

    total_regions = availability["region_id"].nunique()
    matched_countries = availability["iso3"].nunique()
    count_ax.text(
        0.995,
        0.02,
        f"regions={total_regions} | countries={matched_countries}",
        transform=count_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#374151",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    classifications = load_classification_table(args.classification_csv)
    db = DatabaseManager()

    availability = load_region_availability(db)
    availability = merge_with_classifications(availability, classifications)
    availability = add_availability_flags(
        availability,
        args.min_images,
        args.min_osm_features,
        args.osm_available_when_fetched,
    )
    summary = summarize_by_income_group(availability)

    plot_availability(availability, summary, args.output)

    args.per_region_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    availability.sort_values(
        ["Income group", "country", "city", "region_name"]
    ).to_csv(args.per_region_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    print(f"Saved availability plot to {args.output}")
    print(f"Saved per-region availability data to {args.per_region_csv}")
    print(f"Saved income-group summary to {args.summary_csv}")
    print(f"Regions plotted: {availability['region_id'].nunique()}")


if __name__ == "__main__":
    raise SystemExit(main())
