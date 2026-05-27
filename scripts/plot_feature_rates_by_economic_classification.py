import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager, Detection, Image, OSMFeature, Region  # noqa: E402


DEFAULT_INCOME_ORDER = [
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "High income",
]

SOURCE_CONFIG = {
    "osm": {
        "title": "OSM Category Rates by Economic Classification",
        "category_column": "osm_category",
        "category_label": "OSM category",
        "count_column": "category_count",
        "total_column": "total_osm_features",
        "folder_name": "osm_category_rates_by_income_group",
        "csv_name": "osm_category_rates_by_income_group.csv",
        "combined_name": "osm_category_rates_by_income_group.png",
        "colour": "#1f77b4",
        "median_colour": "#b22222",
    },
    "detections": {
        "title": "Detection Label Rates by Economic Classification",
        "category_column": "detection_label",
        "category_label": "Detection label",
        "count_column": "label_count",
        "total_column": "total_detections",
        "folder_name": "detection_label_rates_by_income_group",
        "csv_name": "detection_label_rates_by_income_group.csv",
        "combined_name": "detection_label_rates_by_income_group.png",
        "colour": "#2a9d8f",
        "median_colour": "#a44a3f",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create separate scatter plots for OSM categories and detection labels "
            "showing per-region feature rates by economic classification."
        )
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=Path("data/CLASS_2025_10_07.csv"),
        help="CSV containing Code and Income group columns.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("maps"),
        help="Base folder for the generated plot folders.",
    )
    parser.add_argument(
        "--output-csv-root",
        type=Path,
        default=Path("data"),
        help="Base folder for generated rate CSV files.",
    )
    parser.add_argument(
        "--min-osm-features",
        type=int,
        default=1,
        help="Minimum total OSM feature count required for a region to be included in OSM plots.",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=1,
        help="Minimum total detection count required for a region to be included in detection plots.",
    )
    parser.add_argument(
        "--osm-categories",
        nargs="+",
        default=None,
        help="Optional list of OSM categories to include.",
    )
    parser.add_argument(
        "--detection-labels",
        nargs="+",
        default=None,
        help="Optional list of detection labels to include.",
    )
    parser.add_argument(
        "--exclude-osm-categories",
        nargs="+",
        default=None,
        help="Optional list of OSM categories to exclude.",
    )
    parser.add_argument(
        "--exclude-detection-labels",
        nargs="+",
        default=None,
        help="Optional list of detection labels to exclude.",
    )
    parser.add_argument(
        "--subplot-columns",
        type=int,
        default=3,
        help="Number of columns to use in the combined subplot figures.",
    )
    parser.add_argument(
        "--save-individual-plots",
        action="store_true",
        help="Also save one PNG per OSM category and detection label.",
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


def load_osm_rates(db, min_total_features):
    region_totals = pd.DataFrame(
        db.session.query(
            Region.id.label("region_id"),
            Region.name.label("region_name"),
            Region.city.label("city"),
            Region.country.label("country"),
            Region.iso3.label("iso3"),
            func.count(OSMFeature.id).label("total_osm_features"),
        )
        .join(OSMFeature, OSMFeature.region_id == Region.id)
        .group_by(Region.id, Region.name, Region.city, Region.country, Region.iso3)
        .all(),
        columns=[
            "region_id",
            "region_name",
            "city",
            "country",
            "iso3",
            "total_osm_features",
        ],
    )

    if region_totals.empty:
        return pd.DataFrame()

    region_totals = region_totals[
        region_totals["total_osm_features"] >= min_total_features
    ].copy()
    if region_totals.empty:
        return pd.DataFrame()

    category_counts = pd.DataFrame(
        db.session.query(
            Region.id.label("region_id"),
            OSMFeature.osm_type.label("osm_category"),
            func.count(OSMFeature.id).label("category_count"),
        )
        .join(OSMFeature, OSMFeature.region_id == Region.id)
        .group_by(Region.id, OSMFeature.osm_type)
        .all(),
        columns=["region_id", "osm_category", "category_count"],
    )

    rates = category_counts.merge(region_totals, on="region_id", how="inner")
    rates["osm_category"] = normalize_category_series(rates["osm_category"])
    rates["iso3"] = normalize_iso3_series(rates["iso3"])
    rates["rate"] = rates["category_count"] / rates["total_osm_features"]
    return rates


def load_detection_rates(db, min_total_detections):
    region_totals = pd.DataFrame(
        db.session.query(
            Region.id.label("region_id"),
            Region.name.label("region_name"),
            Region.city.label("city"),
            Region.country.label("country"),
            Region.iso3.label("iso3"),
            func.count(Detection.id).label("total_detections"),
        )
        .join(Image, Image.region_id == Region.id)
        .join(Detection, Detection.image_id == Image.id)
        .group_by(Region.id, Region.name, Region.city, Region.country, Region.iso3)
        .all(),
        columns=[
            "region_id",
            "region_name",
            "city",
            "country",
            "iso3",
            "total_detections",
        ],
    )

    if region_totals.empty:
        return pd.DataFrame()

    region_totals = region_totals[
        region_totals["total_detections"] >= min_total_detections
    ].copy()
    if region_totals.empty:
        return pd.DataFrame()

    label_counts = pd.DataFrame(
        db.session.query(
            Region.id.label("region_id"),
            Detection.label.label("detection_label"),
            func.count(Detection.id).label("label_count"),
        )
        .join(Image, Image.region_id == Region.id)
        .join(Detection, Detection.image_id == Image.id)
        .group_by(Region.id, Detection.label)
        .all(),
        columns=["region_id", "detection_label", "label_count"],
    )

    rates = label_counts.merge(region_totals, on="region_id", how="inner")
    rates["detection_label"] = normalize_category_series(rates["detection_label"])
    rates["iso3"] = normalize_iso3_series(rates["iso3"])
    rates["rate"] = rates["label_count"] / rates["total_detections"]
    return rates


def normalize_iso3_series(series):
    return series.fillna("").astype(str).str.strip().str.upper()


def normalize_category_series(series):
    return series.fillna("").astype(str).str.strip().replace("", "<blank>")


def merge_with_classifications(rates, classifications):
    if rates.empty:
        return rates

    merged = rates.merge(
        classifications,
        left_on="iso3",
        right_on="Code",
        how="left",
    )
    merged = merged[
        merged["Income group"].notna() & (merged["Income group"] != "")
    ].copy()
    return merged


def order_income_groups(values):
    ordered = [group for group in DEFAULT_INCOME_ORDER if group in values]
    extras = sorted(value for value in values if value not in DEFAULT_INCOME_ORDER)
    return ordered + extras


def slugify(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "blank"


def filter_categories(data, category_column, requested_categories):
    if requested_categories is None:
        return data
    requested = {category.strip() for category in requested_categories if category.strip()}
    return data[data[category_column].isin(requested)].copy()


def exclude_categories(data, category_column, excluded_categories):
    if excluded_categories is None:
        return data
    excluded = {category.strip() for category in excluded_categories if category.strip()}
    return data[~data[category_column].isin(excluded)].copy()


def draw_category_panel(ax, subset, category, colour, median_colour, show_xlabel=True):
    income_order = order_income_groups(subset["Income group"].unique())
    x_lookup = {label: index for index, label in enumerate(income_order)}
    rng = np.random.default_rng(42)

    points = subset.copy()
    points["x"] = points["Income group"].map(x_lookup).astype(float)
    points["x"] = points["x"] + rng.uniform(-0.16, 0.16, size=len(points))

    ax.scatter(
        points["x"],
        points["rate"],
        s=24,
        alpha=0.78,
        color=colour,
        edgecolors="none",
    )

    median_rates = (
        subset.groupby("Income group", observed=False)["rate"]
        .median()
        .reindex(income_order)
    )
    valid_medians = median_rates.dropna()
    if not valid_medians.empty:
        ax.plot(
            [x_lookup[group] for group in valid_medians.index],
            valid_medians.values,
            color=median_colour,
            linewidth=1.8,
            marker="o",
            markersize=4,
        )

    ax.set_title(category)
    ax.set_xlabel("Economic classification" if show_xlabel else "")
    ax.set_ylabel("Rate within region")
    ax.set_xticks(range(len(income_order)))
    ax.set_xticklabels(income_order, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
    ax.grid(False, axis="x")

    summary = (
        f"regions={subset['region_id'].nunique()} | "
        f"points={len(subset)} | "
        f"median={subset['rate'].median():.3f}"
    )
    ax.text(
        0.99,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#374151",
    )


def plot_single_category(subset, category, output_path, title, colour, median_colour):
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
    draw_category_panel(ax, subset, category, colour, median_colour)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_combined_categories(data, source_key, output_path, subplot_columns):
    config = SOURCE_CONFIG[source_key]
    category_column = config["category_column"]
    categories = sorted(data[category_column].unique())
    if not categories:
        return 0

    columns = max(1, int(subplot_columns))
    rows = math.ceil(len(categories) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.1 * columns, 3.8 * rows),
        dpi=180,
        squeeze=False,
    )

    for index, category in enumerate(categories):
        row = index // columns
        column = index % columns
        ax = axes[row][column]
        subset = data[data[category_column] == category].copy()
        draw_category_panel(
            ax,
            subset,
            category,
            config["colour"],
            config["median_colour"],
            show_xlabel=(row == rows - 1),
        )

    for index in range(len(categories), rows * columns):
        row = index // columns
        column = index % columns
        axes[row][column].set_visible(False)

    fig.suptitle(config["title"], y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return len(categories)


def export_source_plots(data, source_key, output_root, output_csv_root, subplot_columns, save_individual_plots):
    config = SOURCE_CONFIG[source_key]
    category_column = config["category_column"]

    if data.empty:
        print(f"No matched data available for {source_key}.")
        return

    categories = sorted(data[category_column].unique())
    combined_path = output_root / config["combined_name"]
    plot_count = plot_combined_categories(
        data,
        source_key,
        combined_path,
        subplot_columns,
    )

    if save_individual_plots:
        output_dir = output_root / config["folder_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        for category in categories:
            subset = data[data[category_column] == category].copy()
            output_path = output_dir / f"{slugify(category)}.png"
            plot_single_category(
                subset,
                category,
                output_path,
                config["title"],
                config["colour"],
                config["median_colour"],
            )

    csv_path = output_csv_root / config["csv_name"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = data.sort_values(
        [category_column, "Income group", "rate", "region_name"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    ordered.to_csv(csv_path, index=False)

    print(f"Saved {plot_count} {source_key} subplot(s) to {combined_path}")
    if save_individual_plots:
        print(f"Saved {len(categories)} individual {source_key} plot(s) to {output_dir}")
    print(f"Saved {source_key} per-region rates to {csv_path}")


def main():
    args = parse_args()
    classifications = load_classification_table(args.classification_csv)
    db = DatabaseManager()

    osm_rates = load_osm_rates(db, args.min_osm_features)
    osm_rates = merge_with_classifications(osm_rates, classifications)
    osm_rates = filter_categories(
        osm_rates,
        SOURCE_CONFIG["osm"]["category_column"],
        args.osm_categories,
    )
    osm_rates = exclude_categories(
        osm_rates,
        SOURCE_CONFIG["osm"]["category_column"],
        args.exclude_osm_categories,
    )

    detection_rates = load_detection_rates(db, args.min_detections)
    detection_rates = merge_with_classifications(detection_rates, classifications)
    detection_rates = filter_categories(
        detection_rates,
        SOURCE_CONFIG["detections"]["category_column"],
        args.detection_labels,
    )
    detection_rates = exclude_categories(
        detection_rates,
        SOURCE_CONFIG["detections"]["category_column"],
        args.exclude_detection_labels,
    )

    export_source_plots(
        osm_rates,
        "osm",
        args.output_root,
        args.output_csv_root,
        args.subplot_columns,
        args.save_individual_plots,
    )
    export_source_plots(
        detection_rates,
        "detections",
        args.output_root,
        args.output_csv_root,
        args.subplot_columns,
        args.save_individual_plots,
    )


if __name__ == "__main__":
    raise SystemExit(main())
