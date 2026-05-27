import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager, Detection, Image as RegionImage, OSMFeature, Region  # noqa: E402


DEFAULT_TROPIC_LATITUDE = 23.4367
TROPIC_ZONE_ORDER = [
    "South of Tropic of Capricorn",
    "Between tropics",
    "North of Tropic of Cancer",
]

SOURCE_CONFIG = {
    "osm": {
        "title": "OSM Category Rates by Tropic Zone",
        "category_column": "osm_category",
        "folder_name": "osm_category_rates_by_tropic_zone",
        "csv_name": "osm_category_rates_by_tropic_zone.csv",
        "colour": "#1f77b4",
        "median_colour": "#b22222",
    },
    "detections": {
        "title": "Detection Label Rates by Tropic Zone",
        "category_column": "detection_label",
        "folder_name": "detection_label_rates_by_tropic_zone",
        "csv_name": "detection_label_rates_by_tropic_zone.csv",
        "colour": "#2a9d8f",
        "median_colour": "#a44a3f",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create separate OSM and detection rate scatter plots by whether "
            "each region lies north of the Tropic of Cancer, between the "
            "tropics, or south of the Tropic of Capricorn."
        )
    )
    parser.add_argument(
        "--tropic-latitude",
        type=float,
        default=DEFAULT_TROPIC_LATITUDE,
        help="Absolute latitude of the Tropics of Cancer and Capricorn.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("maps"),
        help="Base folder for generated plot folders.",
    )
    parser.add_argument(
        "--output-csv-root",
        type=Path,
        default=Path("data"),
        help="Base folder for generated CSV files.",
    )
    parser.add_argument(
        "--region-zone-csv",
        type=Path,
        default=Path("data/region_tropic_zones.csv"),
        help="Output per-region tropic-zone assignment CSV.",
    )
    parser.add_argument(
        "--min-osm-features",
        type=int,
        default=1,
        help="Minimum total OSM feature count for a region to be included in OSM plots.",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=1,
        help="Minimum total detection count for a region to be included in detection plots.",
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
    return parser.parse_args()


def classify_tropic_zone(center_lat, tropic_latitude):
    if center_lat > tropic_latitude:
        return "North of Tropic of Cancer"
    if center_lat < -tropic_latitude:
        return "South of Tropic of Capricorn"
    return "Between tropics"


def normalize_category_series(series):
    return series.fillna("").astype(str).str.strip().replace("", "<blank>")


def order_tropic_zones(values):
    ordered = [zone for zone in TROPIC_ZONE_ORDER if zone in values]
    extras = sorted(value for value in values if value not in TROPIC_ZONE_ORDER)
    return ordered + extras


def slugify(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "blank"


def load_region_zone_assignments(db, tropic_latitude):
    rows = []
    for region in db.get_all_regions():
        center_lat = (region.min_lat + region.max_lat) / 2
        rows.append(
            {
                "region_id": region.id,
                "region_name": region.name,
                "city": region.city or "",
                "country": region.country or "",
                "iso3": region.iso3 or "",
                "min_lat": region.min_lat,
                "max_lat": region.max_lat,
                "center_lat": center_lat,
                "tropic_latitude": tropic_latitude,
                "tropic_zone": classify_tropic_zone(center_lat, tropic_latitude),
            }
        )

    assignments = pd.DataFrame(rows)
    if assignments.empty:
        raise SystemExit("No regions found to classify by tropic zone.")
    return assignments


def load_osm_rates(db, min_total_features):
    region_totals = pd.DataFrame(
        db.session.query(
            Region.id.label("region_id"),
            func.count(OSMFeature.id).label("total_osm_features"),
        )
        .join(OSMFeature, OSMFeature.region_id == Region.id)
        .group_by(Region.id)
        .all(),
        columns=["region_id", "total_osm_features"],
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
    rates["rate"] = rates["category_count"] / rates["total_osm_features"]
    return rates


def load_detection_rates(db, min_total_detections):
    region_totals = pd.DataFrame(
        db.session.query(
            Region.id.label("region_id"),
            func.count(Detection.id).label("total_detections"),
        )
        .join(RegionImage, RegionImage.region_id == Region.id)
        .join(Detection, Detection.image_id == RegionImage.id)
        .group_by(Region.id)
        .all(),
        columns=["region_id", "total_detections"],
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
        .join(RegionImage, RegionImage.region_id == Region.id)
        .join(Detection, Detection.image_id == RegionImage.id)
        .group_by(Region.id, Detection.label)
        .all(),
        columns=["region_id", "detection_label", "label_count"],
    )

    rates = label_counts.merge(region_totals, on="region_id", how="inner")
    rates["detection_label"] = normalize_category_series(rates["detection_label"])
    rates["rate"] = rates["label_count"] / rates["total_detections"]
    return rates


def merge_with_zones(rates, zone_assignments):
    if rates.empty:
        return rates
    merged = rates.merge(zone_assignments, on="region_id", how="inner")
    return merged[merged["tropic_zone"].notna()].copy()


def filter_categories(data, category_column, requested_categories):
    if requested_categories is None:
        return data
    requested = {category.strip() for category in requested_categories if category.strip()}
    return data[data[category_column].isin(requested)].copy()


def plot_single_category(subset, category, output_path, title, colour, median_colour):
    zone_order = order_tropic_zones(subset["tropic_zone"].unique())
    x_lookup = {label: index for index, label in enumerate(zone_order)}
    rng = np.random.default_rng(42)

    points = subset.copy()
    points["x"] = points["tropic_zone"].map(x_lookup).astype(float)
    points["x"] = points["x"] + rng.uniform(-0.16, 0.16, size=len(points))

    fig, ax = plt.subplots(figsize=(8.6, 5.3), dpi=180)
    ax.scatter(
        points["x"],
        points["rate"],
        s=24,
        alpha=0.78,
        color=colour,
        edgecolors="none",
    )

    median_rates = (
        subset.groupby("tropic_zone", observed=False)["rate"]
        .median()
        .reindex(zone_order)
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

    ax.set_title(f"{title}\n{category}")
    ax.set_xlabel("Region latitude band")
    ax.set_ylabel("Rate within region")
    ax.set_xticks(range(len(zone_order)))
    ax.set_xticklabels(zone_order, rotation=25, ha="right")
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

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def export_source_plots(data, source_key, output_root, output_csv_root):
    config = SOURCE_CONFIG[source_key]
    category_column = config["category_column"]

    if data.empty:
        print(f"No matched data available for {source_key}.")
        return

    output_dir = output_root / config["folder_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = sorted(data[category_column].unique())
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
    ordered = data.sort_values(
        [category_column, "tropic_zone", "rate", "region_name"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(csv_path, index=False)

    print(f"Saved {len(categories)} {source_key} plot(s) to {output_dir}")
    print(f"Saved {source_key} per-region rates to {csv_path}")


def main():
    args = parse_args()
    db = DatabaseManager()
    zone_assignments = load_region_zone_assignments(db, args.tropic_latitude)

    args.region_zone_csv.parent.mkdir(parents=True, exist_ok=True)
    zone_assignments.to_csv(args.region_zone_csv, index=False)
    print(f"Saved region tropic zones to {args.region_zone_csv}")
    print(
        "Tropic zones found: "
        + ", ".join(order_tropic_zones(zone_assignments["tropic_zone"].dropna().unique()))
    )

    osm_rates = load_osm_rates(db, args.min_osm_features)
    osm_rates = merge_with_zones(osm_rates, zone_assignments)
    osm_rates = filter_categories(
        osm_rates,
        SOURCE_CONFIG["osm"]["category_column"],
        args.osm_categories,
    )

    detection_rates = load_detection_rates(db, args.min_detections)
    detection_rates = merge_with_zones(detection_rates, zone_assignments)
    detection_rates = filter_categories(
        detection_rates,
        SOURCE_CONFIG["detections"]["category_column"],
        args.detection_labels,
    )

    export_source_plots(
        osm_rates,
        "osm",
        args.output_root,
        args.output_csv_root,
    )
    export_source_plots(
        detection_rates,
        "detections",
        args.output_root,
        args.output_csv_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
