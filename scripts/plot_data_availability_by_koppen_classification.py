import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager, Image, OSMFeature, Region  # noqa: E402


KOPPEN_GROUP_MAP = {
    "A": "A: Tropical",
    "B": "B: Arid",
    "C": "C: Temperate",
    "D": "D: Cold",
    "E": "E: Polar",
}

DEFAULT_KOPPEN_GROUP_ORDER = [
    "A: Tropical",
    "B: Arid",
    "C: Temperate",
    "D: Cold",
    "E: Polar",
]

DEFAULT_KOPPEN_CLASS_MAP = {
    1: "Af",
    2: "Am",
    3: "Aw",
    4: "BWh",
    5: "BWk",
    6: "BSh",
    7: "BSk",
    8: "Csa",
    9: "Csb",
    10: "Csc",
    11: "Cwa",
    12: "Cwb",
    13: "Cwc",
    14: "Cfa",
    15: "Cfb",
    16: "Cfc",
    17: "Dsa",
    18: "Dsb",
    19: "Dsc",
    20: "Dsd",
    21: "Dwa",
    22: "Dwb",
    23: "Dwc",
    24: "Dwd",
    25: "Dfa",
    26: "Dfb",
    27: "Dfc",
    28: "Dfd",
    29: "ET",
    30: "EF",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot imagery and OSM data availability by dominant Koppen-Geiger "
            "climate group."
        )
    )
    parser.add_argument(
        "--koppen-tif",
        type=Path,
        default=Path("data/Beck_KG_V1_present_0p083.tif"),
        help="Path to the Beck Koppen-Geiger GeoTIFF.",
    )
    parser.add_argument(
        "--legend-txt",
        type=Path,
        default=Path("data/legend.txt"),
        help="Optional legend text file mapping raster values to Koppen classes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maps/data_availability_by_koppen_group.png"),
        help="Output plot image path.",
    )
    parser.add_argument(
        "--per-region-csv",
        type=Path,
        default=Path("data/data_availability_by_koppen_group_regions.csv"),
        help="Output per-region availability CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("data/data_availability_by_koppen_group_summary.csv"),
        help="Output Koppen-group summary CSV.",
    )
    parser.add_argument(
        "--climate-csv",
        type=Path,
        default=Path("data/region_koppen_groups.csv"),
        help="Output per-region Koppen assignment CSV.",
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
    parser.add_argument(
        "--max-window-pixels",
        type=int,
        default=250000,
        help=(
            "If a region's TIFF crop exceeds this many pixels, downsample it with "
            "nearest-neighbour before computing the dominant class."
        ),
    )
    return parser.parse_args()


def load_koppen_class_map(path: Path | None):
    if path is None or not path.exists():
        return DEFAULT_KOPPEN_CLASS_MAP

    pattern = re.compile(r"^\s*(\d+):\s*([A-Za-z]{2,3})\b")
    class_map = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        class_map[int(match.group(1))] = match.group(2)

    return class_map or DEFAULT_KOPPEN_CLASS_MAP


class KoppenRaster:
    def __init__(self, path: Path, max_window_pixels: int, class_map: dict[int, str]):
        if not path.exists():
            raise SystemExit(f"Koppen TIFF not found: {path}")

        self.max_window_pixels = max(1, int(max_window_pixels))
        self.class_map = class_map
        self.dataset = rasterio.open(path)
        self.bounds = self.dataset.bounds

    def dominant_class_for_bbox(self, min_lng, min_lat, max_lng, max_lat):
        clamped_min_lng = max(min_lng, self.bounds.left)
        clamped_max_lng = min(max_lng, self.bounds.right)
        clamped_min_lat = max(min_lat, self.bounds.bottom)
        clamped_max_lat = min(max_lat, self.bounds.top)

        if clamped_min_lng >= clamped_max_lng or clamped_min_lat >= clamped_max_lat:
            return None, None, None

        raw_window = from_bounds(
            clamped_min_lng,
            clamped_min_lat,
            clamped_max_lng,
            clamped_max_lat,
            self.dataset.transform,
        )
        col_off = int(math.floor(raw_window.col_off))
        row_off = int(math.floor(raw_window.row_off))
        col_end = int(math.ceil(raw_window.col_off + raw_window.width))
        row_end = int(math.ceil(raw_window.row_off + raw_window.height))

        col_off = max(0, min(self.dataset.width - 1, col_off))
        row_off = max(0, min(self.dataset.height - 1, row_off))
        col_end = max(col_off + 1, min(self.dataset.width, col_end))
        row_end = max(row_off + 1, min(self.dataset.height, row_end))

        window = Window(
            col_off=col_off,
            row_off=row_off,
            width=col_end - col_off,
            height=row_end - row_off,
        )
        if window.width <= 0 or window.height <= 0:
            return None, None, None

        output_shape = None
        window_pixels = int(window.width * window.height)
        if window_pixels > self.max_window_pixels:
            scale = (self.max_window_pixels / float(window_pixels)) ** 0.5
            output_shape = (
                1,
                max(1, int(window.height * scale)),
                max(1, int(window.width * scale)),
            )

        values = self.dataset.read(
            1,
            window=window,
            out_shape=output_shape,
            resampling=Resampling.nearest,
        )
        flat_values = np.asarray(values, dtype=np.uint8).reshape(-1)
        if flat_values.size == 0:
            return None, None, None

        unique_values, counts = np.unique(flat_values, return_counts=True)
        valid = [
            (int(value), int(count))
            for value, count in zip(unique_values, counts)
            if int(value) in self.class_map
        ]
        if not valid:
            return None, None, None

        dominant_value, dominant_count = max(valid, key=lambda item: item[1])
        total_valid = sum(count for _, count in valid)
        confidence = dominant_count / total_valid if total_valid else None
        return dominant_value, self.class_map[dominant_value], confidence


def normalize_iso3_series(series):
    return series.fillna("").astype(str).str.strip().str.upper()


def order_koppen_groups(values):
    ordered = [label for label in DEFAULT_KOPPEN_GROUP_ORDER if label in values]
    extras = sorted(value for value in values if value not in DEFAULT_KOPPEN_GROUP_ORDER)
    return ordered + extras


def load_region_climate_assignments(db, raster: KoppenRaster):
    rows = []
    for region in db.get_all_regions():
        climate_value, climate_class, climate_confidence = raster.dominant_class_for_bbox(
            region.min_lng,
            region.min_lat,
            region.max_lng,
            region.max_lat,
        )
        rows.append(
            {
                "region_id": region.id,
                "region_name": region.name,
                "city": region.city or "",
                "country": region.country or "",
                "iso3": region.iso3 or "",
                "min_lng": region.min_lng,
                "min_lat": region.min_lat,
                "max_lng": region.max_lng,
                "max_lat": region.max_lat,
                "koppen_code": climate_value,
                "koppen_class": climate_class,
                "koppen_group": (
                    KOPPEN_GROUP_MAP.get(climate_class[0])
                    if climate_class
                    else None
                ),
                "koppen_dominance": climate_confidence,
            }
        )

    assignments = pd.DataFrame(rows)
    assignments = assignments[assignments["koppen_class"].notna()].copy()
    if assignments.empty:
        raise SystemExit(
            "No regions could be assigned a Koppen-Geiger class from the TIFF."
        )
    return assignments


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


def merge_with_climate(availability, climate_assignments):
    if availability.empty:
        return availability

    climate_columns = [
        "region_id",
        "koppen_code",
        "koppen_class",
        "koppen_group",
        "koppen_dominance",
    ]
    merged = availability.merge(
        climate_assignments[climate_columns],
        on="region_id",
        how="inner",
    )
    return merged[merged["koppen_group"].notna()].copy()


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


def summarize_by_koppen_group(availability):
    if availability.empty:
        return pd.DataFrame()

    grouped = availability.groupby("koppen_group", observed=False)
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
        median_koppen_dominance=("koppen_dominance", "median"),
    ).reset_index()

    for column in ["imagery", "osm", "both"]:
        summary[f"{column}_availability_pct"] = (
            summary[f"regions_with_{column}"] / summary["region_count"] * 100
        )

    group_order = order_koppen_groups(summary["koppen_group"].unique())
    summary["group_order"] = summary["koppen_group"].map(
        {group: index for index, group in enumerate(group_order)}
    )
    return summary.sort_values("group_order").drop(columns=["group_order"])


def plot_availability(availability, summary, output_path):
    if availability.empty or summary.empty:
        raise SystemExit("No matched availability data available to plot.")

    group_order = order_koppen_groups(summary["koppen_group"].unique())
    x = np.arange(len(group_order), dtype=float)
    bar_width = 0.24

    fig, (coverage_ax, count_ax) = plt.subplots(
        2,
        1,
        figsize=(10.8, 8.0),
        dpi=180,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )

    summary_by_group = summary.set_index("koppen_group").reindex(group_order)
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
    coverage_ax.set_title("Data Availability by Koppen-Geiger Climate Group")
    coverage_ax.set_ylabel("Regions with data (%)")
    coverage_ax.set_xticks(x)
    coverage_ax.set_xticklabels(group_order, rotation=25, ha="right")
    coverage_ax.set_ylim(0, 105)
    coverage_ax.legend(frameon=False, loc="upper left")
    coverage_ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)

    rng = np.random.default_rng(42)
    points = availability.copy()
    x_lookup = {label: index for index, label in enumerate(group_order)}
    points["x"] = points["koppen_group"].map(x_lookup).astype(float)
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

    count_ax.set_xlabel("Dominant Koppen-Geiger group for region")
    count_ax.set_ylabel("Per-region count, symlog scale")
    count_ax.set_xticks(x)
    count_ax.set_xticklabels(group_order, rotation=25, ha="right")
    count_ax.set_yscale("symlog", linthresh=1)
    count_ax.legend(frameon=False, loc="upper left", ncols=2)
    count_ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
    count_ax.grid(False, axis="x")

    total_regions = availability["region_id"].nunique()
    koppen_classes = availability["koppen_class"].nunique()
    count_ax.text(
        0.995,
        0.02,
        f"regions={total_regions} | Koppen classes={koppen_classes}",
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
    db = DatabaseManager()
    class_map = load_koppen_class_map(args.legend_txt)
    raster = KoppenRaster(args.koppen_tif, args.max_window_pixels, class_map)
    climate_assignments = load_region_climate_assignments(db, raster)

    args.climate_csv.parent.mkdir(parents=True, exist_ok=True)
    climate_assignments.to_csv(args.climate_csv, index=False)

    availability = load_region_availability(db)
    availability = merge_with_climate(availability, climate_assignments)
    availability = add_availability_flags(
        availability,
        args.min_images,
        args.min_osm_features,
        args.osm_available_when_fetched,
    )
    summary = summarize_by_koppen_group(availability)

    plot_availability(availability, summary, args.output)

    args.per_region_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    availability.sort_values(
        ["koppen_group", "koppen_class", "country", "city", "region_name"]
    ).to_csv(args.per_region_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    print(f"Saved region Koppen groups to {args.climate_csv}")
    print(f"Saved availability plot to {args.output}")
    print(f"Saved per-region availability data to {args.per_region_csv}")
    print(f"Saved Koppen-group summary to {args.summary_csv}")
    print(f"Regions plotted: {availability['region_id'].nunique()}")
    print(
        "Koppen groups found: "
        + ", ".join(sorted(availability["koppen_group"].dropna().unique()))
    )


if __name__ == "__main__":
    raise SystemExit(main())
