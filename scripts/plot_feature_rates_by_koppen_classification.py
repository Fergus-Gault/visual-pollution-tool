import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager, Detection, Image as RegionImage, OSMFeature, Region  # noqa: E402

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

SOURCE_CONFIG = {
    "osm": {
        "title": "OSM Category Rates by Koppen-Geiger Group",
        "category_column": "osm_category",
        "folder_name": "osm_category_rates_by_koppen_group",
        "csv_name": "osm_category_rates_by_koppen_group.csv",
        "colour": "#1f77b4",
        "median_colour": "#b22222",
    },
    "detections": {
        "title": "Detection Label Rates by Koppen-Geiger Group",
        "category_column": "detection_label",
        "folder_name": "detection_label_rates_by_koppen_group",
        "csv_name": "detection_label_rates_by_koppen_group.csv",
        "colour": "#2a9d8f",
        "median_colour": "#a44a3f",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create separate OSM and detection rate scatter plots by dominant "
            "Koppen-Geiger climate group for each region."
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
        "--max-window-pixels",
        type=int,
        default=250000,
        help=(
            "If a region's TIFF crop exceeds this many pixels, downsample it with "
            "nearest-neighbour before computing the dominant class."
        ),
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

        self.path = path
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


def normalize_category_series(series):
    return series.fillna("").astype(str).str.strip().replace("", "<blank>")


def order_koppen_groups(values):
    ordered = [label for label in DEFAULT_KOPPEN_GROUP_ORDER if label in values]
    extras = sorted(
        value for value in values if value not in DEFAULT_KOPPEN_GROUP_ORDER)
    return ordered + extras


def slugify(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "blank"


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
                    KOPPEN_GROUP_MAP.get(
                        climate_class[0]) if climate_class else None
                ),
                "koppen_dominance": climate_confidence,
            }
        )

    assignments = pd.DataFrame(rows)
    assignments = assignments[assignments["koppen_class"].notna()].copy()
    if assignments.empty:
        raise SystemExit(
            "No regions could be assigned a Koppen-Geiger class from the TIFF.")
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
    rates["detection_label"] = normalize_category_series(
        rates["detection_label"])
    rates["rate"] = rates["label_count"] / rates["total_detections"]
    return rates


def merge_with_climate(rates, climate_assignments):
    if rates.empty:
        return rates
    merged = rates.merge(climate_assignments, on="region_id", how="inner")
    return merged[merged["koppen_group"].notna()].copy()


def filter_categories(data, category_column, requested_categories):
    if requested_categories is None:
        return data
    requested = {category.strip()
                 for category in requested_categories if category.strip()}
    return data[data[category_column].isin(requested)].copy()


def plot_single_category(subset, category, output_path, title, colour, median_colour):
    import matplotlib.pyplot as plt

    koppen_order = order_koppen_groups(subset["koppen_group"].unique())
    x_lookup = {label: index for index, label in enumerate(koppen_order)}
    rng = np.random.default_rng(42)

    points = subset.copy()
    points["x"] = points["koppen_group"].map(x_lookup).astype(float)
    points["x"] = points["x"] + rng.uniform(-0.16, 0.16, size=len(points))

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=180)
    ax.scatter(
        points["x"],
        points["rate"],
        s=24,
        alpha=0.78,
        color=colour,
        edgecolors="none",
    )

    median_rates = (
        subset.groupby("koppen_group", observed=False)["rate"]
        .median()
        .reindex(koppen_order)
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
    ax.set_xlabel("Dominant Koppen-Geiger group for region")
    ax.set_ylabel("Rate within region")
    ax.set_xticks(range(len(koppen_order)))
    ax.set_xticklabels(koppen_order, rotation=35, ha="right")
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
        [category_column, "koppen_group", "rate", "region_name"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(csv_path, index=False)

    print(f"Saved {len(categories)} {source_key} plot(s) to {output_dir}")
    print(f"Saved {source_key} per-region rates to {csv_path}")


def main():
    args = parse_args()

    db = DatabaseManager()
    class_map = load_koppen_class_map(args.legend_txt)
    raster = KoppenRaster(args.koppen_tif, args.max_window_pixels, class_map)
    climate_assignments = load_region_climate_assignments(db, raster)

    climate_csv = args.output_csv_root / "region_koppen_groups.csv"
    climate_csv.parent.mkdir(parents=True, exist_ok=True)
    climate_assignments.to_csv(climate_csv, index=False)
    print(f"Saved region Koppen groups to {climate_csv}")
    print(
        "Koppen groups found: "
        + ", ".join(sorted(climate_assignments["koppen_group"].dropna().unique()))
    )
    print(
        "Koppen classes found: "
        + ", ".join(sorted(climate_assignments["koppen_class"].dropna().unique()))
    )

    osm_rates = load_osm_rates(db, args.min_osm_features)
    osm_rates = merge_with_climate(osm_rates, climate_assignments)
    osm_rates = filter_categories(
        osm_rates,
        SOURCE_CONFIG["osm"]["category_column"],
        args.osm_categories,
    )

    detection_rates = load_detection_rates(db, args.min_detections)
    detection_rates = merge_with_climate(detection_rates, climate_assignments)
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
