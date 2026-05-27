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
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DatabaseConfig  # noqa: E402

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
            "Plot image capture time-of-day split by whether each image has "
            "at least one detection, split by Koppen-Geiger climate group."
        )
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres SQLAlchemy URL. Defaults to DATABASE_URL from auth/.env.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Optional region city filter, e.g. Edinburgh.",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Optional region country filter.",
    )
    parser.add_argument(
        "--timezone",
        default="Europe/London",
        help="Timezone used to convert capture timestamps before extracting hour.",
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
        "--max-window-pixels",
        type=int,
        default=250000,
        help=(
            "If a region's TIFF crop exceeds this many pixels, downsample it with "
            "nearest-neighbour before computing the dominant class."
        ),
    )
    parser.add_argument(
        "--climate-csv",
        type=Path,
        default=Path("data/detection_capture_time_region_koppen_groups.csv"),
        help="Output per-region Koppen assignment CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maps/detection_capture_time_of_day.png"),
        help="Output plot image path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/detection_capture_time_of_day.csv"),
        help="Output aggregated CSV path.",
    )
    return parser.parse_args()


def get_database_url(database_url):
    if database_url:
        return database_url
    return DatabaseConfig.get_postgres_url()


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


def order_koppen_groups(values):
    ordered = [label for label in DEFAULT_KOPPEN_GROUP_ORDER if label in values]
    extras = sorted(value for value in values if value not in DEFAULT_KOPPEN_GROUP_ORDER)
    return ordered + extras


def load_capture_times(database_url, city=None, country=None):
    filters = ["images.source_captured_at IS NOT NULL"]
    params = {}
    if city:
        filters.append("lower(coalesce(regions.city, '')) = lower(:city)")
        params["city"] = city
    if country:
        filters.append("lower(coalesce(regions.country, '')) = lower(:country)")
        params["country"] = country

    where_clause = " AND ".join(filters)
    query = text(
        f"""
        SELECT
            images.id AS image_id,
            regions.id AS region_id,
            regions.name AS region_name,
            regions.city,
            regions.country,
            regions.iso3,
            regions.min_lng,
            regions.min_lat,
            regions.max_lng,
            regions.max_lat,
            images.source_captured_at,
            CASE WHEN count(detections.id) > 0 THEN true ELSE false END AS has_detection
        FROM images
        JOIN regions ON regions.id = images.region_id
        LEFT JOIN detections ON detections.image_id = images.id
        WHERE {where_clause}
        GROUP BY
            images.id,
            regions.id,
            regions.name,
            regions.city,
            regions.country,
            regions.iso3,
            regions.min_lng,
            regions.min_lat,
            regions.max_lng,
            regions.max_lat,
            images.source_captured_at
        """
    )
    engine = create_engine(database_url, poolclass=NullPool)
    with engine.connect() as connection:
        rows = pd.read_sql_query(query, connection, params=params)

    if rows.empty:
        raise SystemExit("No images found for the selected filters.")
    rows["source_captured_at"] = pd.to_datetime(
        rows["source_captured_at"],
        errors="coerce",
        utc=True,
    )
    rows = rows.dropna(subset=["source_captured_at"])
    if rows.empty:
        raise SystemExit("No images had parseable source_captured_at values.")
    return rows


def assign_koppen_groups(images, raster):
    region_columns = [
        "region_id",
        "region_name",
        "city",
        "country",
        "iso3",
        "min_lng",
        "min_lat",
        "max_lng",
        "max_lat",
    ]
    regions = images[region_columns].drop_duplicates("region_id").copy()
    rows = []
    for region in regions.itertuples(index=False):
        climate_value, climate_class, climate_confidence = raster.dominant_class_for_bbox(
            region.min_lng,
            region.min_lat,
            region.max_lng,
            region.max_lat,
        )
        rows.append(
            {
                "region_id": region.region_id,
                "region_name": region.region_name,
                "city": region.city or "",
                "country": region.country or "",
                "iso3": region.iso3 or "",
                "koppen_code": climate_value,
                "koppen_class": climate_class,
                "koppen_group": (
                    KOPPEN_GROUP_MAP.get(climate_class[0]) if climate_class else None
                ),
                "koppen_dominance": climate_confidence,
            }
        )

    assignments = pd.DataFrame(rows)
    assignments = assignments[assignments["koppen_class"].notna()].copy()
    if assignments.empty:
        raise SystemExit("No selected regions could be assigned a Koppen-Geiger class.")

    images = images.merge(
        assignments[
            [
                "region_id",
                "koppen_code",
                "koppen_class",
                "koppen_group",
                "koppen_dominance",
            ]
        ],
        on="region_id",
        how="inner",
    )
    if images.empty:
        raise SystemExit("No images remained after joining Koppen-Geiger groups.")
    return images, assignments


def aggregate_by_hour(images, timezone):
    images = images.copy()
    images["local_captured_at"] = images["source_captured_at"].dt.tz_convert(timezone)
    images["capture_hour"] = images["local_captured_at"].dt.hour
    grouped = (
        images.groupby(["koppen_group", "capture_hour", "has_detection"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={True: "images_with_detections", False: "images_without_detections"})
        .reset_index()
    )

    groups = order_koppen_groups(images["koppen_group"].dropna().unique())
    summary_index = pd.MultiIndex.from_product(
        [groups, range(24)],
        names=["koppen_group", "capture_hour"],
    ).to_frame(index=False)
    summary = summary_index.merge(
        grouped,
        on=["koppen_group", "capture_hour"],
        how="left",
    ).fillna(0)
    for column in ["images_with_detections", "images_without_detections"]:
        if column not in summary.columns:
            summary[column] = 0
        summary[column] = summary[column].astype(int)
    summary["total_images"] = (
        summary["images_with_detections"] + summary["images_without_detections"]
    )
    summary["detection_image_rate"] = (
        summary["images_with_detections"] / summary["total_images"].replace(0, pd.NA)
    ).fillna(0)
    summary["time_of_day"] = summary["capture_hour"].map(label_hour)
    return summary


def label_hour(hour):
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 21:
        return "evening/twilight"
    return "night"


def add_time_of_day_bands(ax):
    bands = [
        (-0.5, 5.5, "#111827", "Night"),
        (5.5, 8.5, "#f59e0b", "Morning/twilight"),
        (8.5, 17.5, "#fef3c7", "Day"),
        (17.5, 20.5, "#f59e0b", "Evening/twilight"),
        (20.5, 23.5, "#111827", "Night"),
    ]
    for start, end, color, label in bands:
        ax.axvspan(start, end, color=color, alpha=0.07, linewidth=0)
        ax.text(
            (start + end) / 2,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#374151",
        )


def plot_capture_times(summary, output_path, title_suffix, timezone):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    koppen_groups = order_koppen_groups(summary["koppen_group"].unique())
    if not koppen_groups:
        raise SystemExit("No Koppen groups available to plot.")

    fig, axes = plt.subplots(
        len(koppen_groups),
        2,
        figsize=(15, max(4, 3.2 * len(koppen_groups))),
        dpi=180,
        sharex=True,
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    if len(koppen_groups) == 1:
        axes = np.asarray([axes])

    for row_index, koppen_group in enumerate(koppen_groups):
        subset = summary[summary["koppen_group"] == koppen_group].copy()
        x = subset["capture_hour"]
        count_ax, rate_ax = axes[row_index]

        add_time_of_day_bands(count_ax)
        width = 0.42
        count_ax.bar(
            x - width / 2,
            subset["images_without_detections"],
            width=width,
            color="#2166ac",
            alpha=0.78,
            label="Images without detections",
        )
        count_ax.bar(
            x + width / 2,
            subset["images_with_detections"],
            width=width,
            color="#b2182b",
            alpha=0.78,
            label="Images with detections",
        )
        count_ax.set_ylabel(f"{koppen_group}\nImage count")
        count_ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)
        if row_index == 0:
            count_ax.set_title(
                f"Image Capture Time of Day by Detection Status{title_suffix}"
            )
            count_ax.legend(frameon=False, loc="upper left")

        add_time_of_day_bands(rate_ax)
        rate_ax.plot(
            x,
            subset["detection_image_rate"],
            color="#111827",
            linewidth=2.0,
        )
        rate_ax.set_ylabel("Detection\nimage rate")
        rate_ax.set_ylim(0, 1)
        rate_ax.set_xlim(-0.5, 23.5)
        rate_ax.set_xticks(range(24))
        rate_ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)
        if row_index == 0:
            rate_ax.set_title("Detection image rate")

        total_images = int(subset["total_images"].sum())
        rate_ax.text(
            0.98,
            0.88,
            f"images={total_images}",
            transform=rate_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#374151",
        )

    for ax in axes[-1]:
        ax.set_xlabel(f"Local capture hour ({timezone})")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    database_url = get_database_url(args.database_url)
    images = load_capture_times(database_url, city=args.city, country=args.country)
    class_map = load_koppen_class_map(args.legend_txt)
    raster = KoppenRaster(args.koppen_tif, args.max_window_pixels, class_map)
    images, climate_assignments = assign_koppen_groups(images, raster)
    summary = aggregate_by_hour(images, args.timezone)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    args.climate_csv.parent.mkdir(parents=True, exist_ok=True)
    climate_assignments.to_csv(args.climate_csv, index=False)

    filters = []
    if args.city:
        filters.append(args.city)
    if args.country:
        filters.append(args.country)
    title_suffix = f" ({', '.join(filters)})" if filters else ""
    plot_capture_times(summary, args.output, title_suffix, args.timezone)

    print(f"Saved capture time-of-day plot to {args.output}")
    print(f"Saved hourly capture CSV to {args.output_csv}")
    print(f"Saved region Koppen groups to {args.climate_csv}")
    print(
        "Koppen groups found: "
        + ", ".join(order_koppen_groups(climate_assignments["koppen_group"].dropna().unique()))
    )
    print(f"Images with detections: {int(images['has_detection'].sum())}")
    print(f"Images without detections: {int((~images['has_detection']).sum())}")


if __name__ == "__main__":
    raise SystemExit(main())
