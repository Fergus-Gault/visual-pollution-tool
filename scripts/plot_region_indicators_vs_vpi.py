import argparse
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sqlalchemy import func

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager, Image, Region  # noqa: E402


EARTH_RADIUS_KM = 6371.0088
PLOT_DPI = 300

INDICATORS = [
    {
        "column": "images_per_square_km",
        "label": "Images per square km",
        "scale": "log",
        "color": "#0f766e",
    },
    {
        "column": "gdp",
        "label": "GDP (current US$)",
        "scale": "log",
        "color": "#2563eb",
    },
    {
        "column": "gdppp",
        "label": "GDP per capita (current US$)",
        "scale": "log",
        "color": "#7c3aed",
    },
    {
        "column": "gni",
        "label": "GNI per capita, Atlas method (current US$)",
        "scale": "log",
        "color": "#c2410c",
    },
    {
        "column": "urb",
        "label": "Urban population (% of total)",
        "scale": "linear",
        "color": "#be123c",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        prog="PlotRegionIndicatorsVsVpi",
        description=(
            "Plot region VPI score against image density and country-level "
            "economic/urban indicators. Only regions with at least the requested "
            "number of images are included."
        ),
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=300,
        help="Minimum image count required for a region to be plotted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maps/vpi_region_indicator_relationships.png"),
        help="Output path for the combined scatter plot.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/vpi_region_indicator_relationships.csv"),
        help="Output CSV containing the plotted region rows.",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Optional country filter.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Optional city filter.",
    )
    parser.add_argument(
        "--positive-scores-only",
        action="store_true",
        help="Exclude regions where VPI score is zero or negative.",
    )
    return parser.parse_args()


def normalise(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def region_area_square_km(row):
    min_lat = math.radians(max(-90.0, min(90.0, float(row["min_lat"]))))
    max_lat = math.radians(max(-90.0, min(90.0, float(row["max_lat"]))))
    min_lng = math.radians(float(row["min_lng"]))
    max_lng = math.radians(float(row["max_lng"]))

    lat_span = abs(math.sin(max_lat) - math.sin(min_lat))
    lng_span = abs(max_lng - min_lng)
    if lng_span > math.tau:
        lng_span = math.tau

    return (EARTH_RADIUS_KM ** 2) * lat_span * lng_span


def build_query(db, args):
    query = (
        db.session.query(
            Region.id.label("region_id"),
            Region.name.label("region_name"),
            Region.city.label("city"),
            Region.country.label("country"),
            Region.iso3.label("iso3"),
            Region.min_lng.label("min_lng"),
            Region.min_lat.label("min_lat"),
            Region.max_lng.label("max_lng"),
            Region.max_lat.label("max_lat"),
            Region.score.label("vpi_score"),
            Region.images_per_square_km.label("images_per_square_km"),
            Region.gdp.label("gdp"),
            Region.gdp_year.label("gdp_year"),
            Region.gdppp.label("gdppp"),
            Region.gdppp_year.label("gdppp_year"),
            Region.gni.label("gni"),
            Region.gni_year.label("gni_year"),
            Region.urb.label("urb"),
            Region.urb_year.label("urb_year"),
            func.count(Image.id).label("image_count"),
        )
        .join(Image, Image.region_id == Region.id)
        .filter(Region.score.isnot(None))
        .group_by(
            Region.id,
            Region.name,
            Region.city,
            Region.country,
            Region.iso3,
            Region.min_lng,
            Region.min_lat,
            Region.max_lng,
            Region.max_lat,
            Region.score,
            Region.images_per_square_km,
            Region.gdp,
            Region.gdp_year,
            Region.gdppp,
            Region.gdppp_year,
            Region.gni,
            Region.gni_year,
            Region.urb,
            Region.urb_year,
        )
        .having(func.count(Image.id) >= args.min_images)
    )
    if args.country:
        query = query.filter(
            func.lower(func.coalesce(Region.country, "")) == normalise(args.country)
        )
    if args.city:
        query = query.filter(
            func.lower(func.coalesce(Region.city, "")) == normalise(args.city)
        )
    return query


def load_region_data(db, args):
    rows = build_query(db, args).all()
    data = pd.DataFrame([dict(row._mapping) for row in rows])
    if data.empty:
        return data

    numeric_columns = [
        "vpi_score",
        "images_per_square_km",
        "gdp",
        "gdppp",
        "gni",
        "urb",
        "image_count",
        "min_lng",
        "min_lat",
        "max_lng",
        "max_lat",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    missing_density = data["images_per_square_km"].isna()
    if missing_density.any():
        areas = data.loc[missing_density].apply(region_area_square_km, axis=1)
        valid_area = areas > 0
        data.loc[missing_density, "images_per_square_km"] = np.nan
        data.loc[areas[valid_area].index, "images_per_square_km"] = (
            data.loc[areas[valid_area].index, "image_count"] / areas[valid_area]
        )

    if args.positive_scores_only:
        data = data[data["vpi_score"] > 0].copy()

    return data.sort_values(
        ["country", "city", "region_name", "region_id"],
        na_position="last",
    ).reset_index(drop=True)


def add_trend_line(ax, subset, x_column, scale, color):
    if len(subset) < 3:
        return

    x = subset[x_column].astype(float)
    y = subset["vpi_score"].astype(float)
    fit_x = np.log10(x) if scale == "log" else x
    if fit_x.nunique() < 2:
        return

    slope, intercept = np.polyfit(fit_x, y, 1)
    x_line = np.linspace(fit_x.min(), fit_x.max(), 100)
    y_line = slope * x_line + intercept
    plot_x = np.power(10, x_line) if scale == "log" else x_line
    ax.plot(plot_x, y_line, color=color, linewidth=1.8, alpha=0.92)


def plot_indicator_panel(ax, data, indicator):
    column = indicator["column"]
    scale = indicator["scale"]
    color = indicator["color"]

    subset = data[["vpi_score", column, "image_count"]].dropna().copy()
    if scale == "log":
        subset = subset[subset[column] > 0].copy()

    if subset.empty:
        ax.set_title(indicator["label"])
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#6b7280",
        )
        return

    marker_sizes = np.sqrt(subset["image_count"].clip(lower=1)) * 1.15
    marker_sizes = marker_sizes.clip(lower=16, upper=85)

    ax.scatter(
        subset[column],
        subset["vpi_score"],
        s=marker_sizes,
        color=color,
        alpha=0.48,
        edgecolors="white",
        linewidth=0.35,
    )
    add_trend_line(ax, subset, column, scale, color)

    if scale == "log":
        ax.set_xscale("log")
    ax.set_title(indicator["label"])
    ax.set_xlabel(indicator["label"])
    ax.set_ylabel("VPI score")
    ax.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.7)


def plot_relationships(data, output_path, min_images):
    if data.empty:
        raise SystemExit("No regions matched the requested filters.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=PLOT_DPI)
    axes = axes.flatten()

    for ax, indicator in zip(axes, INDICATORS):
        plot_indicator_panel(ax, data, indicator)

    axes[-1].set_visible(False)
    fig.suptitle(
        f"VPI Score vs Region Coverage and Country Indicators ({min_images}+ images)",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_plotted_data(data, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "region_id",
        "region_name",
        "city",
        "country",
        "iso3",
        "image_count",
        "vpi_score",
        "images_per_square_km",
        "gdp",
        "gdp_year",
        "gdppp",
        "gdppp_year",
        "gni",
        "gni_year",
        "urb",
        "urb_year",
    ]
    data[columns].to_csv(output_csv, index=False)


def main():
    args = parse_args()
    if args.min_images < 1:
        raise SystemExit("--min-images must be at least 1.")

    db = DatabaseManager()
    data = load_region_data(db, args)
    plot_relationships(data, args.output, args.min_images)
    save_plotted_data(data, args.output_csv)

    print(f"Saved plot to {args.output}")
    print(f"Saved plotted data to {args.output_csv}")
    print(f"Regions plotted: {len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
