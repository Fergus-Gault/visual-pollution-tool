import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from shapely.geometry import Point, box
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config, ScoreConfig  # noqa: E402
from src.database import DatabaseManager, Detection, Image  # noqa: E402

ZERO_IMAGES_COLOR = "#eadcff"
MAP_OUTPUT_DPI = 400


def parse_args():
    parser = argparse.ArgumentParser(
        prog="MapCountryPollution",
        description=(
            "Create a ranked CSV, a static region score map, and a static "
            "hotspot plot for a country-wide scan."
        ),
    )
    parser.add_argument(
        "--country",
        default="United Kingdom",
        help="Country name stored on the region rows.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Deprecated: retained for compatibility but no longer limits the CSV export.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=ScoreConfig.IMAGES_PER_REGION_THRESHOLD,
        help="Minimum images required for a region to appear in the outputs.",
    )
    parser.add_argument(
        "--all-regions",
        action="store_true",
        help=(
            "Include all regions for the country. By default, only country-grid "
            "regions created by collect_country.py are used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="maps/country_pollution",
        help="Directory for generated map outputs and CSV summaries.",
    )
    parser.add_argument(
        "--colour-mode",
        choices=["linear", "banded"],
        default="banded",
        help="Use a continuous gradient or percentile-style score bands for region colouring.",
    )
    parser.add_argument(
        "--colour-bands",
        type=int,
        default=10,
        help="Number of percentile bands to use when --colour-mode=banded.",
    )
    parser.add_argument(
        "--show-major-cities",
        action="store_true",
        help="Overlay markers for major cities at their actual positions on the score map.",
    )
    parser.add_argument(
        "--major-city-count",
        type=int,
        default=20,
        help="How many of the country's most populous cities to show on the score map.",
    )
    return parser.parse_args()


def normalize(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def is_country_grid_region(region, country):
    city = normalize(region.city)
    return city.startswith(f"{normalize(country)} grid ")


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def format_float(value, digits=6):
    return f"{float(value):.{digits}f}"


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def hex_to_rgb(value):
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex colour, got {value!r}")
    return tuple(int(value[index:index + 2], 16) for index in range(0, 6, 2))


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def interpolate_rgb(start_rgb, end_rgb, fraction):
    clamped_fraction = clamp(float(fraction), 0.0, 1.0)
    return tuple(
        int(round(start_channel + (end_channel - start_channel) * clamped_fraction))
        for start_channel, end_channel in zip(start_rgb, end_rgb)
    )


def score_palette_color(fraction):
    green_rgb = hex_to_rgb("#1a9850")
    yellow_rgb = hex_to_rgb("#fee08b")
    red_rgb = hex_to_rgb("#d73027")

    clamped_fraction = clamp(float(fraction), 0.0, 1.0)
    if clamped_fraction <= 0.5:
        rgb = interpolate_rgb(green_rgb, yellow_rgb, clamped_fraction / 0.5)
    else:
        rgb = interpolate_rgb(yellow_rgb, red_rgb,
                              (clamped_fraction - 0.5) / 0.5)
    return rgb_to_hex(rgb)


def build_score_band_colors(band_count):
    base_colors = [
        "#1a9850",
        "#43ac5a",
        "#72c065",
        "#a6d96a",
        "#d9ef8b",
        "#fee08b",
        "#fdb863",
        "#f67f4b",
        "#e34a33",
        "#b30000",
    ]
    if band_count <= len(base_colors):
        return [base_colors[index] for index in range(band_count)]
    return [
        score_palette_color(band_index / max(1, band_count - 1))
        for band_index in range(band_count)
    ]


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = clamp(fraction, 0.0, 1.0) * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def quantile_range_label(start_fraction, end_fraction):
    start_percent = int(round(start_fraction * 100.0))
    end_percent = int(round(end_fraction * 100.0))
    if start_percent == 0:
        return f"Bottom {end_percent}%"
    return f"{start_percent}-{end_percent}%"


def format_score_tick(value):
    return f"{float(value):.5g}"


def get_country_regions(db, country, include_all_regions):
    matched = []
    country_key = normalize(country)
    for region in db.get_all_regions():
        if normalize(region.country) != country_key:
            continue
        if include_all_regions or is_country_grid_region(region, country):
            matched.append(region)

    matched.sort(
        key=lambda region: (
            region.city or "",
            region.scanned_at or "",
            region.id,
        )
    )
    return matched


def get_region_image_counts(db, region_ids):
    rows = (
        db.session.query(Image.region_id, func.count(Image.id))
        .filter(Image.region_id.in_(region_ids))
        .group_by(Image.region_id)
        .all()
    )
    return {region_id: int(image_count) for region_id, image_count in rows}


def get_region_detection_counts(db, region_ids):
    rows = (
        db.session.query(Image.region_id, func.count(Detection.id))
        .join(Detection, Detection.image_id == Image.id)
        .filter(Image.region_id.in_(region_ids))
        .group_by(Image.region_id)
        .all()
    )
    return {region_id: int(detection_count) for region_id, detection_count in rows}


def get_heatmap_points(db, region_ids):
    severity_by_label = ScoreConfig.SEVERITY_SCORES
    rows = (
        db.session.query(
            Image.id,
            Image.lat,
            Image.lng,
            Detection.label,
            Detection.confidence,
        )
        .join(Detection, Detection.image_id == Image.id)
        .filter(Image.region_id.in_(region_ids))
        .filter(Image.lat.isnot(None))
        .filter(Image.lng.isnot(None))
        .all()
    )

    image_weights = defaultdict(float)
    image_points = {}
    for image_id, lat, lng, label, confidence in rows:
        severity = float(severity_by_label.get(label, 0.0))
        if severity <= 0.0:
            continue
        confidence_weight = float(
            confidence) if confidence is not None else 1.0
        image_weights[image_id] += severity * \
            clamp(confidence_weight, 0.0, 1.0)
        image_points[image_id] = (float(lat), float(lng))

    heatmap_rows = []
    for image_id, weight in image_weights.items():
        point = image_points.get(image_id)
        if point is None or weight <= 0.0:
            continue
        heatmap_rows.append(
            {
                "lat": point[0],
                "lng": point[1],
                "weight": float(weight),
            }
        )
    return heatmap_rows


def load_worldcities(cities_path, country):
    entries = []
    country_key = normalize(country)
    with Path(cities_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if normalize(row.get("country")) != country_key:
                continue
            lat = parse_float(row.get("lat"))
            lng = parse_float(row.get("lng"))
            if lat is None or lng is None:
                continue
            entries.append(
                {
                    "city": (row.get("city_ascii") or row.get("city") or "").strip(),
                    "admin_name": (row.get("admin_name") or "").strip(),
                    "lat": lat,
                    "lng": lng,
                    "population": parse_int(row.get("population")) or 0,
                }
            )
    return entries


def get_major_cities(country, cities_path, limit):
    country_cities = load_worldcities(cities_path, country)
    country_cities.sort(
        key=lambda city_row: (
            -city_row["population"],
            city_row["city"].casefold(),
            city_row["admin_name"].casefold(),
        )
    )

    major_cities = []
    seen = set()
    for city_row in country_cities:
        dedupe_key = (
            city_row["city"].casefold(),
            round(city_row["lat"], 6),
            round(city_row["lng"], 6),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        major_cities.append(city_row)
        if len(major_cities) >= limit:
            break
    return major_cities


def build_region_rows(regions, image_counts, detection_counts, min_images):
    rows = []
    for region in regions:
        image_count = image_counts.get(region.id, 0)
        detection_count = detection_counts.get(region.id, 0)
        score = float(region.score) if region.score is not None else 0.0
        include_special_case = detection_count == 0 or score <= 0.0
        if image_count < min_images and not include_special_case:
            continue
        rows.append(
            {
                "region_id": region.id,
                "city": region.city or "",
                "country": region.country or "",
                "score": score,
                "images": image_count,
                "detections": detection_count,
                "detections_per_image": (
                    float(detection_count) /
                    float(image_count) if image_count else 0.0
                ),
                "min_lng": float(region.min_lng),
                "min_lat": float(region.min_lat),
                "max_lng": float(region.max_lng),
                "max_lat": float(region.max_lat),
                "region": region,
            }
        )

    rows.sort(
        key=lambda row: (
            -row["score"],
            -row["detections_per_image"],
            -row["detections"],
            row["region_id"],
        )
    )
    return rows


def save_ranked_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "region_id",
                "city",
                "country",
                "score",
                "images",
                "detections",
                "detections_per_image",
                "min_lng",
                "min_lat",
                "max_lng",
                "max_lat",
            ]
        )
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    rank,
                    row["region_id"],
                    row["city"],
                    row["country"],
                    format_float(row["score"]),
                    row["images"],
                    row["detections"],
                    format_float(row["detections_per_image"]),
                    format_float(row["min_lng"]),
                    format_float(row["min_lat"]),
                    format_float(row["max_lng"]),
                    format_float(row["max_lat"]),
                ]
            )


def fetch_country_boundary(country):
    boundary_gdf = ox.geocode_to_gdf(country)
    if boundary_gdf is None or boundary_gdf.empty:
        raise SystemExit(
            f"Could not fetch a country boundary for '{country}'.")
    boundary_gdf = boundary_gdf[boundary_gdf.geometry.notna()].copy()
    boundary_gdf = boundary_gdf[boundary_gdf.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"])]
    if boundary_gdf.empty:
        raise SystemExit(f"No polygon boundary was returned for '{country}'.")
    if boundary_gdf.crs is None:
        boundary_gdf = boundary_gdf.set_crs(epsg=4326)
    else:
        boundary_gdf = boundary_gdf.to_crs(epsg=4326)
    boundary_gdf = boundary_gdf[["geometry"]].explode(
        index_parts=False).reset_index(drop=True)
    return boundary_gdf.dissolve().reset_index(drop=True)


def build_display_boundary(boundary_gdf, country, focus_gdf=None):
    boundary_projected = boundary_gdf.to_crs(epsg=3857).explode(
        index_parts=False).reset_index(drop=True)
    if boundary_projected.empty:
        return boundary_projected

    boundary_projected["area_m2"] = boundary_projected.geometry.area
    largest_index = boundary_projected["area_m2"].idxmax()
    main_geometry = boundary_projected.loc[largest_index, "geometry"]
    largest_area = float(boundary_projected.loc[largest_index, "area_m2"])

    if normalize(country) == "united kingdom":
        keep_mask = (
            (boundary_projected["area_m2"] >= largest_area * 0.00015)
            | (boundary_projected.geometry.distance(main_geometry) <= 260000.0)
        )
        filtered = boundary_projected.loc[keep_mask, ["geometry"]].copy()
    else:
        filtered = boundary_projected[["geometry"]].copy()

    if focus_gdf is not None and not focus_gdf.empty:
        focus_projected = focus_gdf.to_crs(epsg=3857)
        focus_union = focus_projected.union_all()
        filtered = filtered.loc[filtered.geometry.intersects(focus_union)].copy()
        if filtered.empty:
            filtered = boundary_projected[["geometry"]].copy()

    return filtered.dissolve().reset_index(drop=True)


def build_region_geodataframe(rows, boundary_gdf):
    region_gdf = gpd.GeoDataFrame(
        rows,
        geometry=[
            box(row["min_lng"], row["min_lat"], row["max_lng"], row["max_lat"])
            for row in rows
        ],
        crs="EPSG:4326",
    )
    clipped = gpd.clip(region_gdf, boundary_gdf)
    clipped = clipped[clipped.geometry.notna()].copy()
    clipped = clipped[~clipped.geometry.is_empty].copy()
    return clipped


def build_focus_region_geodataframe(rows, boundary_gdf):
    data_rows = [row for row in rows if int(row["images"]) > 0]
    if not data_rows:
        data_rows = rows
    return build_region_geodataframe(data_rows, boundary_gdf)


def build_plotted_region_geodataframe(rows, boundary_gdf):
    return build_region_geodataframe(rows, boundary_gdf)


def build_major_city_geodataframe(major_cities, boundary_gdf):
    if not major_cities:
        return gpd.GeoDataFrame(columns=["city", "population", "geometry"], crs="EPSG:4326")
    city_gdf = gpd.GeoDataFrame(
        major_cities,
        geometry=[Point(city_row["lng"], city_row["lat"])
                  for city_row in major_cities],
        crs="EPSG:4326",
    )
    return gpd.sjoin(city_gdf, boundary_gdf, how="inner", predicate="within").drop(
        columns=["index_right"],
        errors="ignore",
    )


def build_heatmap_geodataframe(heatmap_points, boundary_gdf):
    if not heatmap_points:
        return gpd.GeoDataFrame(columns=["weight", "geometry"], crs="EPSG:4326")
    point_gdf = gpd.GeoDataFrame(
        heatmap_points,
        geometry=[Point(row["lng"], row["lat"]) for row in heatmap_points],
        crs="EPSG:4326",
    )
    return gpd.sjoin(point_gdf, boundary_gdf, how="inner", predicate="within").drop(
        columns=["index_right"],
        errors="ignore",
    )


def build_score_style(rows, colour_mode, colour_bands):
    scores = [float(row["score"]) for row in rows]
    if not scores:
        raise SystemExit("No score values were available to plot.")

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        max_score = min_score + 1e-9

    if colour_mode == "linear":
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "pollution_linear",
            ["#1a9850", "#fee08b", "#d73027"],
        )
        norm = mcolors.Normalize(vmin=min_score, vmax=max_score)
        return {
            "mode": "linear",
            "cmap": cmap,
            "norm": norm,
            "boundaries": None,
            "legend_patches": None,
        }

    band_count = max(2, int(colour_bands))
    percentile_edges = [band_index /
                        band_count for band_index in range(band_count + 1)]
    band_colors = build_score_band_colors(band_count)
    plot_color_by_region_id = {}
    legend_patches = []

    zero_images_color = ZERO_IMAGES_COLOR
    zero_detections_color = "#8B8989"
    positive_zero_score_color = "#81b5e6"

    zero_image_rows = [row for row in rows if int(row["images"]) == 0]
    zero_detection_rows = [
        row for row in rows if int(row["images"]) > 0 and int(row["detections"]) == 0
    ]
    positive_rows = [row for row in rows if float(row["score"]) > 0.0]

    if zero_image_rows:
        legend_patches.append(
            Patch(
                facecolor=zero_images_color,
                edgecolor="#303030",
                linewidth=0.5,
                label="Zero images",
            )
        )
        for row in zero_image_rows:
            plot_color_by_region_id[row["region_id"]] = zero_images_color

    if zero_detection_rows:
        legend_patches.append(
            Patch(
                facecolor=zero_detections_color,
                edgecolor="#303030",
                linewidth=0.5,
                label="Zero detections",
            )
        )
        for row in zero_detection_rows:
            plot_color_by_region_id[row["region_id"]] = zero_detections_color

    zero_score_rows = [
        row
        for row in rows
        if int(row["images"]) > 0 and int(row["detections"]) > 0 and float(row["score"]) <= 0.0
    ]
    if zero_score_rows:
        legend_patches.append(
            Patch(
                facecolor=positive_zero_score_color,
                edgecolor="#303030",
                linewidth=0.5,
                label="Zero score",
            )
        )
        for row in zero_score_rows:
            plot_color_by_region_id[row["region_id"]
                                    ] = positive_zero_score_color

    if not positive_rows:
        return {
            "mode": "banded",
            "cmap": None,
            "norm": None,
            "boundaries": None,
            "legend_patches": legend_patches,
            "plot_color_by_region_id": plot_color_by_region_id,
        }

    positive_rows.sort(key=lambda row: (float(row["score"]), row["region_id"]))
    total_positive = len(positive_rows)

    for band_index, color in enumerate(band_colors):
        lower_position = int(math.floor(
            percentile_edges[band_index] * total_positive))
        upper_position = int(math.floor(
            percentile_edges[band_index + 1] * total_positive))
        if band_index == band_count - 1:
            upper_position = total_positive
        upper_position = max(lower_position + 1, upper_position)
        band_rows = positive_rows[lower_position:upper_position]
        if not band_rows:
            continue

        for row in band_rows:
            plot_color_by_region_id[row["region_id"]] = color

        legend_patches.append(
            Patch(
                facecolor=color,
                edgecolor="#303030",
                linewidth=0.5,
                label=quantile_range_label(
                    percentile_edges[band_index],
                    percentile_edges[band_index + 1],
                ),
            )
        )

    for row in rows:
        if row["region_id"] not in plot_color_by_region_id:
            plot_color_by_region_id[row["region_id"]
                                    ] = positive_zero_score_color

    return {
        "mode": "banded",
        "cmap": None,
        "norm": None,
        "boundaries": None,
        "legend_patches": legend_patches,
        "plot_color_by_region_id": plot_color_by_region_id,
    }


def compute_figure_size(boundary_projected):
    min_x, min_y, max_x, max_y = boundary_projected.total_bounds
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    aspect = height / width
    figure_width = 8.8
    figure_height = clamp(figure_width * aspect, 7.0, 15.5)
    return figure_width, figure_height


def get_reference_figure_size(country, fallback_gdf):
    if normalize(country) == "united kingdom":
        return 8.5, 14.9
    return compute_figure_size(fallback_gdf)


def apply_axis_bounds(ax, focus_gdf, pad_fraction=0.02):
    min_x, min_y, max_x, max_y = focus_gdf.total_bounds
    width = max_x - min_x
    height = max_y - min_y
    pad_x = width * pad_fraction
    pad_y = height * pad_fraction
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)


def build_focus_geometry(source_gdf, pad_fraction=0.04):
    min_x, min_y, max_x, max_y = source_gdf.total_bounds
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    pad_x = width * pad_fraction
    pad_y = height * pad_fraction
    return gpd.GeoDataFrame(
        geometry=[box(min_x - pad_x, min_y - pad_y,
                      max_x + pad_x, max_y + pad_y)],
        crs=source_gdf.crs,
    )


def add_score_legend(fig, ax, score_style):
    if score_style["mode"] == "linear":
        colorbar = fig.colorbar(
            ScalarMappable(norm=score_style["norm"], cmap=score_style["cmap"]),
            ax=ax,
            fraction=0.038,
            pad=0.012,
        )
        colorbar.set_label("Pollution score")
        colorbar.ax.yaxis.label.set_size(17)
        colorbar.ax.yaxis.label.set_weight("bold")
        colorbar.ax.tick_params(labelsize=13)
        return

    legend = ax.legend(
        handles=score_style["legend_patches"],
        title="Pollution score bands",
        loc="center",
        bbox_to_anchor=(0.85, 0.6),
        frameon=True,
        borderaxespad=0.0,
        labelspacing=0.45,
        handlelength=1.6,
        handleheight=1.2,
        columnspacing=1.2,
        ncol=1,
        fontsize=14,
    )
    legend._legend_box.align = "left"
    legend.get_title().set_fontsize(16)
    legend.get_title().set_fontweight("bold")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor("none")


def create_region_score_map(
    country,
    rows,
    png_output_path,
    colour_mode,
    colour_bands,
    major_cities,
):
    boundary_gdf = fetch_country_boundary(country)
    region_gdf = build_plotted_region_geodataframe(rows, boundary_gdf)
    if region_gdf.empty:
        raise SystemExit(
            f"No plottable region polygons were available for '{country}'.")

    major_city_gdf = build_major_city_geodataframe(major_cities, boundary_gdf)
    focus_region_gdf = build_focus_region_geodataframe(rows, boundary_gdf)
    region_projected = region_gdf.to_crs(epsg=3857)
    focus_region_projected = focus_region_gdf.to_crs(epsg=3857)
    display_boundary_projected = build_display_boundary(
        boundary_gdf,
        country,
        focus_gdf=focus_region_gdf,
    )
    focus_projected = build_focus_geometry(
        focus_region_projected, pad_fraction=0.03)
    boundary_projected = gpd.clip(display_boundary_projected, focus_projected)
    if boundary_projected.empty:
        boundary_projected = display_boundary_projected
    major_city_projected = major_city_gdf.to_crs(
        epsg=3857) if not major_city_gdf.empty else None

    score_style = build_score_style(rows, colour_mode, colour_bands)
    fig_width, fig_height = get_reference_figure_size(
        country, display_boundary_projected)
    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height), constrained_layout=True)

    boundary_projected.plot(ax=ax, facecolor="#ffffff",
                            edgecolor="#ffffff", linewidth=0.9, zorder=1)
    if score_style["mode"] == "linear":
        region_projected.plot(
            ax=ax,
            column="score",
            cmap=score_style["cmap"],
            norm=score_style["norm"],
            linewidth=0.0,
            edgecolor="none",
            alpha=0.96,
            zorder=2,
        )
    else:
        region_projected = region_projected.copy()
        region_projected["plot_color"] = region_projected["region_id"].map(
            score_style["plot_color_by_region_id"]
        )
        region_projected.plot(
            ax=ax,
            color=region_projected["plot_color"],
            linewidth=0.0,
            edgecolor="none",
            alpha=0.96,
            zorder=2,
        )
    zero_image_projected = region_projected[region_projected["images"].astype(int) == 0]
    if not zero_image_projected.empty:
        zero_image_projected.plot(
            ax=ax,
            color=ZERO_IMAGES_COLOR,
            linewidth=0.15,
            edgecolor="#ffffff",
            alpha=0.98,
            zorder=3,
        )
    boundary_projected.boundary.plot(
        ax=ax, color="#ffffff", linewidth=0.8, zorder=4)
    if major_city_projected is not None and not major_city_projected.empty:
        ax.scatter(
            major_city_projected.geometry.x,
            major_city_projected.geometry.y,
            s=20,
            facecolor="#08306b",
            edgecolor="#ffffff",
            linewidth=0.5,
            alpha=0.9,
            zorder=5,
        )

    ax.set_axis_off()
    apply_axis_bounds(ax, focus_region_projected, pad_fraction=0.02)
    add_score_legend(fig, ax, score_style)

    png_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_output_path, dpi=MAP_OUTPUT_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_heatmap(country, heatmap_points, png_output_path):
    boundary_gdf = fetch_country_boundary(country)
    heatmap_gdf = build_heatmap_geodataframe(heatmap_points, boundary_gdf)
    if heatmap_gdf.empty:
        raise SystemExit(
            "Ranked CSV and region score map were created, but no weighted detection "
            "points were available for the hotspot plot."
        )

    heatmap_projected = heatmap_gdf.to_crs(epsg=3857)
    display_boundary_projected = build_display_boundary(
        boundary_gdf,
        country,
        focus_gdf=heatmap_gdf,
    )
    focus_projected = build_focus_geometry(
        heatmap_projected, pad_fraction=0.03)
    boundary_projected = gpd.clip(display_boundary_projected, focus_projected)
    if boundary_projected.empty:
        boundary_projected = display_boundary_projected

    fig_width, fig_height = get_reference_figure_size(
        country, display_boundary_projected)
    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height), constrained_layout=True)

    boundary_projected.plot(ax=ax, facecolor="#ffffff",
                            edgecolor="#ffffff", linewidth=0.9, zorder=1)
    hexbin = ax.hexbin(
        heatmap_projected.geometry.x.to_numpy(),
        heatmap_projected.geometry.y.to_numpy(),
        C=heatmap_projected["weight"].to_numpy(),
        reduce_C_function=np.sum,
        gridsize=85,
        mincnt=1,
        cmap="YlOrRd",
        linewidths=0.0,
        alpha=0.92,
        zorder=2,
    )
    boundary_projected.boundary.plot(
        ax=ax, color="#ffffff", linewidth=0.8, zorder=4)
    ax.set_axis_off()
    apply_axis_bounds(ax, heatmap_projected, pad_fraction=0.02)

    colorbar = fig.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("Weighted detections")

    png_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_output_path, dpi=MAP_OUTPUT_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    if args.min_images < 0:
        raise SystemExit("--min-images must be at least 0.")
    if args.top_n < 1:
        raise SystemExit("--top-n must be at least 1.")
    if args.colour_bands < 2:
        raise SystemExit("--colour-bands must be at least 2.")

    db = DatabaseManager()
    regions = get_country_regions(db, args.country, args.all_regions)
    if not regions:
        scope = "all regions" if args.all_regions else "country grid regions"
        raise SystemExit(f"No {scope} found for country '{args.country}'.")

    region_ids = [region.id for region in regions]
    image_counts = get_region_image_counts(db, region_ids)
    detection_counts = get_region_detection_counts(db, region_ids)
    rows = build_region_rows(
        regions,
        image_counts,
        detection_counts,
        args.min_images,
    )
    major_cities = []
    if args.show_major_cities:
        major_cities = get_major_cities(
            args.country,
            Config.DEFAULT_CSV,
            args.major_city_count,
        )

    if not rows:
        raise SystemExit(
            f"No regions found for '{args.country}' with at least {args.min_images} images."
        )

    output_dir = Path(args.output_dir)
    country_slug = normalize(args.country).replace(" ", "_")
    csv_path = output_dir / f"{country_slug}_top_regions.csv"
    regions_map_png_path = output_dir / f"{country_slug}_region_scores.png"
    heatmap_png_path = output_dir / f"{country_slug}_pollution_heatmap.png"

    save_ranked_csv(rows, csv_path)
    create_region_score_map(
        args.country,
        rows,
        regions_map_png_path,
        args.colour_mode,
        args.colour_bands,
        major_cities,
    )

    heatmap_points = get_heatmap_points(db, [row["region_id"] for row in rows])
    create_heatmap(
        args.country,
        heatmap_points,
        heatmap_png_path,
    )

    print(f"Saved ranked regions to {csv_path}")
    print(f"Saved region score PNG map to {regions_map_png_path}")
    print(f"Saved hotspot PNG map to {heatmap_png_path}")


if __name__ == "__main__":
    raise SystemExit(main())
