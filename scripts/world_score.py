import importlib.util
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from shapely.geometry import Point
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.api import BoundingBox
from src.database import DatabaseManager, Detection, Image
from src.mapping import WorldScores
from src.utils import RegionManager


PNG_DPI = 800
PNG_OUTPUT = Path("./maps/world_scores.png")
FIGSIZE = (18, 10)
MAP_EXTENT = (-168, 190, -57, 82)
NEUTRAL_SCORE_COLORS = {
    "no_score": "#737373",
    "zero_images": "#111827",
    "zero_detections": "#9ca3af",
    "zero_score": "#a78bfa",
}
NEUTRAL_SCORE_LABELS = {
    "no_score": "No score",
    "zero_images": "Zero images",
    "zero_detections": "Zero detections",
    "zero_score": "< 300 images",
}
BASE_WORLD_COLOR = "#eef3f6"
BASE_EDGE_COLOR = "#94a3b8"


def resolve_naturalearth_world_path():
    spec = importlib.util.find_spec("pyogrio")
    if spec is None or spec.origin is None:
        return None

    candidate = (
        Path(spec.origin).resolve().parent
        / "tests"
        / "fixtures"
        / "naturalearth_lowres"
        / "naturalearth_lowres.shp"
    )
    if candidate.exists():
        return candidate
    return None


def load_world_boundaries():
    world_path = resolve_naturalearth_world_path()
    if world_path is None:
        raise SystemExit("Could not find a local Natural Earth world boundary dataset.")

    world = gpd.read_file(world_path)
    world = world[world.geometry.notna()].copy()
    world = world[~world.geometry.is_empty].copy()
    if world.crs is None:
        world = world.set_crs(epsg=4326)
    else:
        world = world.to_crs(epsg=4326)
    return world


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, float(fraction))) * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def build_score_style(scores, band_count=10):
    sorted_scores = sorted(float(score) for score in scores if pd.notna(score) and float(score) > 0.0)
    if not sorted_scores:
        return None, None, []

    effective_band_count = max(2, min(int(band_count), len(sorted_scores)))
    thresholds = [min(sorted_scores)]
    for band_index in range(1, effective_band_count):
        thresholds.append(percentile(sorted_scores, band_index / effective_band_count))
    thresholds.append(max(sorted_scores))

    for index in range(1, len(thresholds)):
        if thresholds[index] <= thresholds[index - 1]:
            thresholds[index] = thresholds[index - 1] + 1e-9

    colors = [
        WorldScores.score_palette_color(
            band_index / max(1, effective_band_count - 1)
        )
        for band_index in range(effective_band_count)
    ]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(thresholds, cmap.N)
    return cmap, norm, thresholds


def scale_marker_sizes(densities, min_size=10.0, max_size=72.0):
    numeric = pd.to_numeric(densities, errors="coerce")
    valid = numeric[numeric > 0.0]
    if valid.empty:
        return pd.Series(min_size, index=densities.index)

    lower = valid.quantile(0.05)
    upper = valid.quantile(0.95)
    if upper <= lower:
        return pd.Series((min_size + max_size) / 2.0, index=densities.index)

    clipped = numeric.fillna(lower).clip(lower=lower, upper=upper)
    scaled = (clipped - lower) / (upper - lower)
    scaled = scaled.pow(0.75)
    return min_size + (scaled * (max_size - min_size))


def get_region_image_counts(db):
    rows = (
        db.session.query(Image.region_id, func.count(Image.id))
        .group_by(Image.region_id)
        .all()
    )
    return {region_id: int(image_count) for region_id, image_count in rows}


def get_region_detection_counts(db):
    rows = (
        db.session.query(Image.region_id, func.count(Detection.id))
        .join(Detection, Detection.image_id == Image.id)
        .group_by(Image.region_id)
        .all()
    )
    return {region_id: int(detection_count) for region_id, detection_count in rows}


def classify_score_category(score, image_count, detection_count):
    if score is None:
        return "no_score"
    if score > 0.0:
        return "positive"
    if image_count <= 0:
        return "zero_images"
    if detection_count <= 0:
        return "zero_detections"
    return "zero_score"


def build_region_points(db):
    regions = db.get_all_regions()
    image_counts = get_region_image_counts(db)
    detection_counts = get_region_detection_counts(db)
    density_by_city_country, density_by_city = WorldScores.load_ghs_density_lookup()
    rows = []

    for region in regions:
        if None in (region.min_lng, region.min_lat, region.max_lng, region.max_lat):
            continue

        bbox = BoundingBox(region.min_lng, region.min_lat, region.max_lng, region.max_lat)
        lng, lat = RegionManager.get_region_mid(bbox)
        if not (-180.0 <= lng <= 190.0 and -90.0 <= lat <= 90.0):
            continue

        score = float(region.score) if region.score is not None else None
        image_count = image_counts.get(region.id, 0)
        detection_count = detection_counts.get(region.id, 0)
        score_category = classify_score_category(score, image_count, detection_count)
        city_key = WorldScores.normalize_place_name(region.city)
        country_key = WorldScores.normalize_place_name(region.country)
        density = density_by_city_country.get((city_key, country_key))

        if density is None and city_key:
            city_matches = density_by_city.get(city_key, set())
            if len(city_matches) == 1:
                density = next(iter(city_matches))[1]

        if density is None:
            population = (
                float(region.population)
                if region.population is not None and float(region.population) > 0.0
                else None
            )
            if population is not None:
                area_km2 = WorldScores.estimate_bbox_area_km2(region)
                if area_km2 > 0.0:
                    density = population / area_km2

        rows.append(
            {
                "region_id": region.id,
                "city": region.city or "",
                "country": region.country or "",
                "score": score,
                "score_category": score_category,
                "has_positive_score": score_category == "positive",
                "image_count": image_count,
                "detection_count": detection_count,
                "density": density,
                "geometry": Point(lng, lat),
            }
        )

    if not rows:
        raise SystemExit("No mappable regions were found in the database.")

    points = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    points["marker_size"] = scale_marker_sizes(points["density"])
    return points


def plot_static_world_scores(points, output_path):
    world = load_world_boundaries()
    positive = points[points["has_positive_score"]].copy()
    cmap, norm, thresholds = build_score_style(positive["score"])

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=PNG_DPI)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#d7e0e3")

    world.plot(
        ax=ax,
        color=BASE_WORLD_COLOR,
        edgecolor=BASE_EDGE_COLOR,
        linewidth=0.8,
        zorder=1,
    )

    for category in NEUTRAL_SCORE_COLORS:
        category_points = points[points["score_category"] == category].copy()
        if category_points.empty:
            continue
        category_points.plot(
            ax=ax,
            color=NEUTRAL_SCORE_COLORS[category],
            markersize=category_points["marker_size"],
            alpha=0.45,
            edgecolor="white",
            linewidth=0.12,
            zorder=2,
        )

    if not positive.empty and cmap is not None:
        positive.plot(
            ax=ax,
            column="score",
            cmap=cmap,
            norm=norm,
            markersize=positive["marker_size"],
            alpha=0.76,
            edgecolor="white",
            linewidth=0.14,
            zorder=10,
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        colorbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            fraction=0.035,
            pad=0.018,
            shrink=0.52,
            boundaries=thresholds,
            ticks=[thresholds[0], thresholds[len(thresholds) // 2], thresholds[-1]],
        )
        colorbar.set_label(
            f"Region score percentile bands ({len(thresholds) - 1} bands, green = low, red = high)",
            fontsize=15,
            fontweight="bold",
        )
        colorbar.ax.tick_params(labelsize=13, width=0.8, length=3)

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="white",
            label=NEUTRAL_SCORE_LABELS[category],
        )
        for category, color in NEUTRAL_SCORE_COLORS.items()
        if (points["score_category"] == category).any()
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#94a3b8",
        framealpha=0.94,
        fontsize=13,
    )

    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    db = DatabaseManager()

    region_points = build_region_points(db)
    plot_static_world_scores(region_points, PNG_OUTPUT)
