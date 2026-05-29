import argparse
import math
import textwrap
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, box

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager
from src.utils import setup_logger

logger = setup_logger(__name__)

LANDUSE_ORDER = [
    "commercial",
    "residential",
    "industrial",
    "retail",
    "park/leisure/green",
]

LANDUSE_COLOURS = {
    "commercial": "#f59e0b",
    "residential": "#facc15",
    "industrial": "#9333ea",
    "retail": "#e31a1c",
    "park/leisure/green": "#16a34a",
}

LANDUSE_PANEL_ALPHA = {
    "commercial": 0.06,
    "residential": 0.10,
    "industrial": 0.10,
    "retail": 0.10,
    "park/leisure/green": 0.10,
}

LANDUSE_POLYGON_ALPHA = {
    "commercial": 0.72,
    "residential": 0.52,
    "industrial": 0.52,
    "retail": 0.52,
    "park/leisure/green": 0.52,
}

COMMERCIAL_VALUES = {"commercial"}
RESIDENTIAL_VALUES = {"residential"}
INDUSTRIAL_VALUES = {"industrial"}
RETAIL_VALUES = {"retail"}
MIXED_VALUES = {"mixed", "mixed_use", "mixed-use"}
GREEN_LANDUSE_VALUES = {
    "grass",
    "forest",
    "meadow",
    "village_green",
    "recreation_ground",
    "allotments",
    "cemetery",
}
GREEN_LEISURE_VALUES = {
    "park",
    "garden",
    "nature_reserve",
    "common",
    "golf_course",
    "pitch",
    "playground",
    "dog_park",
    "sports_centre",
    "recreation_ground",
}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="LandUsePollutantGrid",
        description=(
            "Create a grid of static maps where rows are pollutant labels and "
            "columns are land-use classes."
        ),
    )
    parser.add_argument("region_ids", nargs="+", type=str, help="List of region IDs.")
    parser.add_argument(
        "--nearby-meters",
        type=float,
        default=150.0,
        help="Maximum distance in meters to match nearby land-use polygons.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=10.0,
        help="Point size for detection markers in the grid.",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=4.2,
        help="Base subplot size in inches.",
    )
    parser.add_argument(
        "--row-height-scale",
        type=float,
        default=0.52,
        help="Row height multiplier relative to --figscale for wide map panels.",
    )
    return parser.parse_args()


def normalize_region_ids(region_ids):
    normalized = []
    for token in region_ids:
        parts = [part.strip() for part in token.split(",") if part.strip()]
        normalized.extend(parts)
    return normalized


def normalize_label(label: Optional[str]) -> str:
    if not label:
        return "other"
    value = label.strip().lower()
    return value if value else "other"


def display_label(label: str) -> str:
    return label.replace("_", " ")


def format_row_label(label: str) -> str:
    formatted = display_label(label)
    custom_breaks = {
        "mobile advertisement": "mobile\nadvertise\nment",
        "utility pole": "utility\npole",
        "road sign": "road\nsign",
        "shop sign": "shop\nsign",
    }
    if formatted in custom_breaks:
        return custom_breaks[formatted]
    return "\n".join(
        textwrap.wrap(
            formatted,
            width=7,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def display_region_title(city: str, country: Optional[str]) -> str:
    city_text = (city or "").strip() or "Unknown city"
    country_text = (country or "").strip()
    if not country_text or country_text.lower() == "unknown country":
        return city_text
    return f"{city_text}, {country_text}"


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def normalize_tag_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip().lower()


def classify_landuse(landuse_value: Any, leisure_value: Any) -> Optional[str]:
    landuse = normalize_tag_value(landuse_value)
    leisure = normalize_tag_value(leisure_value)

    if landuse in MIXED_VALUES:
        return "mixed-use"
    if landuse in COMMERCIAL_VALUES:
        return "commercial"
    if landuse in RESIDENTIAL_VALUES:
        return "residential"
    if landuse in INDUSTRIAL_VALUES:
        return "industrial"
    if landuse in RETAIL_VALUES:
        return "retail"
    if landuse in GREEN_LANDUSE_VALUES or leisure in GREEN_LEISURE_VALUES:
        return "park/leisure/green"
    return None


def fetch_features_from_polygon(region_polygon, tags):
    if hasattr(ox, "features_from_polygon"):
        return ox.features_from_polygon(region_polygon, tags)
    return ox.features.features_from_polygon(region_polygon, tags)


def fetch_landuse_polygons(region):
    region_polygon = box(region.min_lng, region.min_lat, region.max_lng, region.max_lat)
    tags = {"landuse": True, "leisure": True}
    try:
        gdf = fetch_features_from_polygon(region_polygon, tags)
    except Exception as exc:
        logger.warning(f"Failed to fetch OSM land-use features for region {region.id}: {exc}")
        return gpd.GeoDataFrame(columns=["landuse_class", "geometry"], crs="EPSG:4326")

    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(columns=["landuse_class", "geometry"], crs="EPSG:4326")

    gdf = gdf.reset_index(drop=True)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    gdf["landuse_class"] = gdf.apply(
        lambda row: classify_landuse(row.get("landuse"), row.get("leisure")),
        axis=1,
    )
    gdf = gdf[gdf["landuse_class"].notna()].copy()
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["landuse_class", "geometry"], crs="EPSG:4326")

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["landuse_class", "geometry"], crs="EPSG:4326")

    return gdf[["landuse_class", "geometry"]].explode(index_parts=False).reset_index(drop=True)


def resolve_landuse_category(candidate_categories):
    categories = set(candidate_categories)
    if len(categories) > 1:
        if "residential" in categories and (
            "commercial" in categories or "retail" in categories or "industrial" in categories
        ):
            return "mixed-use"
    if "mixed-use" in categories:
        return "mixed-use"
    for landuse in LANDUSE_ORDER:
        if landuse in categories:
            return landuse
    return None


def assign_image_to_landuse(image_points_3857, polygons_3857, nearby_meters):
    polygons_view = polygons_3857[["landuse_class", "geometry"]]
    image_landuse = {}

    inside = gpd.sjoin(
        image_points_3857,
        polygons_view,
        how="left",
        predicate="within",
    )
    inside = inside[inside["landuse_class"].notna()].copy()
    if not inside.empty:
        grouped_inside = inside.groupby("image_id")["landuse_class"].apply(list)
        for image_id, categories in grouped_inside.items():
            resolved = resolve_landuse_category(categories)
            if resolved is not None:
                image_landuse[image_id] = resolved

    unmatched = image_points_3857[~image_points_3857["image_id"].isin(image_landuse.keys())].copy()
    if unmatched.empty:
        return image_landuse

    nearest = gpd.sjoin_nearest(
        unmatched,
        polygons_view,
        how="left",
        max_distance=nearby_meters,
        distance_col="distance_m",
    )
    nearest = nearest[nearest["landuse_class"].notna()].copy()
    if nearest.empty:
        return image_landuse

    nearest.sort_values(["image_id", "distance_m"], inplace=True)
    for image_id, group in nearest.groupby("image_id"):
        min_distance = group["distance_m"].min()
        categories = group[group["distance_m"] == min_distance]["landuse_class"].tolist()
        resolved = resolve_landuse_category(categories)
        if resolved is not None:
            image_landuse[image_id] = resolved

    return image_landuse


def build_detection_points(images_by_id, detections, image_landuse):
    grouped_points = defaultdict(list)
    detections_by_label_landuse = Counter()
    labels = set()

    for detection in detections:
        image = images_by_id.get(detection.image_id)
        if image is None or image.lng is None or image.lat is None:
            continue
        landuse = image_landuse.get(image.id)
        if landuse is None:
            continue

        label = normalize_label(detection.label)
        labels.add(label)
        detections_by_label_landuse[(label, landuse)] += 1
        grouped_points[(label, landuse)].append(Point(image.lng, image.lat))

    return sorted(labels), detections_by_label_landuse, grouped_points


def build_density_table(labels, images_by_landuse, detections_by_label_landuse):
    rows = []
    for label in labels:
        for landuse in LANDUSE_ORDER:
            image_count = int(images_by_landuse.get(landuse, 0))
            detection_count = int(detections_by_label_landuse.get((label, landuse), 0))
            detections_per_image = None
            if image_count > 0:
                detections_per_image = detection_count / image_count
            rows.append(
                {
                    "label": label,
                    "landuse": landuse,
                    "image_count": image_count,
                    "detection_count": detection_count,
                    "detections_per_image": detections_per_image,
                }
            )
    return pd.DataFrame(rows)


def make_detection_geodataframe(grouped_points):
    rows = []
    for (label, landuse), points in grouped_points.items():
        for point in points:
            rows.append({"label": label, "landuse": landuse, "geometry": point})
    if not rows:
        return gpd.GeoDataFrame(columns=["label", "landuse", "geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def save_grid_map(region, dissolved_polygons, detection_points_gdf, density_df, output_file, point_size, figscale, row_height_scale):
    label_order = (
        density_df.groupby("label", observed=False)["detection_count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    labels = label_order
    if not labels:
        raise ValueError("No labels available for grid map.")

    nrows = len(labels)
    ncols = len(LANDUSE_ORDER)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figscale * ncols, figscale * row_height_scale * nrows),
        dpi=200,
        squeeze=False,
    )

    xmin, ymin, xmax, ymax = dissolved_polygons.total_bounds
    xpad = max((xmax - xmin) * 0.03, 1e-5)
    ypad = max((ymax - ymin) * 0.01, 1e-5)

    density_lookup = {
        (row.label, row.landuse): (
            int(row.image_count),
            int(row.detection_count),
            row.detections_per_image,
        )
        for row in density_df.itertuples(index=False)
    }
    image_count_by_landuse = (
        density_df.groupby("landuse", observed=False)["image_count"]
        .max()
        .to_dict()
    )
    for row_index, label in enumerate(labels):
        for col_index, landuse in enumerate(LANDUSE_ORDER):
            ax = axes[row_index][col_index]
            ax.set_facecolor("#f8fafc")

            selected_polygons = dissolved_polygons[dissolved_polygons["landuse_class"] == landuse]
            if not selected_polygons.empty:
                selected_polygons.plot(
                    ax=ax,
                    color=LANDUSE_COLOURS.get(landuse, "#6b7280"),
                    edgecolor=LANDUSE_COLOURS.get(landuse, "#6b7280"),
                    linewidth=1.0 if landuse == "commercial" else 0.75,
                    alpha=LANDUSE_POLYGON_ALPHA.get(landuse, 0.52),
                )

            selected_points = detection_points_gdf[
                (detection_points_gdf["label"] == label)
                & (detection_points_gdf["landuse"] == landuse)
            ]
            if not selected_points.empty:
                selected_points.plot(
                    ax=ax,
                    color="#111827",
                    markersize=point_size * 1.15,
                    alpha=0.92,
                    edgecolor="#ffffff",
                    linewidth=0.25,
                )

            image_count, detection_count, density = density_lookup.get(
                (label, landuse),
                (0, 0, None),
            )
            density_text = "NA" if pd.isna(density) else f"{density:.2f}"
            panel_colour = LANDUSE_COLOURS.get(landuse, "#6b7280")
            ax.add_patch(
                plt.Rectangle(
                    (xmin - xpad, ymin - ypad),
                    (xmax - xmin) + (2 * xpad),
                    (ymax - ymin) + (2 * ypad),
                    facecolor=panel_colour,
                    edgecolor="none",
                    alpha=LANDUSE_PANEL_ALPHA.get(landuse, 0.10),
                    zorder=0,
                )
            )

            stat_lines = [
                f"Detections: {detection_count}",
                f"Density: {density_text}",
            ]
            ax.text(
                0.02,
                0.98,
                "\n".join(stat_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11.1,
                family="monospace",
                color="#111827",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.82,
                    "edgecolor": panel_colour,
                    "linewidth": 0.9,
                    "pad": 1.6,
                },
            )

            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)
            ax.set_aspect("auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(panel_colour)
                spine.set_alpha(0.55)
                spine.set_linewidth(1.1)

            if row_index == 0:
                header_image_count = int(image_count_by_landuse.get(landuse, 0))
                ax.set_title(
                    f"{display_label(landuse)}\nImages: {header_image_count}",
                    fontsize=12.5,
                    pad=4,
                    fontweight="semibold",
                    bbox={
                        "facecolor": "#ffffff",
                        "edgecolor": panel_colour,
                        "alpha": 0.9,
                        "boxstyle": "square,pad=0.2",
                    },
                )
            if col_index == 0:
                ax.set_ylabel(
                    format_row_label(label),
                    fontsize=13.5,
                    labelpad=4,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontweight="semibold",
                )

    fig.subplots_adjust(
        left=0.055,
        right=1.0,
        bottom=0.0,
        top=0.965,
        wspace=0.0,
        hspace=0.0,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def main():
    args = parse_args()
    region_ids = normalize_region_ids(args.region_ids)

    if not region_ids:
        print("No region IDs provided.")
        return 1

    if args.nearby_meters <= 0:
        print("--nearby-meters must be > 0.")
        return 1

    db = DatabaseManager()

    found = 0
    for region_id in region_ids:
        region_start = time.perf_counter()
        logger.info(f"Processing region {region_id}")
        region = db.get_region(region_id)
        if region is None:
            logger.warning(f"Region not found: {region_id}")
            continue

        found += 1
        images = db.get_images_by_region(region_id)
        detections = db.get_detections_by_region(region_id)
        if not images:
            logger.warning(f"No images found for region {region_id}")
            continue

        landuse_polygons = fetch_landuse_polygons(region)
        if landuse_polygons.empty:
            logger.warning(f"No OSM land-use polygons found for region {region_id}")
            continue

        dissolved_polygons = landuse_polygons.dissolve(by="landuse_class", as_index=False)
        if dissolved_polygons.crs is None:
            dissolved_polygons = dissolved_polygons.set_crs(epsg=4326)
        else:
            dissolved_polygons = dissolved_polygons.to_crs(epsg=4326)

        image_records = []
        images_by_id = {}
        for image in images:
            if image.lng is None or image.lat is None:
                continue
            images_by_id[image.id] = image
            image_records.append({"image_id": image.id, "geometry": Point(image.lng, image.lat)})

        if not image_records:
            logger.warning(f"No images with valid coordinates found for region {region_id}")
            continue

        image_points_4326 = gpd.GeoDataFrame(image_records, geometry="geometry", crs="EPSG:4326")
        image_points_3857 = image_points_4326.to_crs(epsg=3857)
        polygons_3857 = landuse_polygons.to_crs(epsg=3857).reset_index(drop=True)

        image_landuse = assign_image_to_landuse(image_points_3857, polygons_3857, args.nearby_meters)
        images_by_landuse = Counter(image_landuse.values())

        labels, detections_by_label_landuse, grouped_points = build_detection_points(
            images_by_id,
            detections,
            image_landuse,
        )
        if not labels:
            logger.warning(f"No matched detections found for region {region_id}")
            continue

        density_df = build_density_table(labels, images_by_landuse, detections_by_label_landuse)
        detection_points_gdf = make_detection_geodataframe(grouped_points)

        output_dir = Path(f"maps/{region.country}/{region.city}/land_use_pollutant_grid")
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_file = output_dir / f"{region.id}_density.csv"
        density_df.to_csv(csv_file, index=False)
        logger.info(f"Saved density table to {csv_file}")

        png_file = output_dir / f"{region.id}_grid.png"
        save_grid_map(
            region,
            dissolved_polygons,
            detection_points_gdf,
            density_df,
            png_file,
            args.point_size,
            args.figscale,
            args.row_height_scale,
        )
        logger.info(f"Saved grid map to {png_file}")
        logger.info(f"Finished region {region_id} in {time.perf_counter() - region_start:.2f}s")

    return 0 if found > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
