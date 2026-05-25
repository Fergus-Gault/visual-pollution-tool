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
    "commercial": "#1f78b4",
    "residential": "#33a02c",
    "industrial": "#ff7f00",
    "retail": "#e31a1c",
    "park/leisure/green": "#2ca25f",
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
        prog="LandUseOSMGrid",
        description=(
            "Create a grid of static maps where rows are OSM feature types and "
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
        help="Point size for OSM markers in the grid.",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=4.2,
        help="Base subplot width in inches.",
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


def normalize_osm_type(osm_type: Optional[str]) -> str:
    if not osm_type:
        return "other"
    value = osm_type.strip().lower()
    return value if value else "other"


def display_label(label: str) -> str:
    return label.replace("_", " ")


def display_region_title(city: str, country: Optional[str]) -> str:
    city_text = (city or "").strip() or "Unknown city"
    country_text = (country or "").strip()
    if not country_text or country_text.lower() == "unknown country":
        return city_text
    return f"{city_text}, {country_text}"


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


def assign_points_to_landuse(points_3857, polygons_3857, nearby_meters):
    polygons_view = polygons_3857[["landuse_class", "geometry"]]
    point_landuse = {}

    inside = gpd.sjoin(
        points_3857,
        polygons_view,
        how="left",
        predicate="within",
    )
    inside = inside[inside["landuse_class"].notna()].copy()
    if not inside.empty:
        grouped_inside = inside.groupby("point_id")["landuse_class"].apply(list)
        for point_id, categories in grouped_inside.items():
            resolved = resolve_landuse_category(categories)
            if resolved is not None:
                point_landuse[point_id] = resolved

    unmatched = points_3857[~points_3857["point_id"].isin(point_landuse.keys())].copy()
    if unmatched.empty:
        return point_landuse

    nearest = gpd.sjoin_nearest(
        unmatched,
        polygons_view,
        how="left",
        max_distance=nearby_meters,
        distance_col="distance_m",
    )
    nearest = nearest[nearest["landuse_class"].notna()].copy()
    if nearest.empty:
        return point_landuse

    nearest.sort_values(["point_id", "distance_m"], inplace=True)
    for point_id, group in nearest.groupby("point_id"):
        min_distance = group["distance_m"].min()
        categories = group[group["distance_m"] == min_distance]["landuse_class"].tolist()
        resolved = resolve_landuse_category(categories)
        if resolved is not None:
            point_landuse[point_id] = resolved

    return point_landuse


def build_osm_points(osm_features):
    point_rows = []
    feature_lookup = {}
    for feature in osm_features:
        if feature.lng is None or feature.lat is None:
            continue
        feature_type = normalize_osm_type(feature.osm_type)
        point_id = feature.id
        feature_lookup[point_id] = {
            "osm_type": feature_type,
            "geometry": Point(feature.lng, feature.lat),
        }
        point_rows.append({"point_id": point_id, "geometry": Point(feature.lng, feature.lat)})
    return feature_lookup, point_rows


def aggregate_osm_by_landuse(feature_lookup, point_landuse):
    grouped_points = defaultdict(list)
    features_by_type_landuse = Counter()
    osm_types = set()

    for point_id, landuse in point_landuse.items():
        feature = feature_lookup.get(point_id)
        if feature is None:
            continue
        osm_type = feature["osm_type"]
        osm_types.add(osm_type)
        features_by_type_landuse[(osm_type, landuse)] += 1
        grouped_points[(osm_type, landuse)].append(feature["geometry"])

    return sorted(osm_types), features_by_type_landuse, grouped_points


def build_density_table(osm_types, points_by_landuse, features_by_type_landuse):
    rows = []
    for osm_type in osm_types:
        for landuse in LANDUSE_ORDER:
            point_count = int(points_by_landuse.get(landuse, 0))
            feature_count = int(features_by_type_landuse.get((osm_type, landuse), 0))
            features_per_point = None
            if point_count > 0:
                features_per_point = feature_count / point_count
            rows.append(
                {
                    "osm_type": osm_type,
                    "landuse": landuse,
                    "point_count": point_count,
                    "feature_count": feature_count,
                    "features_per_point": features_per_point,
                }
            )
    return pd.DataFrame(rows)


def make_osm_geodataframe(grouped_points):
    rows = []
    for (osm_type, landuse), points in grouped_points.items():
        for point in points:
            rows.append({"osm_type": osm_type, "landuse": landuse, "geometry": point})
    if not rows:
        return gpd.GeoDataFrame(columns=["osm_type", "landuse", "geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def save_grid_map(region, dissolved_polygons, osm_points_gdf, density_df, output_file, point_size, figscale, row_height_scale):
    type_order = (
        density_df.groupby("osm_type", observed=False)["feature_count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    osm_types = type_order
    if not osm_types:
        raise ValueError("No OSM feature types available for grid map.")

    nrows = len(osm_types)
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
        (row.osm_type, row.landuse): (
            int(row.point_count),
            int(row.feature_count),
            row.features_per_point,
        )
        for row in density_df.itertuples(index=False)
    }
    point_count_by_landuse = (
        density_df.groupby("landuse", observed=False)["point_count"]
        .max()
        .to_dict()
    )

    for row_index, osm_type in enumerate(osm_types):
        for col_index, landuse in enumerate(LANDUSE_ORDER):
            ax = axes[row_index][col_index]
            ax.set_facecolor("#fbfbfb")

            selected_polygons = dissolved_polygons[dissolved_polygons["landuse_class"] == landuse]
            if not selected_polygons.empty:
                selected_polygons.plot(
                    ax=ax,
                    color=LANDUSE_COLOURS.get(landuse, "#6b7280"),
                    edgecolor=LANDUSE_COLOURS.get(landuse, "#6b7280"),
                    linewidth=0.75,
                    alpha=0.52,
                )

            selected_points = osm_points_gdf[
                (osm_points_gdf["osm_type"] == osm_type)
                & (osm_points_gdf["landuse"] == landuse)
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

            point_count, feature_count, density = density_lookup.get(
                (osm_type, landuse),
                (0, 0, None),
            )
            density_text = "NA" if pd.isna(density) else f"{density:.2f}"
            ax.text(
                0.02,
                0.98,
                f"Features: {feature_count}\nDensity: {density_text}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10.2,
                family="monospace",
                color="#111827",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.94,
                    "edgecolor": "#9ca3af",
                    "linewidth": 0.7,
                    "pad": 2.0,
                },
            )

            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d1d5db")
                spine.set_linewidth(0.7)

            if row_index == 0:
                header_point_count = int(point_count_by_landuse.get(landuse, 0))
                ax.set_title(
                    f"{display_label(landuse)}\nPoints: {header_point_count}",
                    fontsize=12.5,
                    pad=10,
                    fontweight="semibold",
                    bbox={
                        "facecolor": "#f3f4f6",
                        "edgecolor": "#e5e7eb",
                        "boxstyle": "round,pad=0.25",
                    },
                )
            if col_index == 0:
                ax.set_ylabel(
                    "\n".join(textwrap.wrap(display_label(osm_type), width=12)),
                    fontsize=13.5,
                    labelpad=24,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontweight="semibold",
                )

    fig.suptitle(
        f"{display_region_title(region.city, region.country)}\nOSM Features by Land Use",
        fontsize=17,
        y=0.975,
        fontweight="semibold",
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.995,
        bottom=0.04,
        top=0.875,
        wspace=0.035,
        hspace=0.001,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.05)
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
        osm_features = db.get_osm_features_by_region(region_id)
        if not osm_features:
            logger.warning(f"No OSM features found for region {region_id}")
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

        feature_lookup, point_rows = build_osm_points(osm_features)
        if not point_rows:
            logger.warning(f"No OSM features with valid coordinates found for region {region_id}")
            continue

        point_gdf_4326 = gpd.GeoDataFrame(point_rows, geometry="geometry", crs="EPSG:4326")
        point_gdf_3857 = point_gdf_4326.to_crs(epsg=3857)
        polygons_3857 = landuse_polygons.to_crs(epsg=3857).reset_index(drop=True)

        point_landuse = assign_points_to_landuse(point_gdf_3857, polygons_3857, args.nearby_meters)
        points_by_landuse = Counter(point_landuse.values())

        osm_types, features_by_type_landuse, grouped_points = aggregate_osm_by_landuse(
            feature_lookup,
            point_landuse,
        )
        if not osm_types:
            logger.warning(f"No matched OSM feature points found for region {region_id}")
            continue

        density_df = build_density_table(osm_types, points_by_landuse, features_by_type_landuse)
        osm_points_gdf = make_osm_geodataframe(grouped_points)

        output_dir = Path(f"maps/{region.country}/{region.city}/land_use_osm_grid")
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_file = output_dir / f"{region.id}_density.csv"
        density_df.to_csv(csv_file, index=False)
        logger.info(f"Saved density table to {csv_file}")

        png_file = output_dir / f"{region.id}_grid.png"
        save_grid_map(
            region,
            dissolved_polygons,
            osm_points_gdf,
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
