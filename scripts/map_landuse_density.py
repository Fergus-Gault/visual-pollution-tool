import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import branca.colormap as cm
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, box

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import MapConfig
from src.database import DatabaseManager
from src.utils import RegionManager, setup_logger

logger = setup_logger(__name__)

ARCGIS_TILES_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ARCGIS_TILES_ATTR = "Tiles (C) Esri"

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
        prog="LandUseDensity",
        description="Map per-class detections-to-images density by nearby OSM land use.",
    )
    parser.add_argument("region_ids", nargs="+", type=str,
                        help="List of region IDs.")
    parser.add_argument(
        "--nearby-meters",
        type=float,
        default=150.0,
        help="Maximum distance in meters to match a nearby land-use polygon when point is not inside one.",
    )
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Include unmatched points as category 'unmatched' in CSV output.",
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
    region_polygon = box(region.min_lng, region.min_lat,
                         region.max_lng, region.max_lat)
    tags = {"landuse": True, "leisure": True}
    try:
        gdf = fetch_features_from_polygon(region_polygon, tags)
    except Exception as exc:
        logger.warning(
            f"Failed to fetch OSM land-use features for region {region.id}: {exc}")
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

    gdf = gdf[["landuse_class", "geometry"]].explode(
        index_parts=False).reset_index(drop=True)
    return gdf


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
    total_images = len(image_points_3857)
    logger.info(
        f"Land-use assignment start: images={total_images}, polygons={len(polygons_view)}")

    inside = gpd.sjoin(
        image_points_3857,
        polygons_view,
        how="left",
        predicate="within",
    )
    inside = inside[inside["landuse_class"].notna()].copy()
    if not inside.empty:
        grouped_inside = inside.groupby(
            "image_id")["landuse_class"].apply(list)
        for image_id, categories in grouped_inside.items():
            resolved = resolve_landuse_category(categories)
            if resolved is not None:
                image_landuse[image_id] = resolved
    logger.info(
        f"Land-use assignment after within-join: matched={len(image_landuse)}/{total_images}")

    unmatched = image_points_3857[~image_points_3857["image_id"].isin(
        image_landuse.keys())].copy()
    if unmatched.empty:
        logger.info(
            "Land-use assignment complete: all images matched via within-join")
        return image_landuse

    logger.info(
        f"Land-use assignment nearest-join for unmatched images: count={len(unmatched)}")

    nearest = gpd.sjoin_nearest(
        unmatched,
        polygons_view,
        how="left",
        max_distance=nearby_meters,
        distance_col="distance_m",
    )
    nearest = nearest[nearest["landuse_class"].notna()].copy()
    if nearest.empty:
        logger.info(
            f"Land-use assignment complete: matched={len(image_landuse)}/{total_images}")
        return image_landuse

    nearest.sort_values(["image_id", "distance_m"], inplace=True)
    for image_id, group in nearest.groupby("image_id"):
        min_distance = group["distance_m"].min()
        categories = group[group["distance_m"] ==
                           min_distance]["landuse_class"].tolist()
        resolved = resolve_landuse_category(categories)
        if resolved is not None:
            image_landuse[image_id] = resolved

    logger.info(
        f"Land-use assignment complete: matched={len(image_landuse)}/{total_images}")

    return image_landuse


def build_landuse_density_table(labels, images_by_landuse, detections_by_label_landuse):
    landuse_values = LANDUSE_ORDER.copy()
    rows = []
    for label in labels:
        for landuse in landuse_values:
            image_count = int(images_by_landuse.get(landuse, 0))
            detection_count = int(
                detections_by_label_landuse.get((label, landuse), 0))
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


def make_label_map(region, dissolved_polygons, label, label_density_df):
    value_lookup = {
        row.landuse: (
            row.detections_per_image,
            int(row.image_count),
            int(row.detection_count),
        )
        for row in label_density_df.itertuples(index=False)
    }

    valid = label_density_df[label_density_df["detections_per_image"].notna()]
    if valid.empty:
        colormap = cm.LinearColormap(
            colors=["#f7f7f7", "#f7f7f7"],
            vmin=0,
            vmax=1,
            caption=f"{label}: detections per image",
        )
    else:
        min_value = float(valid["detections_per_image"].min())
        max_value = float(valid["detections_per_image"].max())
        if math.isclose(min_value, max_value):
            max_value = min_value + 1e-9
        colormap = cm.LinearColormap(
            colors=["#fee8c8", "#fdbb84", "#e34a33"],
            vmin=min_value,
            vmax=max_value,
            caption=f"{label}: detections per image",
        )

    _, centre = RegionManager.get_combined_bbox([region])
    m = folium.Map(location=[centre[1], centre[0]],
                   zoom_start=MapConfig.ZOOM_START)

    folium.TileLayer(
        tiles=ARCGIS_TILES_URL,
        attr=ARCGIS_TILES_ATTR,
        overlay=False,
        name="ArcGIS World Imagery",
    ).add_to(m)

    for poly in dissolved_polygons.itertuples(index=False):
        landuse = poly.landuse_class
        values = value_lookup.get(landuse)
        if values is None:
            detections_per_image = None
            image_count = 0
            detection_count = 0
        else:
            detections_per_image, image_count, detection_count = values

        colour = "#bdbdbd" if pd.isna(
            detections_per_image) else colormap(float(detections_per_image))

        tooltip_text = f"Land-use: {landuse}"
        if not pd.isna(detections_per_image):
            tooltip_text += f"\nImages: {image_count}\nDetections: {detection_count}\nDensity: {detections_per_image:.2f}"

        folium.GeoJson(
            data=poly.geometry.__geo_interface__,
            style_function=lambda _feature, c=colour: {
                "color": c,
                "fillColor": c,
                "fillOpacity": 0.60,
                "weight": 0.6,
            },
            tooltip=folium.Tooltip(tooltip_text),
        ).add_to(m)

    colormap.add_to(m)

    folium.LayerControl().add_to(m)
    return m


def save_label_png_map(dissolved_polygons, label, label_density_df, output_file):
    value_lookup = {
        row.landuse: row.detections_per_image
        for row in label_density_df.itertuples(index=False)
    }
    plot_gdf = dissolved_polygons.copy()
    plot_gdf["density"] = plot_gdf["landuse_class"].map(value_lookup)

    valid = plot_gdf[plot_gdf["density"].notna()]
    if valid.empty:
        vmin = 0.0
        vmax = 1.0
    else:
        vmin = float(valid["density"].min())
        vmax = float(valid["density"].max())
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-9

    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    plot_gdf.plot(
        column="density",
        cmap="viridis",
        linewidth=0.05,
        edgecolor="#f2f2f2",
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        legend=True,
        legend_kwds={
            "label": f"{label}: detections per image",
            "shrink": 0.72,
            "fraction": 0.04,
            "pad": 0.02,
        },
        missing_kwds={"color": "#9e9e9e", "label": "No data"},
    )
    ax.set_facecolor("#efefef")
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_class_landuse_graph(label, label_density_df, output_file):
    ordered = label_density_df.set_index(
        "landuse").reindex(LANDUSE_ORDER).reset_index()
    scores = ordered["detections_per_image"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.bar(ordered["landuse"], scores, color="#1f78b4")
    ax.set_title(f"{label}: detections per image by land use")
    ax.set_xlabel("Land use")
    ax.set_ylabel("Detections per image")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def save_landuse_class_graph(landuse, landuse_density_df, output_file):
    ordered = landuse_density_df.sort_values("label").reset_index(drop=True)
    scores = ordered["detections_per_image"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
    ax.bar(ordered["label"], scores, color="#33a02c")
    ax.set_title(f"{landuse}: detections per image by class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Detections per image")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
    plt.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
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
    logger.info(
        f"Starting land-use density mapping for {len(region_ids)} region(s); nearby_meters={args.nearby_meters}, include_unmatched={args.include_unmatched}")
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
        logger.info(
            f"Loaded region data: images={len(images)}, detections={len(detections)}")
        if not images:
            logger.warning(f"No images found for region {region_id}")
            continue

        osm_start = time.perf_counter()
        logger.info("Fetching OSM land-use polygons")
        landuse_polygons = fetch_landuse_polygons(region)
        logger.info(
            f"Fetched OSM land-use polygons: count={len(landuse_polygons)} in {time.perf_counter() - osm_start:.2f}s")
        if landuse_polygons.empty:
            logger.warning(
                f"No OSM land-use polygons found for region {region_id}")
            continue

        prep_start = time.perf_counter()
        polygons_4326 = landuse_polygons.reset_index(drop=True).copy()
        polygons_3857 = polygons_4326.to_crs(epsg=3857).reset_index(drop=True)

        dissolved_polygons = polygons_4326.dissolve(
            by="landuse_class", as_index=False)
        if dissolved_polygons.crs is None:
            dissolved_polygons = dissolved_polygons.set_crs(epsg=4326)
        else:
            dissolved_polygons = dissolved_polygons.to_crs(epsg=4326)

        images_by_id = {}

        image_records = []
        for image in images:
            if image.lng is None or image.lat is None:
                continue
            images_by_id[image.id] = image
            image_records.append({
                "image_id": image.id,
                "geometry": Point(image.lng, image.lat),
            })

        logger.info(
            f"Prepared image points with coordinates: count={len(image_records)}")

        if not image_records:
            logger.warning(
                f"No images with valid coordinates found for region {region_id}")
            continue

        image_points_4326 = gpd.GeoDataFrame(
            image_records,
            geometry="geometry",
            crs="EPSG:4326",
        )
        image_points_3857 = image_points_4326.to_crs(epsg=3857)

        image_landuse = assign_image_to_landuse(
            image_points_3857,
            polygons_3857,
            args.nearby_meters,
        )
        logger.info(
            f"Spatial assignment completed in {time.perf_counter() - prep_start:.2f}s")

        images_by_landuse = Counter(image_landuse.values())

        detections_by_label_landuse = Counter()
        labels = set()
        total_detections = len(detections)
        progress_step = 50000

        for idx, detection in enumerate(detections, start=1):
            image = images_by_id.get(detection.image_id)
            if image is None:
                continue
            label = normalize_label(detection.label)
            labels.add(label)

            landuse = image_landuse.get(image.id)
            if landuse is None:
                continue
            detections_by_label_landuse[(label, landuse)] += 1

            if idx % progress_step == 0:
                logger.info(
                    f"Detection aggregation progress: {idx}/{total_detections}")

        logger.info(
            f"Detection aggregation complete: classes={len(labels)}, matched_pairs={len(detections_by_label_landuse)}")

        if not labels:
            logger.warning(f"No detections found for region {region_id}")
            continue

        landuse_density_df = build_landuse_density_table(
            sorted(labels),
            images_by_landuse,
            detections_by_label_landuse,
        )
        logger.info(
            f"Built landuse density table rows={len(landuse_density_df)}")

        output_dir = Path(
            f"maps/{region.country}/{region.city}/land_use_density")
        output_dir.mkdir(parents=True, exist_ok=True)
        class_graph_dir = output_dir / "graphs" / "by_class"
        landuse_graph_dir = output_dir / "graphs" / "by_landuse"
        class_graph_dir.mkdir(parents=True, exist_ok=True)
        landuse_graph_dir.mkdir(parents=True, exist_ok=True)

        csv_file = output_dir / f"{region.id}_density.csv"
        landuse_density_df.to_csv(csv_file, index=False)
        logger.info(f"Saved density table to {csv_file}")

        dissolved_polygons = landuse_polygons.dissolve(
            by="landuse_class", as_index=False)
        if dissolved_polygons.crs is None:
            dissolved_polygons = dissolved_polygons.set_crs(epsg=4326)
        else:
            dissolved_polygons = dissolved_polygons.to_crs(epsg=4326)

        for label in sorted(labels):
            logger.info(f"Building and saving map for label '{label}'")
            label_df = landuse_density_df[landuse_density_df["label"] == label]
            m = make_label_map(region, dissolved_polygons, label, label_df)
            safe_label = safe_name(label)
            html_map_file = output_dir / f"{region.id}_{safe_label}.html"
            m.save(html_map_file)
            logger.info(f"Saved density map to {html_map_file}")

            png_map_file = output_dir / f"{region.id}_{safe_label}.png"
            save_label_png_map(dissolved_polygons, label,
                               label_df, png_map_file)
            logger.info(f"Saved density PNG map to {png_map_file}")

            class_graph_file = class_graph_dir / \
                f"{region.id}_{safe_label}_by_landuse.png"
            save_class_landuse_graph(label, label_df, class_graph_file)
            logger.info(f"Saved class graph to {class_graph_file}")

        logger.info("Building aggregate map for all detections")
        aggregate_rows = []
        for landuse in LANDUSE_ORDER:
            counts = landuse_density_df[landuse_density_df["landuse"] == landuse]
            if counts.empty:
                image_count = 0
                detection_count = 0
            else:
                image_count = int(counts["image_count"].iloc[0])
                detection_count = int(counts["detection_count"].sum())
            detections_per_image = None
            if image_count > 0:
                detections_per_image = detection_count / image_count
            aggregate_rows.append({
                "label": "all_detections",
                "landuse": landuse,
                "image_count": image_count,
                "detection_count": detection_count,
                "detections_per_image": detections_per_image,
            })
        aggregate_df = pd.DataFrame(aggregate_rows)
        m = make_label_map(region, dissolved_polygons,
                           "all detections", aggregate_df)
        aggregate_html_map_file = output_dir / \
            f"{region.id}_all_detections.html"
        m.save(aggregate_html_map_file)
        logger.info(
            f"Saved aggregate density map to {aggregate_html_map_file}")

        aggregate_png_map_file = output_dir / f"{region.id}_all_detections.png"
        save_label_png_map(dissolved_polygons, "all detections",
                           aggregate_df, aggregate_png_map_file)
        logger.info(
            f"Saved aggregate density PNG map to {aggregate_png_map_file}")

        aggregate_graph_file = class_graph_dir / \
            f"{region.id}_all_detections_by_landuse.png"
        save_class_landuse_graph(
            "all detections", aggregate_df, aggregate_graph_file)
        logger.info(f"Saved aggregate class graph to {aggregate_graph_file}")

        for landuse in LANDUSE_ORDER:
            landuse_df = landuse_density_df[landuse_density_df["landuse"] == landuse]
            landuse_graph_file = landuse_graph_dir / \
                f"{region.id}_{safe_name(landuse)}_by_class.png"
            save_landuse_class_graph(landuse, landuse_df, landuse_graph_file)
            logger.info(f"Saved land-use graph to {landuse_graph_file}")

        logger.info(
            f"Finished region {region_id} in {time.perf_counter() - region_start:.2f}s")

    return 0 if found > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
