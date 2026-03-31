import argparse
import sys
from io import BytesIO
from pathlib import Path

import folium
from folium.plugins import HeatMap
from PIL import Image as PILImage

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import MapConfig, ScoreConfig
from src.database import DatabaseManager
from src.utils import RegionManager, setup_logger

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="HeatmapDetections",
        description="Generate detection and coverage heatmaps for regions.",
    )
    parser.add_argument(
        "region_ids",
        nargs="*",
        type=str,
        help="List of region IDs.",
    )
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="City name to resolve into one or more regions.",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Optional country filter when using --city.",
    )
    parser.add_argument(
        "--metric",
        choices=["count", "sum"],
        default="sum",
        help="Detection heatmap weighting metric.",
    )
    return parser.parse_args()


def normalize_region_ids(region_ids):
    normalized = []
    for token in region_ids:
        parts = [part.strip() for part in token.split(",") if part.strip()]
        normalized.extend(parts)
    return normalized


def normalize(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def find_regions_by_city(db, city, country=None):
    city_key = normalize(city)
    country_key = normalize(country) if country else None

    matches = []
    for region in db.get_all_regions():
        if normalize(region.city) != city_key:
            continue
        if country_key and normalize(region.country) != country_key:
            continue
        matches.append(region)

    matches.sort(
        key=lambda region: (
            normalize(region.country),
            normalize(region.city),
            region.start_captured_at or region.scanned_at,
            region.id,
        )
    )
    return matches


def resolve_regions(db, region_ids, city=None, country=None):
    regions = []

    for region_id in region_ids:
        region = db.get_region(region_id)
        if region is None:
            logger.warning(f"Region not found: {region_id}")
            continue
        regions.append(region)

    if city:
        city_regions = find_regions_by_city(db, city, country)
        if not city_regions:
            country_text = f" in {country}" if country else ""
            logger.warning(f"No regions found for city '{city}'{country_text}")
        regions.extend(city_regions)

    deduped_regions = []
    seen_region_ids = set()
    for region in regions:
        if region.id in seen_region_ids:
            continue
        seen_region_ids.add(region.id)
        deduped_regions.append(region)
    return deduped_regions


def build_detection_heatmap_points(detections, images_by_id, metric):
    severity_scores = ScoreConfig.SEVERITY_SCORES
    points = []

    for detection in detections:
        image = images_by_id.get(detection.image_id)
        if image is None:
            continue

        if metric == "count":
            weight = 1.0
        else:
            weight = severity_scores.get((detection.label or "").strip().lower(), 0.0)

        if weight <= 0:
            continue
        points.append([image.lat, image.lng, weight])

    return points


def build_coverage_heatmap_points(images):
    points = []
    images_by_id = {}

    for image in images:
        if image.lat is None or image.lng is None:
            continue
        images_by_id[image.id] = image
        points.append([image.lat, image.lng, 1.0])

    return images_by_id, points


def add_heatmap_layers(m, detection_points, coverage_points, metric):
    detection_layer = folium.FeatureGroup(
        name=f"Detection Heatmap ({metric})",
        show=True,
    )
    HeatMap(
        detection_points,
        min_opacity=0.35,
        radius=18,
        blur=24,
        max_zoom=16,
    ).add_to(detection_layer)
    detection_layer.add_to(m)

    coverage_layer = folium.FeatureGroup(
        name="Image Coverage Heatmap",
        show=False,
    )
    HeatMap(
        coverage_points,
        min_opacity=0.25,
        radius=14,
        blur=18,
        max_zoom=16,
        gradient={
            0.2: "#edf8fb",
            0.4: "#b3cde3",
            0.6: "#8c96c6",
            1.0: "#88419d",
        },
    ).add_to(coverage_layer)
    coverage_layer.add_to(m)


def save_heatmap_outputs(m, output_dir, region_id):
    html_output_file = output_dir / f"{region_id}.html"
    m.save(html_output_file)
    logger.info(f"Saved heatmap HTML for region {region_id} to {html_output_file}")

    png_output_file = output_dir / f"{region_id}.png"
    img_data = m._to_png(5)
    img = PILImage.open(BytesIO(img_data))
    img.save(png_output_file)
    logger.info(f"Saved heatmap PNG for region {region_id} to {png_output_file}")


def main():
    args = parse_args()
    region_ids = normalize_region_ids(args.region_ids)

    db = DatabaseManager()
    regions = resolve_regions(db, region_ids, args.city, args.country)

    if not regions:
        print("No regions provided. Use region IDs and/or --city [--country].")
        return 1

    found = 0
    for region in regions:
        found += 1
        region_id = region.id

        detections = db.get_detections_by_region(region_id)
        images = db.get_images_by_region(region_id)

        if not images:
            logger.warning(f"No images found for region {region_id}")
            continue

        images_by_id, coverage_points = build_coverage_heatmap_points(images)
        if not coverage_points:
            logger.warning(f"No valid coordinates for images in region {region_id}")
            continue

        detection_points = build_detection_heatmap_points(
            detections, images_by_id, args.metric
        )
        if not detection_points:
            logger.warning(
                f"No detections with valid image coordinates in region {region_id}"
            )
            continue

        _, centre = RegionManager.get_combined_bbox([region])
        m = folium.Map(
            location=[centre[1], centre[0]],
            zoom_start=MapConfig.ZOOM_START,
            tiles=MapConfig.get_tiles_url(),
            attr=MapConfig.TILES_ATTR,
        )

        add_heatmap_layers(m, detection_points, coverage_points, args.metric)
        folium.LayerControl().add_to(m)

        output_dir = Path(f"maps/{region.country}/{region.city}/heatmaps")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_heatmap_outputs(m, output_dir, region.id)

    return 0 if found > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
