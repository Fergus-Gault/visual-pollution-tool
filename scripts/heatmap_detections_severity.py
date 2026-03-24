import argparse
import math
import sys
from pathlib import Path

import branca.colormap as cm
import folium

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import setup_logger, RegionManager
from src.config import MapConfig, ScoreConfig
from src.database import DatabaseManager

logger = setup_logger(__name__)
MIN_IMAGES_PER_CELL = 4
ARCGIS_TILES_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ARCGIS_TILES_ATTR = "Tiles © Esri"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GridDetections",
        description="Generate grid-based detection hotspot and coverage layers for regions.",
    )
    parser.add_argument("region_ids", nargs="+", type=str,
                        help="List of region IDs.")
    parser.add_argument(
        "--metric",
        choices=["count", "sum"],
        default="sum",
        help="Detection aggregation metric per cell.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.002,
        help="Grid cell size in degrees.",
    )
    return parser.parse_args()


def normalize_region_ids(region_ids):
    normalized = []
    for token in region_ids:
        parts = [part.strip() for part in token.split(",") if part.strip()]
        normalized.extend(parts)
    return normalized


def get_cell_key(lat, lng, cell_size):
    return (math.floor(lat / cell_size), math.floor(lng / cell_size))


def get_cell_bounds(cell_key, cell_size):
    lat0 = cell_key[0] * cell_size
    lng0 = cell_key[1] * cell_size
    return [[lat0, lng0], [lat0 + cell_size, lng0 + cell_size]]


def main():
    args = parse_args()
    region_ids = normalize_region_ids(args.region_ids)
    cell_size = args.cell_size

    if not region_ids:
        print("No region IDs provided.")
        return 1

    if cell_size <= 0:
        print("Cell size must be > 0.")
        return 1

    db = DatabaseManager()

    found = 0
    for region_id in region_ids:
        region = db.get_region(region_id)
        if region is None:
            logger.warning(f"Region not found: {region_id}")
            continue
        found += 1

        detections = db.get_detections_by_region(region_id)
        images = db.get_images_by_region(region_id)

        if not images:
            logger.warning(f"No images found for region {region_id}")
            continue

        severity_scores = ScoreConfig.SEVERITY_SCORES

        images_by_id = {}
        image_count_by_cell = {}
        for image in images:
            if image.lat is None or image.lng is None:
                continue

            images_by_id[image.id] = image
            cell_key = get_cell_key(image.lat, image.lng, cell_size)
            image_count_by_cell[cell_key] = image_count_by_cell.get(
                cell_key, 0) + 1

        if not image_count_by_cell:
            logger.warning(
                f"No valid coordinates for images in region {region_id}")
            continue

        eligible_image_count_by_cell = {
            cell_key: count
            for cell_key, count in image_count_by_cell.items()
            if count > MIN_IMAGES_PER_CELL
        }

        if not eligible_image_count_by_cell:
            logger.warning(
                f"No cells with more than {MIN_IMAGES_PER_CELL} images in region {region_id}")
            continue

        detection_value_by_cell = {}
        detection_count_by_cell = {}
        for detection in detections:
            image = images_by_id.get(detection.image_id)
            if image is None:
                continue

            cell_key = get_cell_key(image.lat, image.lng, cell_size)
            if cell_key not in eligible_image_count_by_cell:
                continue
            if args.metric == "count":
                value = 1.0
            else:
                value = severity_scores.get(
                    (detection.label or "").strip().lower(), 0.0)

            detection_value_by_cell[cell_key] = detection_value_by_cell.get(
                cell_key, 0.0) + value
            detection_count_by_cell[cell_key] = detection_count_by_cell.get(
                cell_key, 0) + 1

        normalized_score_by_cell = {}
        for cell_key, raw_value in detection_value_by_cell.items():
            image_count = eligible_image_count_by_cell.get(cell_key, 0)
            if image_count > 0:
                normalized_score_by_cell[cell_key] = raw_value / image_count

        if not normalized_score_by_cell:
            logger.warning(
                f"No detections with valid image coordinates in region {region_id}")
            continue

        _, centre = RegionManager.get_combined_bbox([region])

        m = folium.Map(
            location=[centre[1], centre[0]],
            zoom_start=MapConfig.ZOOM_START
        )

        folium.TileLayer(
            tiles=ARCGIS_TILES_URL,
            attr=ARCGIS_TILES_ATTR,
            overlay=False,
            name='ArcGIS World Imagery'
        ).add_to(m)

        log_normalized_score_by_cell = {
            cell_key: math.log1p(score)
            for cell_key, score in normalized_score_by_cell.items()
        }

        score_values = list(log_normalized_score_by_cell.values())
        score_min = min(score_values)
        score_max = max(score_values)
        if score_min == score_max:
            score_max = score_min + 1e-9

        score_colormap = cm.LinearColormap(
            colors=["#2b83ba", "#abdda4", "#fdae61", "#d7191c"],
            vmin=score_min,
            vmax=score_max,
            caption=f"Log-normalized detection score per cell (log1p({args.metric} / image_count))",
        )

        hotspot_layer = folium.FeatureGroup(
            name='Detection Hotspots', show=True)
        for cell_key, normalized_score in normalized_score_by_cell.items():
            log_normalized_score = log_normalized_score_by_cell[cell_key]
            bounds = get_cell_bounds(cell_key, cell_size)
            raw_value = detection_value_by_cell.get(cell_key, 0.0)
            detection_count = detection_count_by_cell.get(cell_key, 0)
            image_count = eligible_image_count_by_cell.get(cell_key, 0)
            folium.Rectangle(
                bounds=bounds,
                color=score_colormap(log_normalized_score),
                fill=True,
                fill_color=score_colormap(log_normalized_score),
                fill_opacity=0.65,
                weight=0.4,
                tooltip=(
                    f"norm_score={normalized_score:.4f}; log_norm={log_normalized_score:.4f}; raw={raw_value:.4f}; "
                    f"detections={detection_count}; images={image_count}"
                ),
            ).add_to(hotspot_layer)
        hotspot_layer.add_to(m)

        coverage_values = list(eligible_image_count_by_cell.values())
        coverage_min = min(coverage_values)
        coverage_max = max(coverage_values)
        if coverage_min == coverage_max:
            coverage_max = coverage_min + 1e-9

        coverage_colormap = cm.LinearColormap(
            colors=["#edf8fb", "#b3cde3", "#8c96c6", "#88419d"],
            vmin=coverage_min,
            vmax=coverage_max,
            caption="Image coverage per cell",
        )

        coverage_layer = folium.FeatureGroup(name='Coverage', show=False)
        for cell_key, image_count in eligible_image_count_by_cell.items():
            bounds = get_cell_bounds(cell_key, cell_size)
            folium.Rectangle(
                bounds=bounds,
                color=coverage_colormap(image_count),
                fill=True,
                fill_color=coverage_colormap(image_count),
                fill_opacity=0.35,
                weight=0.25,
                tooltip=f"images={image_count}",
            ).add_to(coverage_layer)
        coverage_layer.add_to(m)

        score_colormap.add_to(m)
        coverage_colormap.add_to(m)

        folium.LayerControl().add_to(m)

        output_dir = Path(f"maps/{region.country}/{region.city}/heatmaps")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{region.id}.html"
        m.save(output_file)
        logger.info(f"Saved grid map for region {region_id} to {output_file}")

    return 0 if found > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
