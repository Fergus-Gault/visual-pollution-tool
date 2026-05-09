import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import folium
from branca.colormap import LinearColormap
from folium.plugins import HeatMap
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config, MapConfig, ScoreConfig  # noqa: E402
from src.database import DatabaseManager, Detection, Image  # noqa: E402
from src.utils import RegionManager  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        prog="MapCountryPollution",
        description=(
            "Create a ranked export, a region score map, and a hotspot heatmap "
            "for a country-wide scan."
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
        help="How many top regions to include in the ranked CSV summary.",
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
        help="Directory for generated HTML maps and CSV summaries.",
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


def build_region_popup(row):
    nearest_city = row["nearest_city"] or "Unknown"
    return (
        f"<b>{row['city'] or 'Unknown region'}</b><br>"
        f"Country: {row['country'] or 'Unknown'}<br>"
        f"Closest city: {nearest_city}<br>"
        f"Region ID: {row['region_id']}<br>"
        f"Score: {row['score']:.6f}<br>"
        f"Images: {row['images']}<br>"
        f"Detections: {row['detections']}<br>"
        f"Detections / image: {row['detections_per_image']:.6f}"
    )


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
        heatmap_rows.append([point[0], point[1], weight])
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


def point_in_region(city_row, region):
    return (
        float(region.min_lng) <= city_row["lng"] <= float(region.max_lng)
        and float(region.min_lat) <= city_row["lat"] <= float(region.max_lat)
    )


def distance_to_region_centre(city_row, region):
    centre_lng = (float(region.min_lng) + float(region.max_lng)) / 2.0
    centre_lat = (float(region.min_lat) + float(region.max_lat)) / 2.0
    return math.hypot(city_row["lng"] - centre_lng, city_row["lat"] - centre_lat)


def format_nearest_city(city_row):
    if city_row is None:
        return ""
    if city_row["admin_name"]:
        return f"{city_row['city']} ({city_row['admin_name']})"
    return city_row["city"]


def get_nearest_city_by_region(regions, country, cities_path):
    country_cities = load_worldcities(cities_path, country)
    nearest_by_region_id = {}
    for region in regions:
        candidates = [city_row for city_row in country_cities if point_in_region(city_row, region)]
        if not candidates:
            nearest_by_region_id[region.id] = None
            continue
        candidates.sort(
            key=lambda city_row: (
                distance_to_region_centre(city_row, region),
                -city_row["population"],
                city_row["city"].casefold(),
            )
        )
        nearest_by_region_id[region.id] = candidates[0]
    return nearest_by_region_id


def build_region_rows(regions, image_counts, detection_counts, nearest_city_by_region, min_images):
    rows = []
    for region in regions:
        image_count = image_counts.get(region.id, 0)
        if image_count < min_images:
            continue
        score = float(region.score) if region.score is not None else 0.0
        detection_count = detection_counts.get(region.id, 0)
        rows.append(
            {
                "region_id": region.id,
                "city": region.city or "",
                "country": region.country or "",
                "score": score,
                "images": image_count,
                "detections": detection_count,
                "nearest_city": format_nearest_city(nearest_city_by_region.get(region.id)),
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


def save_ranked_csv(rows, output_path, top_n):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "region_id",
                "city",
                "country",
                "nearest_city",
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
        for rank, row in enumerate(rows[:top_n], start=1):
            writer.writerow(
                [
                    rank,
                    row["region_id"],
                    row["city"],
                    row["country"],
                    row["nearest_city"],
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


def add_region_rectangles(map_obj, rows):
    positive_scores = [row["score"] for row in rows if row["score"] > 0.0]
    score_scale = None
    if positive_scores:
        min_score = min(positive_scores)
        max_score = max(positive_scores)
        if min_score == max_score:
            max_score = min_score + 1e-9
        score_scale = LinearColormap(
            colors=["#1a9850", "#fee08b", "#d73027"],
            vmin=min_score,
            vmax=max_score,
        )
        score_scale.caption = "Pollution score (green = low, red = high)"
        score_scale.add_to(map_obj)

    for row in rows:
        bounds = [
            [row["min_lat"], row["min_lng"]],
            [row["max_lat"], row["max_lng"]],
        ]
        if score_scale is None:
            fill_color = "#9aa0a6"
        elif row["score"] <= 0.0:
            fill_color = "#1a9850"
        else:
            fill_color = score_scale(row["score"])

        folium.Rectangle(
            bounds=bounds,
            color="#111111",
            weight=0.9,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.82,
            popup=folium.Popup(build_region_popup(row), max_width=320),
            tooltip=(
                f"{row['city'] or 'Unknown region'} | score={row['score']:.4f} | "
                f"images={row['images']}"
            ),
        ).add_to(map_obj)


def create_region_score_map(country, rows, output_path):
    regions = [row["region"] for row in rows]
    _, centre = RegionManager.get_combined_bbox(regions)
    map_obj = folium.Map(
        location=[centre[1], centre[0]],
        zoom_start=6,
        tiles=MapConfig.get_tiles_url(),
        attr=MapConfig.get_tiles_attr(),
    )
    add_region_rectangles(map_obj, rows)
    map_obj.fit_bounds(
        [
            [min(row["min_lat"] for row in rows), min(row["min_lng"]
                                                      for row in rows)],
            [max(row["max_lat"] for row in rows), max(row["max_lng"]
                                                      for row in rows)],
        ]
    )

    title_html = (
        f"<h3 align='center' style='font-size:16px'><b>{country} pollution region scores</b></h3>"
    )
    map_obj.get_root().html.add_child(folium.Element(title_html))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(output_path)


def create_heatmap(country, rows, heatmap_points, output_path):
    regions = [row["region"] for row in rows]
    _, centre = RegionManager.get_combined_bbox(regions)
    map_obj = folium.Map(
        location=[centre[1], centre[0]],
        zoom_start=6,
        tiles=MapConfig.get_tiles_url(),
        attr=MapConfig.get_tiles_attr(),
    )

    HeatMap(
        heatmap_points,
        radius=10,
        blur=8,
        min_opacity=0.35,
        max_zoom=10,
    ).add_to(map_obj)
    map_obj.fit_bounds(
        [
            [min(row["min_lat"] for row in rows), min(row["min_lng"]
                                                      for row in rows)],
            [max(row["max_lat"] for row in rows), max(row["max_lng"]
                                                      for row in rows)],
        ]
    )

    title_html = (
        f"<h3 align='center' style='font-size:16px'><b>{country} pollution hotspots</b></h3>"
    )
    map_obj.get_root().html.add_child(folium.Element(title_html))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(output_path)


def main():
    args = parse_args()
    if args.min_images < 1:
        raise SystemExit("--min-images must be at least 1.")
    if args.top_n < 1:
        raise SystemExit("--top-n must be at least 1.")

    db = DatabaseManager()
    regions = get_country_regions(db, args.country, args.all_regions)
    if not regions:
        scope = "all regions" if args.all_regions else "country grid regions"
        raise SystemExit(
            f"No {scope} found for country '{args.country}'."
        )

    region_ids = [region.id for region in regions]
    image_counts = get_region_image_counts(db, region_ids)
    detection_counts = get_region_detection_counts(db, region_ids)
    nearest_city_by_region = get_nearest_city_by_region(
        regions,
        args.country,
        Config.DEFAULT_CSV,
    )
    rows = build_region_rows(
        regions,
        image_counts,
        detection_counts,
        nearest_city_by_region,
        args.min_images,
    )

    if not rows:
        raise SystemExit(
            f"No regions found for '{args.country}' with at least {args.min_images} images."
        )

    output_dir = Path(args.output_dir)
    country_slug = normalize(args.country).replace(" ", "_")
    csv_path = output_dir / f"{country_slug}_top_regions.csv"
    regions_map_path = output_dir / f"{country_slug}_region_scores.html"
    heatmap_path = output_dir / f"{country_slug}_pollution_heatmap.html"

    save_ranked_csv(rows, csv_path, args.top_n)
    create_region_score_map(args.country, rows, regions_map_path)

    heatmap_points = get_heatmap_points(db, [row["region_id"] for row in rows])
    if not heatmap_points:
        raise SystemExit(
            "Ranked CSV and region score map were created, but no weighted detection "
            "points were available for the heatmap."
        )
    create_heatmap(args.country, rows, heatmap_points, heatmap_path)

    print(f"Saved ranked regions to {csv_path}")
    print(f"Saved region score map to {regions_map_path}")
    print(f"Saved hotspot heatmap to {heatmap_path}")


if __name__ == "__main__":
    raise SystemExit(main())
