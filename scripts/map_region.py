import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.database import DatabaseManager
from src.mapping import Mapper
from src.utils import setup_logger

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("region_ids", nargs="*", type=str)
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
        "--file-type",
        type=str,
        default="html",
        choices=["html", "png"],
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching regions without generating maps.",
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


def describe_region(region):
    parts = [f"region_id={region.id}", region.city or "Unknown city", region.country or "Unknown country"]
    if region.start_captured_at or region.end_captured_at:
        start = region.start_captured_at.isoformat() if region.start_captured_at else "?"
        end = region.end_captured_at.isoformat() if region.end_captured_at else "?"
        parts.append(f"capture_window={start}..{end}")
    if region.dense_scan:
        parts.append("dense_scan=True")
    return ", ".join(parts)


def main():
    args = parse_args()
    region_ids = normalize_region_ids(args.region_ids)
    db = DatabaseManager()
    mapper = Mapper(db)
    regions = []

    for region_id in region_ids:
        region = db.get_region(region_id)
        if region is None:
            logger.warning(f"Region not found: {region_id}")
            continue
        regions.append(region)

    if args.city:
        city_regions = find_regions_by_city(db, args.city, args.country)
        if not city_regions:
            country_text = f" in {args.country}" if args.country else ""
            logger.warning(f"No regions found for city '{args.city}'{country_text}")
        regions.extend(city_regions)

    deduped_regions = []
    seen_region_ids = set()
    for region in regions:
        if region.id in seen_region_ids:
            continue
        seen_region_ids.add(region.id)
        deduped_regions.append(region)
    regions = deduped_regions

    if not regions:
        print("No regions provided. Use region IDs and/or --city [--country].")
        return 1

    for region in regions:
        logger.info(f" - {describe_region(region)}")

    if args.list:
        return 0

    found = 0
    for region in regions:
        found += 1

        images_map = mapper.map_region_images(region)
        if images_map is not None:
            mapper.save(images_map, region, args.file_type,
                        map_type="region_images")

        detections_map = mapper.map_region_detections(region)
        if detections_map is not None:
            mapper.save(detections_map, region, args.file_type,
                        map_type="region_detections")

    return 0 if found > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
