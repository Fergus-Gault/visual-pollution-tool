import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.database import DatabaseManager
from src.mapping import Mapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("region_ids", nargs="+", type=str)
    parser.add_argument(
        "--file-type",
        type=str,
        default="html",
        choices=["html", "png"],
    )
    return parser.parse_args()


def normalize_region_ids(region_ids):
    normalized = []
    for token in region_ids:
        parts = [part.strip() for part in token.split(",") if part.strip()]
        normalized.extend(parts)
    return normalized


def main():
    args = parse_args()
    region_ids = normalize_region_ids(args.region_ids)
    db = DatabaseManager()
    mapper = Mapper(db)

    if not region_ids:
        print("No region IDs provided.")
        return 1

    found = 0
    for region_id in region_ids:
        region = db.get_region(region_id)
        if region is None:
            print(f"Region not found: {region_id}")
            continue
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
