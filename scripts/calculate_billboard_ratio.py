import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.database import DatabaseManager


def parse_args():
    parser = argparse.ArgumentParser(
        prog="CalculateBillboardRatio",
        description="Calculate billboard detection ratio for one or more regions.",
    )
    parser.add_argument(
        "region_ids",
        nargs="+",
        type=str,
        help="List of region IDs.",
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

    if not region_ids:
        print("No region IDs provided.")
        return 1

    db = DatabaseManager()

    print("region_id,city,country,start_captured_at,end_captured_at,score,images,billboards,total_detections,billboards_per_image")
    for region_id in region_ids:
        region = db.get_region(region_id)
        if region is None:
            print(f"{region_id},NOT_FOUND,NOT_FOUND,,,0.000000,0,0,0,0.000000")
            continue

        images = db.get_images_by_region(region_id)
        image_count = len(images)
        detections = db.get_detections_by_region(region_id)
        total_detections = len(detections)
        billboard_count = sum(
            1 for detection in detections if (detection.label or "").strip().lower() == "billboard"
        )
        ratio = (billboard_count / image_count) if image_count > 0 else 0.0
        score = float(region.score) if region.score is not None else 0.0

        city = region.city or ""
        country = region.country or ""
        start_captured_at = region.start_captured_at.isoformat(
        ) if region.start_captured_at else ""
        end_captured_at = region.end_captured_at.isoformat() if region.end_captured_at else ""

        print(
            f"{region_id},{city},{country},{start_captured_at},{end_captured_at},{score:.6f},{image_count},{billboard_count},{total_detections},{ratio:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
