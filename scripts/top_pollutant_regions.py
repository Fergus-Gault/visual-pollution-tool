import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager, Detection, Image  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        prog="TopPollutantRegions",
        description=(
            "Calculate the top regions by detections-per-image rate for each pollutant label."
        ),
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=300,
        help="Minimum number of images required for a region to be included.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top regions to keep for each pollutant type.",
    )
    parser.add_argument(
        "--output",
        default="data/top_pollutant_regions.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def format_dt(value):
    return value.isoformat() if value else ""


def main():
    args = parse_args()
    if args.min_images < 1:
        raise SystemExit("--min-images must be at least 1.")
    if args.top_n < 1:
        raise SystemExit("--top-n must be at least 1.")

    db = DatabaseManager()

    image_rows = (
        db.session.query(Image.region_id, func.count(Image.id))
        .group_by(Image.region_id)
        .having(func.count(Image.id) >= args.min_images)
        .all()
    )
    image_count_by_region = {
        region_id: int(image_count) for region_id, image_count in image_rows
    }

    if not image_count_by_region:
        print(f"No regions found with at least {args.min_images} images.")
        return 0

    regions_by_id = {
        region.id: region
        for region in db.get_all_regions()
        if (
            region.id in image_count_by_region
            and region.start_captured_at is None
            and region.end_captured_at is None
        )
    }

    image_count_by_region = {
        region_id: image_count
        for region_id, image_count in image_count_by_region.items()
        if region_id in regions_by_id
    }

    if not image_count_by_region:
        print(
            "No unconstrained regions found with at least "
            f"{args.min_images} images."
        )
        return 0

    detection_rows = (
        db.session.query(Image.region_id, Detection.label, func.count(Detection.id))
        .join(Detection, Detection.image_id == Image.id)
        .filter(Image.region_id.in_(list(image_count_by_region.keys())))
        .group_by(Image.region_id, Detection.label)
        .all()
    )

    top_rows_by_label = defaultdict(list)
    for region_id, raw_label, detection_count in detection_rows:
        label = (raw_label or "").strip()
        if not label:
            continue
        image_count = image_count_by_region.get(region_id, 0)
        if image_count <= 0:
            continue
        rate = float(detection_count) / float(image_count)
        region = regions_by_id.get(region_id)
        if region is None:
            continue
        top_rows_by_label[label].append(
            {
                "region_id": region_id,
                "city": region.city or "",
                "country": region.country or "",
                "images": image_count,
                "detections": int(detection_count),
                "rate": rate,
                "score": float(region.score) if region.score is not None else 0.0,
                "start_captured_at": format_dt(region.start_captured_at),
                "end_captured_at": format_dt(region.end_captured_at),
                "dense_scan": bool(region.dense_scan),
            }
        )

    if not top_rows_by_label:
        print("No pollutant detections found in regions meeting the image threshold.")
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pollutant",
                "rank",
                "region_id",
                "city",
                "country",
                "images",
                "detections",
                "detections_per_image",
                "score",
                "start_captured_at",
                "end_captured_at",
                "dense_scan",
            ]
        )

        for label in sorted(top_rows_by_label):
            rows = sorted(
                top_rows_by_label[label],
                key=lambda row: (
                    -row["rate"],
                    -row["detections"],
                    -row["images"],
                    row["country"].casefold(),
                    row["city"].casefold(),
                    row["region_id"],
                ),
            )[: args.top_n]

            print(f"\n{label} (top {len(rows)})")
            for rank, row in enumerate(rows, start=1):
                print(
                    f"{rank}. {row['city']}, {row['country']} | "
                    f"region_id={row['region_id']} | "
                    f"images={row['images']} | "
                    f"detections={row['detections']} | "
                    f"rate={row['rate']:.6f}"
                )
                writer.writerow(
                    [
                        label,
                        rank,
                        row["region_id"],
                        row["city"],
                        row["country"],
                        row["images"],
                        row["detections"],
                        f"{row['rate']:.6f}",
                        f"{row['score']:.6f}",
                        row["start_captured_at"],
                        row["end_captured_at"],
                        str(row["dense_scan"]).lower(),
                    ]
                )

    print(f"\nSaved results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
