import argparse
import csv
from pathlib import Path
import sys

from sqlalchemy import func
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from src.database import DatabaseManager, Detection, Image, Region
from src.pipeline import Scorer
from src.utils import setup_logger


logger = setup_logger(__name__)


def normalize(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def parse_args():
    parser = argparse.ArgumentParser(
        prog="ScoreImages",
        description="Calculate image-level pollution scores and persist them to images.score.",
    )
    parser.add_argument(
        "--image-id",
        action="append",
        dest="image_ids",
        default=None,
        help="Optional image ID to score. Pass multiple times to score several images.",
    )
    parser.add_argument(
        "--region-id",
        action="append",
        dest="region_ids",
        default=None,
        help="Optional region ID filter. Pass multiple times to include several regions.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Optional city filter applied through the parent region.",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Optional country filter applied through the parent region.",
    )
    parser.add_argument(
        "--output",
        default="data/image_scores.csv",
        help="CSV path for the scored image export.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate scores and export the CSV without updating images.score in the database.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="How many images to process per batch.",
    )
    return parser.parse_args()


def build_image_query(db, args):
    query = db.session.query(Image).outerjoin(Region, Image.region_id == Region.id)

    if args.image_ids:
        query = query.filter(Image.id.in_(args.image_ids))
    if args.region_ids:
        query = query.filter(Image.region_id.in_(args.region_ids))
    if args.city:
        query = query.filter(func.lower(func.coalesce(Region.city, "")) == normalize(args.city))
    if args.country:
        query = query.filter(func.lower(func.coalesce(Region.country, "")) == normalize(args.country))

    return query


def iter_image_batches(query, batch_size):
    last_image_id = None
    while True:
        batch_query = query
        if last_image_id is not None:
            batch_query = batch_query.filter(Image.id > last_image_id)
        batch = batch_query.limit(batch_size).all()
        if not batch:
            break
        yield batch
        last_image_id = batch[-1].id


def fetch_detection_counts(db, image_ids):
    if not image_ids:
        return {}

    rows = (
        db.session.query(Detection.image_id, func.count(Detection.id))
        .filter(Detection.image_id.in_(image_ids))
        .group_by(Detection.image_id)
        .all()
    )
    return {image_id: int(total) for image_id, total in rows}


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")

    db = DatabaseManager()
    scorer = Scorer(db)

    image_query = build_image_query(db, args)
    total_images = image_query.count()
    if total_images == 0:
        logger.warning("No images matched the requested filters.")
        raise SystemExit(1)

    selected_rows = image_query.with_entities(
        Image.id,
        Image.region_id,
        Region.city,
        Region.country,
        Image.source,
        Image.id_from_source,
        Image.source_captured_at,
    ).order_by(Image.id)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "region_id",
                "city",
                "country",
                "source",
                "source_image_id",
                "captured_at",
                "detection_count",
                "severity_detection_count",
                "score",
            ],
        )
        writer.writeheader()

        scored_count = 0
        positive_count = 0

        with tqdm(total=total_images, desc="Scoring images", unit="img") as progress:
            for batch in iter_image_batches(selected_rows, args.batch_size):
                image_ids = [row.id for row in batch]
                scores_by_image, severity_detection_count_by_image = scorer.score_images_with_summary(
                    image_ids=image_ids
                )
                detection_count_by_image = fetch_detection_counts(db, image_ids)

                if not args.dry_run:
                    db.session.bulk_update_mappings(
                        Image,
                        [
                            {"id": image_id, "score": float(scores_by_image.get(image_id, 0.0))}
                            for image_id in image_ids
                        ],
                    )
                    db.session.commit()

                for row in batch:
                    score = float(scores_by_image.get(row.id, 0.0))
                    if score > 0.0:
                        positive_count += 1
                    writer.writerow(
                        {
                            "image_id": row.id,
                            "region_id": row.region_id or "",
                            "city": row.city or "",
                            "country": row.country or "",
                            "source": row.source or "",
                            "source_image_id": row.id_from_source or "",
                            "captured_at": row.source_captured_at.isoformat() if row.source_captured_at else "",
                            "detection_count": detection_count_by_image.get(row.id, 0),
                            "severity_detection_count": severity_detection_count_by_image.get(row.id, 0),
                            "score": score,
                        }
                    )

                batch_size = len(batch)
                scored_count += batch_size
                progress.update(batch_size)

    logger.info(
        "Scored %s image(s); %s image(s) have score > 0.0.",
        total_images,
        positive_count,
    )
    if not args.dry_run:
        logger.info("Updated images.score in the database.")
    logger.info("Saved image scores to %s", output_path)


if __name__ == "__main__":
    main()
