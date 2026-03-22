import argparse
from pathlib import Path
import csv
from tqdm import tqdm
from sqlalchemy import func

from src.database import DatabaseManager, Image, OSMFeature
from src.pipeline import Scorer
from src.utils import setup_logger


logger = setup_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ScoreRegions",
        description="Score all regions using the selected scoring method.",
    )
    parser.add_argument(
        "--method",
        choices=["bulk", "naive", "bulk_osm"],
        default="bulk",
    )
    args = parser.parse_args()

    db = DatabaseManager()
    scorer = Scorer(db)
    regions = db.get_all_regions()
    region_ids = [region.id for region in regions]

    image_rows = (
        db.session.query(Image.region_id, func.count(Image.id))
        .group_by(Image.region_id)
        .all()
    )
    image_count_by_region = {
        region_id: count for region_id, count in image_rows}

    osm_feature_count_by_region = {}
    if args.method == "bulk_osm":
        osm_rows = (
            db.session.query(OSMFeature.region_id, func.count(OSMFeature.id))
            .group_by(OSMFeature.region_id)
            .all()
        )
        osm_feature_count_by_region = {
            region_id: count for region_id, count in osm_rows}

    if args.method == "bulk":
        scores_by_region = scorer.score_regions(region_ids=region_ids)
    elif args.method == "bulk_osm":
        scores_by_region = scorer.score_regions_with_osm(region_ids=region_ids)
    else:
        scores_by_region = {
            region_id: scorer.score_region(region_id, method="naive")
            for region_id in tqdm(region_ids, total=len(region_ids), desc="Scoring regions")
        }

    scored_rows = []
    for region in regions:
        score = scores_by_region.get(region.id, 0.0)
        region.score = score
        scored_rows.append((region.id, region.city, region.country, score))

    db.session.commit()

    scores_path = Path(f"./data/scores_{args.method}.csv")
    scored_rows = sorted(scored_rows, key=lambda row: row[3], reverse=True)
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for region_id, city, country, score in scored_rows:
            image_count = image_count_by_region.get(region_id, 0)
            if args.method == "bulk_osm":
                osm_feature_count = osm_feature_count_by_region.get(
                    region_id, 0)
                writer.writerow([region_id, city, country,
                                image_count, osm_feature_count, score])
            else:
                writer.writerow([region_id, city, country, image_count, score])

    positive_scores = [row for row in scored_rows if row[3] > 0.0]
    if positive_scores:
        highest = sorted(
            positive_scores, key=lambda row: row[3], reverse=True)[:10]
        lowest = sorted(positive_scores, key=lambda row: row[3])[:10]
        logger.info("Highest positive scores:")
        for region_id, city, country, score in highest:
            image_count = image_count_by_region.get(region_id, 0)
            if args.method == "bulk_osm":
                osm_feature_count = osm_feature_count_by_region.get(
                    region_id, 0)
                logger.info(
                    f"region_id={region_id}, city={city}, country={country}, images={image_count}, osm_features={osm_feature_count}, score={score:.6f}")
            else:
                logger.info(
                    f"region_id={region_id}, city={city}, country={country}, images={image_count}, score={score:.6f}")
        logger.info("Lowest positive scores:")
        for region_id, city, country, score in lowest:
            image_count = image_count_by_region.get(region_id, 0)
            if args.method == "bulk_osm":
                osm_feature_count = osm_feature_count_by_region.get(
                    region_id, 0)
                logger.info(
                    f"region_id={region_id}, city={city}, country={country}, images={image_count}, osm_features={osm_feature_count}, score={score:.6f}")
            else:
                logger.info(
                    f"region_id={region_id}, city={city}, country={country}, images={image_count}, score={score:.6f}")
    else:
        logger.info("No regions with score > 0.0")
