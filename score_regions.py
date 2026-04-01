import argparse
from pathlib import Path
import csv
from sqlalchemy import func

from src.database import DatabaseManager, Image, OSMFeature
from src.pipeline import Scorer
from src.utils import setup_logger


logger = setup_logger(__name__)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ScoreRegions",
        description="Score all regions using the selected scoring method.",
    )
    parser.add_argument(
        "--method",
        choices=["vpi", "vpi_osm"],
        default="vpi",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Optional city name to score only matching regions.",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Optional country filter when using --city.",
    )
    args = parser.parse_args()

    db = DatabaseManager()
    scorer = Scorer(db)
    if args.city:
        regions = find_regions_by_city(db, args.city, args.country)
        if not regions:
            country_text = f" in {args.country}" if args.country else ""
            logger.warning(f"No regions found for city '{args.city}'{country_text}")
            raise SystemExit(1)
    else:
        regions = db.get_all_regions()

    region_ids = [region.id for region in regions]

    if not region_ids:
        logger.warning("No regions found to score.")
        raise SystemExit(1)

    image_rows = (
        db.session.query(Image.region_id, func.count(Image.id))
        .filter(Image.region_id.in_(region_ids))
        .group_by(Image.region_id)
        .all()
    )
    image_count_by_region = {
        region_id: count for region_id, count in image_rows}

    osm_feature_count_by_region = {}
    if args.method == "vpi_osm":
        osm_rows = (
            db.session.query(OSMFeature.region_id, func.count(OSMFeature.id))
            .filter(OSMFeature.region_id.in_(region_ids))
            .group_by(OSMFeature.region_id)
            .all()
        )
        osm_feature_count_by_region = {
            region_id: count for region_id, count in osm_rows}

    if args.method == "vpi":
        scores_by_region = scorer.score_regions(region_ids=region_ids)
    else:
        scores_by_region = scorer.score_regions_with_osm(region_ids=region_ids)

    scored_rows = []
    for region in regions:
        score = scores_by_region.get(region.id, 0.0)
        region.score = score
        scored_rows.append((region.id, region.city, region.country, score))

    db.session.commit()

    suffix = ""
    if args.city:
        city_token = normalize(args.city).replace(" ", "_")
        suffix = f"_{city_token}"
        if args.country:
            country_token = normalize(args.country).replace(" ", "_")
            suffix += f"_{country_token}"

    scores_path = Path(f"./data/scores_{args.method}{suffix}.csv")
    scored_rows = sorted(scored_rows, key=lambda row: row[3], reverse=True)
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for region_id, city, country, score in scored_rows:
            image_count = image_count_by_region.get(region_id, 0)
            if args.method == "vpi_osm":
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
            if args.method == "vpi_osm":
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
            if args.method == "vpi_osm":
                osm_feature_count = osm_feature_count_by_region.get(
                    region_id, 0)
                logger.info(
                    f"region_id={region_id}, city={city}, country={country}, images={image_count}, osm_features={osm_feature_count}, score={score:.6f}")
            else:
                logger.info(
                    f"region_id={region_id}, city={city}, country={country}, images={image_count}, score={score:.6f}")
    else:
        logger.info("No regions with score > 0.0")
