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
        choices=["vpi", "osm", "vpi_osm"],
        default="vpi",
    )
    parser.add_argument(
        "--update-db",
        action="store_true",
        help="Also persist computed scores to region.score in the database.",
    )
    parser.add_argument(
        "--compare-vpi-osm",
        action="store_true",
        help="Export a side-by-side comparison of VPI-only and OSM-only scores.",
    )
    parser.add_argument(
        "--exclude-zero-comparisons",
        action="store_true",
        help="When comparing VPI and OSM scores, exclude rows where either score is zero.",
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
    if args.compare_vpi_osm and args.update_db:
        parser.error("--compare-vpi-osm cannot be combined with --update-db because it computes two scores per region.")
    if args.exclude_zero_comparisons and not args.compare_vpi_osm:
        parser.error("--exclude-zero-comparisons requires --compare-vpi-osm.")

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
    if args.method in {"osm", "vpi_osm"} or args.compare_vpi_osm:
        osm_rows = (
            db.session.query(OSMFeature.region_id, func.count(OSMFeature.id))
            .filter(OSMFeature.region_id.in_(region_ids))
            .group_by(OSMFeature.region_id)
            .all()
        )
        osm_feature_count_by_region = {
            region_id: count for region_id, count in osm_rows}

    if args.compare_vpi_osm:
        vpi_scores_by_region = scorer.score_regions(region_ids=region_ids)
        osm_scores_by_region = scorer.score_regions_with_osm_only(
            region_ids=region_ids)
    elif args.method == "vpi":
        scores_by_region = scorer.score_regions(region_ids=region_ids)
    elif args.method == "osm":
        scores_by_region = scorer.score_regions_with_osm_only(
            region_ids=region_ids)
    else:
        scores_by_region = scorer.score_regions_with_osm(region_ids=region_ids)

    suffix = ""
    if args.city:
        city_token = normalize(args.city).replace(" ", "_")
        suffix = f"_{city_token}"
        if args.country:
            country_token = normalize(args.country).replace(" ", "_")
            suffix += f"_{country_token}"

    if args.compare_vpi_osm:
        comparison_rows = []
        for region in regions:
            vpi_score = vpi_scores_by_region.get(region.id, 0.0)
            osm_score = osm_scores_by_region.get(region.id, 0.0)
            if args.exclude_zero_comparisons and (vpi_score == 0.0 or osm_score == 0.0):
                continue
            delta = osm_score - vpi_score
            comparison_rows.append(
                (region.id, region.city, region.country, vpi_score, osm_score, delta, abs(delta)))

        comparison_rows = sorted(
            comparison_rows, key=lambda row: row[6], reverse=True)
        comparison_path = Path(f"./data/scores_compare_vpi_osm{suffix}.csv")
        with open(comparison_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "region_id",
                "city",
                "country",
                "image_count",
                "osm_feature_count",
                "vpi_score",
                "osm_score",
                "osm_minus_vpi",
                "absolute_difference",
            ])
            for region_id, city, country, vpi_score, osm_score, delta, absolute_delta in comparison_rows:
                writer.writerow([
                    region_id,
                    city,
                    country,
                    image_count_by_region.get(region_id, 0),
                    osm_feature_count_by_region.get(region_id, 0),
                    vpi_score,
                    osm_score,
                    delta,
                    absolute_delta,
                ])

        differing_rows = [row for row in comparison_rows if row[6] > 0.0]
        if differing_rows:
            logger.info("Largest VPI vs OSM score differences:")
            for region_id, city, country, vpi_score, osm_score, delta, absolute_delta in differing_rows[:10]:
                logger.info(
                    f"region_id={region_id}, city={city}, country={country}, vpi_score={vpi_score:.6f}, osm_score={osm_score:.6f}, osm_minus_vpi={delta:.6f}, absolute_difference={absolute_delta:.6f}")
        else:
            logger.info("No VPI vs OSM score differences found.")
        raise SystemExit(0)

    scored_rows = []
    for region in regions:
        score = scores_by_region.get(region.id, 0.0)
        if args.update_db:
            region.score = score
        scored_rows.append((region.id, region.city, region.country, score))

    if args.update_db:
        db.session.commit()

    scores_path = Path(f"./data/scores_{args.method}{suffix}.csv")
    scored_rows = sorted(scored_rows, key=lambda row: row[3], reverse=True)
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if args.method in {"osm", "vpi_osm"}:
            writer.writerow([
                "region_id",
                "city",
                "country",
                "image_count",
                "osm_feature_count",
                "score",
            ])
        else:
            writer.writerow([
                "region_id",
                "city",
                "country",
                "image_count",
                "score",
            ])
        for region_id, city, country, score in scored_rows:
            image_count = image_count_by_region.get(region_id, 0)
            if args.method in {"osm", "vpi_osm"}:
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
            if args.method in {"osm", "vpi_osm"}:
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
            if args.method in {"osm", "vpi_osm"}:
                osm_feature_count = osm_feature_count_by_region.get(
                    region_id, 0)
                logger.info(
                    f"region_id={region_id}, city={city}, country={country}, images={image_count}, osm_features={osm_feature_count}, score={score:.6f}")
            else:
                logger.info(
                    f"region_id={region_id}, city={city}, country={country}, images={image_count}, score={score:.6f}")
    else:
        logger.info("No regions with score > 0.0")
