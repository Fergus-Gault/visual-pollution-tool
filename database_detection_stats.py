import argparse
import csv
from pathlib import Path
import sys

from sqlalchemy import distinct, func

sys.path.append(str(Path(__file__).resolve().parent))

from src.database import DatabaseManager, Detection, Image, OSMFeature, Region


def normalize(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def parse_args():
    parser = argparse.ArgumentParser(
        prog="DatabaseDetectionStats",
        description="Summarise database detection coverage overall and by class.",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Optional country filter applied through the parent region.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Optional city filter applied through the parent region.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional image source filter, e.g. mapillary.",
    )
    parser.add_argument(
        "--output",
        default="data/database_detection_stats.csv",
        help="CSV path for per-class stats.",
    )
    return parser.parse_args()


def build_image_query(db, args):
    query = db.session.query(Image).outerjoin(Region, Image.region_id == Region.id)
    if args.country:
        query = query.filter(func.lower(func.coalesce(Region.country, "")) == normalize(args.country))
    if args.city:
        query = query.filter(func.lower(func.coalesce(Region.city, "")) == normalize(args.city))
    if args.source:
        query = query.filter(func.lower(func.coalesce(Image.source, "")) == normalize(args.source))
    return query


def build_detection_query(db, args):
    query = (
        db.session.query(Detection)
        .join(Image, Detection.image_id == Image.id)
        .outerjoin(Region, Image.region_id == Region.id)
    )
    if args.country:
        query = query.filter(func.lower(func.coalesce(Region.country, "")) == normalize(args.country))
    if args.city:
        query = query.filter(func.lower(func.coalesce(Region.city, "")) == normalize(args.city))
    if args.source:
        query = query.filter(func.lower(func.coalesce(Image.source, "")) == normalize(args.source))
    return query


def print_overall_stats(db, args, image_query, detection_query):
    region_query = db.session.query(Region)
    if args.country:
        region_query = region_query.filter(func.lower(func.coalesce(Region.country, "")) == normalize(args.country))
    if args.city:
        region_query = region_query.filter(func.lower(func.coalesce(Region.city, "")) == normalize(args.city))
    if args.source:
        region_query = region_query.filter(
            Region.id.in_(image_query.with_entities(Image.region_id).distinct())
        )

    region_count = region_query.distinct(Region.id).count()
    image_count = image_query.count()
    scored_image_count = image_query.filter(Image.score.isnot(None)).count()
    detection_count = detection_query.count()
    image_with_detection_count = (
        detection_query.with_entities(distinct(Detection.image_id)).count()
    )
    class_count = detection_query.with_entities(distinct(Detection.label)).count()
    osm_feature_count = (
        db.session.query(OSMFeature)
        .filter(
            OSMFeature.region_id.in_(region_query.with_entities(Region.id))
        )
        .count()
    )

    print("Overall stats")
    print(f"Regions: {region_count}")
    print(f"Images: {image_count}")
    print(f"Images with stored score: {scored_image_count}")
    print(f"Images with detections: {image_with_detection_count}")
    print(f"Detections: {detection_count}")
    print(f"Detection classes present: {class_count}")
    print(f"OSM features: {osm_feature_count}")


def fetch_label_stats(detection_query):
    rows = (
        detection_query.with_entities(
            Detection.label.label("label"),
            func.count(Detection.id).label("detection_count"),
            func.count(distinct(Detection.image_id)).label("image_count"),
            func.count(distinct(Image.region_id)).label("region_count"),
            func.count(distinct(Region.country)).label("country_count"),
            func.count(distinct(Region.city)).label("city_count"),
            func.avg(Detection.confidence).label("avg_confidence"),
            func.min(Detection.confidence).label("min_confidence"),
            func.max(Detection.confidence).label("max_confidence"),
        )
        .group_by(Detection.label)
        .order_by(func.count(Detection.id).desc(), Detection.label.asc())
        .all()
    )
    return rows


def save_label_stats(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "detection_count",
                "image_count",
                "region_count",
                "country_count",
                "city_count",
                "avg_confidence",
                "min_confidence",
                "max_confidence",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.label,
                    int(row.detection_count or 0),
                    int(row.image_count or 0),
                    int(row.region_count or 0),
                    int(row.country_count or 0),
                    int(row.city_count or 0),
                    "" if row.avg_confidence is None else f"{float(row.avg_confidence):.6f}",
                    "" if row.min_confidence is None else f"{float(row.min_confidence):.6f}",
                    "" if row.max_confidence is None else f"{float(row.max_confidence):.6f}",
                ]
            )


def print_label_stats(rows):
    print("\nPer-class stats")
    for row in rows:
        avg_conf = "n/a" if row.avg_confidence is None else f"{float(row.avg_confidence):.3f}"
        print(
            f"{row.label}: detections={int(row.detection_count or 0)} | "
            f"images={int(row.image_count or 0)} | "
            f"regions={int(row.region_count or 0)} | "
            f"countries={int(row.country_count or 0)} | "
            f"cities={int(row.city_count or 0)} | "
            f"avg_conf={avg_conf}"
        )


def main():
    args = parse_args()
    db = DatabaseManager()
    image_query = build_image_query(db, args)
    detection_query = build_detection_query(db, args)

    print_overall_stats(db, args, image_query, detection_query)
    label_rows = fetch_label_stats(detection_query)
    if not label_rows:
        raise SystemExit("No detections matched the requested filters.")

    print_label_stats(label_rows)
    output_path = Path(args.output)
    save_label_stats(label_rows, output_path)
    print(f"\nSaved per-class stats to {output_path}")


if __name__ == "__main__":
    main()
