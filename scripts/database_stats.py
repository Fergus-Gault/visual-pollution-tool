import argparse
import csv
import sys
from pathlib import Path

from sqlalchemy import distinct, func

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.database import DatabaseManager, Detection, Image, OSMFeature, Region


def normalize(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def parse_args():
    parser = argparse.ArgumentParser(
        prog="DatabaseStats",
        description=(
            "Summarise database coverage using only base regions "
            "(excluding dense scans and time-window scans)."
        ),
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
        "--output-dir",
        default="data/database_stats",
        help="Directory where CSV outputs will be written.",
    )
    return parser.parse_args()


def apply_region_filters(query, args):
    query = query.filter(
        Region.dense_scan.is_(False),
        Region.start_captured_at.is_(None),
        Region.end_captured_at.is_(None),
    )
    if args.country:
        query = query.filter(
            func.lower(func.coalesce(Region.country, "")) == normalize(args.country)
        )
    if args.city:
        query = query.filter(
            func.lower(func.coalesce(Region.city, "")) == normalize(args.city)
        )
    return query


def build_region_query(db, args):
    return apply_region_filters(db.session.query(Region), args)


def build_image_query(db, args):
    query = db.session.query(Image).join(Region, Image.region_id == Region.id)
    query = apply_region_filters(query, args)
    if args.source:
        query = query.filter(
            func.lower(func.coalesce(Image.source, "")) == normalize(args.source)
        )
    return query


def build_detection_query(db, args):
    query = (
        db.session.query(Detection)
        .join(Image, Detection.image_id == Image.id)
        .join(Region, Image.region_id == Region.id)
    )
    query = apply_region_filters(query, args)
    if args.source:
        query = query.filter(
            func.lower(func.coalesce(Image.source, "")) == normalize(args.source)
        )
    return query


def build_osm_query(db, args, region_query):
    return db.session.query(OSMFeature).filter(
        OSMFeature.region_id.in_(region_query.with_entities(Region.id))
    )


def fetch_overall_stats(region_query, image_query, detection_query, osm_query):
    return {
        "region_count": region_query.distinct(Region.id).count(),
        "country_count": region_query.with_entities(
            distinct(Region.country)
        ).count(),
        "city_count": region_query.with_entities(distinct(Region.city)).count(),
        "scored_region_count": region_query.filter(Region.score.isnot(None))
        .distinct(Region.id)
        .count(),
        "image_count": image_query.count(),
        "scored_image_count": image_query.filter(Image.score.isnot(None)).count(),
        "image_with_detection_count": detection_query.with_entities(
            distinct(Detection.image_id)
        ).count(),
        "detection_count": detection_query.count(),
        "detection_class_count": detection_query.with_entities(
            distinct(Detection.label)
        ).count(),
        "osm_feature_count": osm_query.count(),
        "osm_type_count": osm_query.with_entities(distinct(OSMFeature.osm_type)).count(),
        "regions_with_osm_count": osm_query.with_entities(
            distinct(OSMFeature.region_id)
        ).count(),
    }


def fetch_detection_label_stats(detection_query):
    return (
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


def fetch_osm_type_stats(osm_query):
    return (
        osm_query.join(Region, OSMFeature.region_id == Region.id)
        .with_entities(
            OSMFeature.osm_type.label("osm_type"),
            func.count(OSMFeature.id).label("feature_count"),
            func.count(distinct(OSMFeature.region_id)).label("region_count"),
            func.count(distinct(Region.country)).label("country_count"),
            func.count(distinct(Region.city)).label("city_count"),
        )
        .group_by(OSMFeature.osm_type)
        .order_by(func.count(OSMFeature.id).desc(), OSMFeature.osm_type.asc())
        .all()
    )


def fetch_source_stats(image_query, detection_query):
    source_expr = func.coalesce(Image.source, "")
    detection_counts = {
        row.source_key: row
        for row in (
            detection_query.with_entities(
                source_expr.label("source_key"),
                func.count(Detection.id).label("detection_count"),
                func.count(distinct(Detection.image_id)).label(
                    "image_with_detection_count"
                ),
            )
            .group_by(source_expr)
            .all()
        )
    }
    rows = (
        image_query.with_entities(
            source_expr.label("source_key"),
            func.count(Image.id).label("image_count"),
            func.count(distinct(Image.region_id)).label("region_count"),
            func.count(Image.score).label("scored_image_count"),
        )
        .group_by(source_expr)
        .order_by(func.count(Image.id).desc(), source_expr.asc())
        .all()
    )
    merged = []
    for row in rows:
        source_key = row.source_key
        detection_row = detection_counts.get(source_key)
        merged.append(
            {
                "source": source_key or "<blank>",
                "image_count": int(row.image_count or 0),
                "region_count": int(row.region_count or 0),
                "scored_image_count": int(row.scored_image_count or 0),
                "image_with_detection_count": int(
                    detection_row.image_with_detection_count if detection_row else 0
                ),
                "detection_count": int(
                    detection_row.detection_count if detection_row else 0
                ),
            }
        )
    return merged


def print_overall_stats(stats):
    print("Overall stats")
    print("Region scope: dense_scan = false, start_captured_at is null, end_captured_at is null")
    print(f"Regions: {stats['region_count']}")
    print(f"Countries: {stats['country_count']}")
    print(f"Cities: {stats['city_count']}")
    print(f"Regions with stored score: {stats['scored_region_count']}")
    print(f"Images: {stats['image_count']}")
    print(f"Images with stored score: {stats['scored_image_count']}")
    print(f"Images with detections: {stats['image_with_detection_count']}")
    print(f"Detections: {stats['detection_count']}")
    print(f"Detection classes present: {stats['detection_class_count']}")
    print(f"OSM features: {stats['osm_feature_count']}")
    print(f"OSM types present: {stats['osm_type_count']}")
    print(f"Regions with OSM features: {stats['regions_with_osm_count']}")


def print_detection_label_stats(rows):
    print("\nPer-class stats")
    if not rows:
        print("No detections matched the requested filters.")
        return
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


def print_osm_type_stats(rows):
    print("\nPer-OSM-type stats")
    if not rows:
        print("No OSM features matched the requested filters.")
        return
    for row in rows:
        osm_type = (row.osm_type or "").strip() or "<blank>"
        print(
            f"{osm_type}: features={int(row.feature_count or 0)} | "
            f"regions={int(row.region_count or 0)} | "
            f"countries={int(row.country_count or 0)} | "
            f"cities={int(row.city_count or 0)}"
        )


def print_source_stats(rows):
    print("\nPer-source stats")
    if not rows:
        print("No images matched the requested filters.")
        return
    for row in rows:
        print(
            f"{row['source']}: images={row['image_count']} | "
            f"regions={row['region_count']} | "
            f"scored_images={row['scored_image_count']} | "
            f"images_with_detections={row['image_with_detection_count']} | "
            f"detections={row['detection_count']}"
        )


def save_overall_stats(stats, output_dir):
    output_path = output_dir / "overall_stats.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, int(value)])
    return output_path


def save_detection_label_stats(rows, output_dir):
    output_path = output_dir / "detection_label_stats.csv"
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
    return output_path


def save_osm_type_stats(rows, output_dir):
    output_path = output_dir / "osm_type_stats.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["osm_type", "feature_count", "region_count", "country_count", "city_count"]
        )
        for row in rows:
            writer.writerow(
                [
                    (row.osm_type or "").strip() or "<blank>",
                    int(row.feature_count or 0),
                    int(row.region_count or 0),
                    int(row.country_count or 0),
                    int(row.city_count or 0),
                ]
            )
    return output_path


def save_source_stats(rows, output_dir):
    output_path = output_dir / "source_stats.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "image_count",
                "region_count",
                "scored_image_count",
                "image_with_detection_count",
                "detection_count",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["source"],
                    row["image_count"],
                    row["region_count"],
                    row["scored_image_count"],
                    row["image_with_detection_count"],
                    row["detection_count"],
                ]
            )
    return output_path


def main():
    args = parse_args()
    db = DatabaseManager()

    region_query = build_region_query(db, args)
    image_query = build_image_query(db, args)
    detection_query = build_detection_query(db, args)
    osm_query = build_osm_query(db, args, region_query)

    overall_stats = fetch_overall_stats(
        region_query, image_query, detection_query, osm_query
    )
    detection_label_rows = fetch_detection_label_stats(detection_query)
    osm_type_rows = fetch_osm_type_stats(osm_query)
    source_rows = fetch_source_stats(image_query, detection_query)

    print_overall_stats(overall_stats)
    print_detection_label_stats(detection_label_rows)
    print_osm_type_stats(osm_type_rows)
    print_source_stats(source_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = [
        save_overall_stats(overall_stats, output_dir),
        save_detection_label_stats(detection_label_rows, output_dir),
        save_osm_type_stats(osm_type_rows, output_dir),
        save_source_stats(source_rows, output_dir),
    ]
    print("\nSaved CSVs")
    for path in saved_paths:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
