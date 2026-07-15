import argparse
import math
import sys
from pathlib import Path

from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.database import DatabaseManager, Image, Region


EARTH_RADIUS_KM = 6371.0088


def parse_args():
    parser = argparse.ArgumentParser(
        prog="CalculateImagesPerSquareKm",
        description=(
            "Calculate image density for each region and store it in "
            "regions.images_per_square_km."
        ),
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Optional country filter for regions to update.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Optional city filter for regions to update.",
    )
    parser.add_argument(
        "--region-id",
        default=None,
        help="Optional single region id to update.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional image source filter, e.g. mapillary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print calculated densities without writing them to the database.",
    )
    return parser.parse_args()


def normalise(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def region_area_square_km(region):
    min_lat = math.radians(max(-90.0, min(90.0, region.min_lat)))
    max_lat = math.radians(max(-90.0, min(90.0, region.max_lat)))
    min_lng = math.radians(region.min_lng)
    max_lng = math.radians(region.max_lng)

    lat_span = abs(math.sin(max_lat) - math.sin(min_lat))
    lng_span = abs(max_lng - min_lng)
    if lng_span > math.tau:
        lng_span = math.tau

    return (EARTH_RADIUS_KM ** 2) * lat_span * lng_span


def build_region_query(db, args):
    query = db.session.query(Region)
    if args.region_id:
        query = query.filter(Region.id == args.region_id)
    if args.country:
        query = query.filter(
            func.lower(func.coalesce(Region.country, "")) == normalise(args.country)
        )
    if args.city:
        query = query.filter(
            func.lower(func.coalesce(Region.city, "")) == normalise(args.city)
        )
    return query.order_by(Region.country.asc(), Region.city.asc(), Region.name.asc())


def get_image_counts(db, region_ids, source=None):
    if not region_ids:
        return {}

    query = (
        db.session.query(Image.region_id, func.count(Image.id))
        .filter(Image.region_id.in_(region_ids))
    )
    if source:
        query = query.filter(
            func.lower(func.coalesce(Image.source, "")) == normalise(source)
        )

    return {
        region_id: int(image_count or 0)
        for region_id, image_count in query.group_by(Image.region_id).all()
    }


def calculate_rows(db, args):
    regions = build_region_query(db, args).all()
    image_counts = get_image_counts(db, [region.id for region in regions], args.source)

    rows = []
    for region in regions:
        area_square_km = region_area_square_km(region)
        image_count = image_counts.get(region.id, 0)
        images_per_square_km = (
            image_count / area_square_km if area_square_km > 0 else None
        )
        rows.append((region, image_count, area_square_km, images_per_square_km))
    return rows


def print_rows(rows, dry_run):
    action = "Would update" if dry_run else "Updated"
    print(f"{action} {len(rows)} regions")
    for region, image_count, area_square_km, density in rows[:20]:
        density_text = "n/a" if density is None else f"{density:.6f}"
        print(
            f"{region.id} | {region.name} | images={image_count} | "
            f"area_km2={area_square_km:.6f} | images_per_square_km={density_text}"
        )
    if len(rows) > 20:
        print(f"... {len(rows) - 20} more regions omitted")


def update_regions(db, rows):
    for region, _image_count, _area_square_km, density in rows:
        region.images_per_square_km = density
    db.session.commit()


def main():
    args = parse_args()
    db = DatabaseManager()

    rows = calculate_rows(db, args)
    if not args.dry_run:
        update_regions(db, rows)
    print_rows(rows, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
