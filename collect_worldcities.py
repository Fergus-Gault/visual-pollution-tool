import argparse
import csv

from src.pipeline import Pipeline
from src.config import Config


def normalize(value):
    return value.strip() if isinstance(value, str) else ""


def parse_population(row):
    population_raw = normalize(row.get("population"))
    return int(float(population_raw)) if population_raw else None


def parse_coords(row):
    try:
        lng = float(normalize(row.get("lng")))
        lat = float(normalize(row.get("lat")))
    except (TypeError, ValueError):
        return None, None
    return lng, lat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="World cities collection",
        description="Collects street view imagery for all cities with population >= threshold")
    parser.add_argument("--file", "-f", default=Config.DEFAULT_CSV)
    parser.add_argument("--min-population", "-p",
                        type=int, default=Config.MIN_POPULATION)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--collect-only", "-co", action="store_true")
    parser.add_argument("--override", "-or", action="store_true")
    parser.add_argument("--region-method", "-mr", default="shape")
    parser.add_argument("--image-sources", "-is", default="mapillary",
                        help="Comma-separated image sources: mapillary, kartaview, or both.")
    parser.add_argument("--dense", "-dn", action="store_true")
    parser.add_argument("--fetch-osm", "-fo",
                        action="store_true", default=True)
    args = parser.parse_args()

    if args.debug:
        Config.DEBUG = True

    with open(args.file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            row for row in reader
            if parse_population(row) is not None and parse_population(row) >= args.min_population
        ]

    pipeline = Pipeline(image_sources=args.image_sources)
    for row in rows:
        city = normalize(row.get("city_ascii")) or None
        country = normalize(row.get("country")) or None
        iso3 = normalize(row.get("iso3")).upper() or None
        start_captured_at = (
            normalize(row.get("start_captured_at"))
            or normalize(row.get("start_capture_date"))
            or None
        )
        end_captured_at = (
            normalize(row.get("end_captured_at"))
            or normalize(row.get("end_capture_date"))
            or None
        )
        population = parse_population(row)
        lng, lat = parse_coords(row)
        if lng is None or lat is None:
            coords = pipeline.get_lnglat(
                city, country) if city and country else None
            if coords is None:
                continue
            lng, lat = coords
        pipeline.run_region_coords(
            lng=lng,
            lat=lat,
            city=city,
            country=country,
            iso3=iso3,
            population=population,
            collect_only=args.collect_only,
            override=args.override,
            region_method=args.region_method,
            dense_scan=args.dense,
            fetch_osm=args.fetch_osm,
            start_captured_at=start_captured_at,
            end_captured_at=end_captured_at,
        )
