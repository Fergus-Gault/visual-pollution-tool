from src.pipeline import Pipeline
from src.config import Config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Data collection", description="Collects street view imagery and OSM points")
    parser.add_argument("fileorcity")
    parser.add_argument("--country", "-c", default="")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--collect-only", "-co", action="store_true")
    parser.add_argument("--override", "-or", action="store_true")
    parser.add_argument("--region-method", "-mr", default="shape")
    parser.add_argument("--dense", "-dn", action="store_true")
    parser.add_argument("--fetch-osm", "-fo", action="store_true")
    parser.add_argument("--image-sources", "-is", default="mapillary",
                        help="Comma-separated image sources: mapillary, kartaview, or both.")
    args = parser.parse_args()
    if args.debug:
        Config.DEBUG = True
    pipeline = Pipeline(image_sources=args.image_sources)
    if ".csv" in args.fileorcity or ".txt" in args.fileorcity:
        pipeline.run(
            file_path=args.fileorcity, collect_only=args.collect_only, override=args.override, region_method=args.region_method, dense_scan=args.dense, fetch_osm=args.fetch_osm)
    else:
        pipeline.run(args=[args.fileorcity, args.country], collect_only=args.collect_only,
                     override=args.override, region_method=args.region_method, dense_scan=args.dense, fetch_osm=args.fetch_osm)
