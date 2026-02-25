from src.pipeline import Pipeline
from src.config import Config, ArgsConfig
import sys
import argparse

if __name__ == "__main__":
    # TODO: Convert to argparse
    # There is an issue with arguments causing errors if a country is not entered
    parser = argparse.ArgumentParser(
        prog="Data collection", description="Collects street view imagery and OSM points")
    parser.add_argument("fileorcity")
    parser.add_argument("country")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--collect-only", "-co", action="store_true")
    parser.add_argument("--override", "-or", action="store_true")
    parser.add_argument("--region-method", "-mr", default="shape")
    parser.add_argument("--dense", "-d", action="store_true")
    pipeline = Pipeline()
    args = parser.parse_args()
    if args.debug:
        Config.DEBUG = True
    if ".csv" in args.fileorcity or ".txt" in args.fileorcity:
        pipeline.run(
            file_path=args.fileorcity, collect_only=args.collect_only, override=args.override, region_method=args.region_method, dense_scan=args.dense)
    else:
        pipeline.run(args=[args.fileorcity, args.country], collect_only=args.collect_only,
                     override=args.override, region_method=args.region_method, dense_scan=args.dense)
