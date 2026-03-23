import argparse
import csv
import tempfile
import os

from src.pipeline import PipelineMP
from src.config import Config


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
    parser.add_argument("--dense", "-dn", action="store_true")
    parser.add_argument("--fetch-osm", "-fo",
                        action="store_true", default=True)
    args = parser.parse_args()

    if args.debug:
        Config.DEBUG = True

    with open(args.file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        pop_idx = header.index("population")
        rows = [
            row for row in reader
            if row[pop_idx].strip() and float(row[pop_idx].strip()) >= args.min_population
        ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="") as tmp:
        writer = csv.writer(tmp)
        writer.writerow(header)
        writer.writerows(rows)
        tmp_path = tmp.name

    try:
        pipeline = PipelineMP(tmp_path, collect_only=args.collect_only, override=args.override,
                              region_method=args.region_method, dense_scan=args.dense,
                              fetch_osm=args.fetch_osm)
        pipeline.start_mp()
    finally:
        os.unlink(tmp_path)
