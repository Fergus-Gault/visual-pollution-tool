import argparse
import csv
import sys
from pathlib import Path

from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager, OSMFeature  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        prog="CountOSMFeatureTypes",
        description="Output the number of rows for each OSM feature type in the database.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db = DatabaseManager()

    rows = (
        db.session.query(OSMFeature.osm_type, func.count(OSMFeature.id))
        .group_by(OSMFeature.osm_type)
        .order_by(func.count(OSMFeature.id).desc(), OSMFeature.osm_type.asc())
        .all()
    )

    if not rows:
        print("No OSM features found in the database.")
        return 0

    total = 0
    for osm_type, count in rows:
        feature_type = (osm_type or "").strip() or "<blank>"
        feature_count = int(count)
        total += feature_count
        print(f"{feature_type},{feature_count}")

    print(f"total,{total}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["osm_type", "count"])
            for osm_type, count in rows:
                feature_type = (osm_type or "").strip() or "<blank>"
                writer.writerow([feature_type, int(count)])
            writer.writerow(["total", total])
        print(f"saved_csv,{output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
