import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.database import DatabaseManager
from src.pipeline.scanner import Scanner


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rescan OSM data for regions already stored in the database and "
            "append only new targeted features: non-billboard advertising, "
            "selected barriers, and road signs."
        )
    )
    parser.add_argument(
        "region_ids",
        nargs="*",
        help="Optional list of region IDs to rescan. Defaults to all regions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of regions to process.",
    )
    parser.add_argument(
        "--include-dense",
        action="store_true",
        help="Include dense-scan regions as well.",
    )
    return parser.parse_args()


def get_target_regions(db, region_ids, include_dense):
    if region_ids:
        regions = []
        missing_ids = []
        for region_id in region_ids:
            region = db.get_region(region_id)
            if region is None:
                missing_ids.append(region_id)
                continue
            if region.dense_scan and not include_dense:
                continue
            regions.append(region)
        if missing_ids:
            print(f"Skipping missing region IDs: {', '.join(missing_ids)}")
        return regions

    regions = db.get_all_regions()
    if not include_dense:
        regions = [region for region in regions if not region.dense_scan]
    return regions


def describe_region(region):
    city = region.city or "Unknown city"
    country = region.country or "Unknown country"
    return f"{region.id} | {city}, {country}"


def main():
    args = parse_args()
    db = DatabaseManager()
    scanner = Scanner(db, apis=[])

    regions = get_target_regions(
        db,
        region_ids=args.region_ids,
        include_dense=args.include_dense,
    )
    if args.limit is not None:
        regions = regions[:args.limit]

    if not regions:
        print("No regions matched the requested filters.")
        return

    print(
        f"Rescanning targeted OSM data for {len(regions)} region(s): "
        "advertising, selected barriers, and road signs."
    )

    scanned_count = 0
    changed_count = 0
    for index, region in enumerate(regions, start=1):
        before_count = db.count_osm_features_by_region(region.id)
        success = scanner.rescan_targeted_features_region(region)
        after_count = db.count_osm_features_by_region(region.id)
        added_count = after_count - before_count

        scanned_count += 1
        if added_count > 0:
            changed_count += 1

        status = "updated" if added_count > 0 else ("checked" if success else "no-data")
        print(
            f"[{index}/{len(regions)}] {status}: {describe_region(region)} | "
            f"added_targeted_features={added_count} | total_osm_features={after_count}"
        )

    print(
        f"Finished rescanning {scanned_count} region(s). "
        f"Regions with new targeted OSM features: {changed_count}."
    )


if __name__ == "__main__":
    main()
