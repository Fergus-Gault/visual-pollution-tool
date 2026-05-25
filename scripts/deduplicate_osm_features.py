import argparse
import sys
from pathlib import Path

from sqlalchemy import func, select, tuple_
from sqlalchemy.sql import over

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager, OSMFeature  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        prog="DeduplicateOSMFeatures",
        description=(
            "Check for duplicate osm_features by (region_id, osm_id) and "
            "optionally delete extra rows while keeping one row per pair."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete duplicate rows. Without this flag, the script is dry-run only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of duplicate osm_id groups to print in the preview.",
    )
    return parser.parse_args()


def duplicate_counts_stmt():
    return (
        select(
            OSMFeature.region_id.label("region_id"),
            OSMFeature.osm_id.label("osm_id"),
            func.count(OSMFeature.id).label("row_count"),
        )
        .where(OSMFeature.osm_id.is_not(None))
        .group_by(OSMFeature.region_id, OSMFeature.osm_id)
        .having(func.count(OSMFeature.id) > 1)
    )


def duplicate_rows_subquery():
    return (
        select(
            OSMFeature.id.label("id"),
            OSMFeature.region_id.label("region_id"),
            OSMFeature.osm_id.label("osm_id"),
            OSMFeature.osm_type.label("osm_type"),
            OSMFeature.created_at.label("created_at"),
            over(
                func.row_number(),
                partition_by=(OSMFeature.region_id, OSMFeature.osm_id),
                order_by=(OSMFeature.created_at.asc(), OSMFeature.id.asc()),
            ).label("rn"),
        )
        .where(OSMFeature.osm_id.is_not(None))
        .subquery()
    )


def main():
    args = parse_args()
    db = DatabaseManager()

    duplicate_counts = duplicate_counts_stmt().subquery()

    duplicate_group_count = db.session.scalar(
        select(func.count()).select_from(duplicate_counts)
    ) or 0
    if duplicate_group_count == 0:
        print("No duplicate osm_features found by osm_id.")
        return 0

    duplicate_row_count = db.session.scalar(
        select(func.coalesce(func.sum(duplicate_counts.c.row_count - 1), 0))
    ) or 0

    print(f"Duplicate osm_id groups: {duplicate_group_count}")
    print(f"Duplicate rows removable: {duplicate_row_count}")
    print("Preview:")

    preview_groups = db.session.execute(
        select(
            duplicate_counts.c.region_id,
            duplicate_counts.c.osm_id,
            duplicate_counts.c.row_count,
        )
        .order_by(
            duplicate_counts.c.row_count.desc(),
            duplicate_counts.c.region_id.asc(),
            duplicate_counts.c.osm_id.asc(),
        )
        .limit(args.limit)
    ).all()

    duplicate_rows = duplicate_rows_subquery()
    preview_keys = [(region_id, osm_id) for region_id, osm_id, _row_count in preview_groups]

    if preview_keys:
        preview_rows = db.session.execute(
            select(
                duplicate_rows.c.id,
                duplicate_rows.c.region_id,
                duplicate_rows.c.osm_id,
                duplicate_rows.c.osm_type,
                duplicate_rows.c.created_at,
                duplicate_rows.c.rn,
            )
            .where(
                tuple_(duplicate_rows.c.region_id, duplicate_rows.c.osm_id).in_(preview_keys)
            )
            .order_by(
                duplicate_rows.c.region_id.asc(),
                duplicate_rows.c.osm_id.asc(),
                duplicate_rows.c.rn.asc(),
                duplicate_rows.c.id.asc(),
            )
        ).all()

        grouped_preview = {}
        for row in preview_rows:
            grouped_preview.setdefault((row.region_id, row.osm_id), []).append(row)

        for region_id, osm_id, row_count in preview_groups:
            rows = grouped_preview.get((region_id, osm_id), [])
            kept = next((row for row in rows if row.rn == 1), None)
            kept_text = kept.id if kept is not None else "<missing>"
            print(
                f"region_id={region_id}, osm_id={osm_id}, rows={int(row_count)}, keep={kept_text}"
            )
            for row in rows:
                if row.rn == 1:
                    continue
                print(
                    "  delete "
                    f"id={row.id}, region_id={row.region_id}, osm_type={row.osm_type}, "
                    f"created_at={row.created_at}"
                )

    if not args.apply:
        print("Dry run only. Re-run with --apply to delete duplicate rows.")
        return 0

    delete_ids_subquery = (
        select(duplicate_rows.c.id)
        .where(duplicate_rows.c.rn > 1)
    )

    deleted = (
        db.session.query(OSMFeature)
        .filter(OSMFeature.id.in_(delete_ids_subquery))
        .delete(synchronize_session=False)
    )
    db.session.commit()

    print(f"Deleted duplicate osm_features: {deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
