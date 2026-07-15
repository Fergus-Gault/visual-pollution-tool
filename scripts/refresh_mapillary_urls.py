import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from dateutil import parser as date_parser
from dateutil.parser import ParserError
from sqlalchemy import or_, select
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils import RateLimiter, setup_logger
from src.database.models import Image
from src.database import DatabaseManager
from src.config import MapillaryConfig, PipelineConfig
from src.api import MapillaryAPI
from src.api.models import ImageMetadata


logger = setup_logger(__name__)

REFRESH_AFTER_DAYS = 28
DEFAULT_BATCH_SIZE = 10000
REFRESHABLE_FIELDS = (
    "lng",
    "lat",
    "source_captured_at",
    "source",
    "width",
    "height",
    "altitude",
    "atomic_scale",
    "camera_parameters",
    "camera_type",
    "compass_angle",
    "computed_altitude",
    "computed_compass_angle",
    "computed_rotation",
    "creator_id",
    "creator_username",
    "exif_orientation",
    "is_pano",
    "camera_make",
    "camera_model",
    "on_foot",
    "organization_id",
    "organization_name",
    "organization_slug",
    "quality_score",
    "sequence",
    "source_metadata",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refresh stored Mapillary image URLs from the Graph API."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh all Mapillary image URLs regardless of url_fetched_at.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to refresh.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of database rows to stream and process at a time.",
    )
    return parser.parse_args()


def build_query(force: bool, cutoff: datetime):
    stmt = select(Image.id, Image.id_from_source).where(
        Image.source == "mapillary",
        Image.id_from_source.isnot(None),
    )
    if not force:
        stmt = stmt.where(
            or_(
                Image.url_fetched_at.is_(None),
                Image.url_fetched_at < cutoff,
            )
        )
        return stmt.order_by(Image.url_fetched_at.asc().nullsfirst(), Image.id.asc())
    return stmt.order_by(Image.id.asc())


def iter_image_batches(session, stmt, batch_size: int):
    result = session.execute(stmt.execution_options(yield_per=batch_size))
    batch = []
    for row in result:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def normalise_mapillary_image_id(image_id: str):
    return image_id.split("|", 1)[0]


def create_session(num_workers: int):
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=num_workers,
        pool_maxsize=num_workers * 4,
        max_retries=0,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_one(api: MapillaryAPI, image: Image, session):
    source_id = normalise_mapillary_image_id(image.id_from_source)
    values = fetch_mapillary_image_values(api, source_id, session=session)
    return image.id, source_id, values


def fetch_mapillary_image_values(api: MapillaryAPI, image_id: str, session=None):
    response = api.send_request(
        image_id,
        params={"fields": MapillaryConfig.DEFAULT_FIELDS},
        session=session,
    )
    metadata = ImageMetadata.from_mapillary(response).to_dict()
    geometry = metadata.get("geometry") or {}
    coords = geometry.get("coordinates") or [None, None]

    return {
        "url": metadata.get("thumb_1024_url"),
        "lng": coords[0] if len(coords) > 0 else None,
        "lat": coords[1] if len(coords) > 1 else None,
        "source_captured_at": parse_captured_at(metadata.get("captured_at")),
        "source": metadata.get("_source"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "altitude": metadata.get("altitude"),
        "atomic_scale": metadata.get("atomic_scale"),
        "camera_parameters": metadata.get("camera_parameters"),
        "camera_type": metadata.get("camera_type"),
        "compass_angle": metadata.get("compass_angle"),
        "computed_altitude": metadata.get("computed_altitude"),
        "computed_compass_angle": metadata.get("computed_compass_angle"),
        "computed_rotation": metadata.get("computed_rotation"),
        "creator_id": metadata.get("creator_id"),
        "creator_username": metadata.get("creator_username"),
        "exif_orientation": metadata.get("exif_orientation"),
        "is_pano": metadata.get("is_pano"),
        "camera_make": metadata.get("make"),
        "camera_model": metadata.get("model"),
        "on_foot": metadata.get("on_foot"),
        "organization_id": metadata.get("organization_id"),
        "organization_name": metadata.get("organization_name"),
        "organization_slug": metadata.get("organization_slug"),
        "quality_score": metadata.get("quality_score"),
        "sequence": metadata.get("sequence"),
        "source_metadata": metadata.get("mapillary_metadata"),
    }


def parse_captured_at(value):
    if value is None:
        return None
    if isinstance(value, int):
        return datetime.fromtimestamp(value / 1000.0)
    if isinstance(value, str):
        try:
            return date_parser.parse(value)
        except (ParserError, ValueError, TypeError):
            return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return None


def is_empty(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) == 0
    return False


def fill_empty_fields(image: Image, values: dict):
    updated = 0
    for field_name in REFRESHABLE_FIELDS:
        value = values.get(field_name)
        if not is_empty(value) and is_empty(getattr(image, field_name)):
            setattr(image, field_name, value)
            updated += 1
    return updated


def main():
    args = parse_args()
    cutoff = datetime.now(timezone.utc) - timedelta(days=REFRESH_AFTER_DAYS)
    batch_size = max(1, args.batch_size)

    read_db = DatabaseManager()
    write_db = DatabaseManager()
    rate_limiter = RateLimiter(max_calls=PipelineConfig.MAPILLARY_RATE_LIMIT)
    api = MapillaryAPI(rate_limiter=rate_limiter)

    stmt = build_query(args.force, cutoff)
    refreshed = 0
    filled_fields = 0
    skipped = 0
    failed = 0
    processed = 0
    has_work = False
    remaining = args.limit

    logger.info(
        "Starting Mapillary URL refresh%s with batch_size=%s.",
        " with --force" if args.force else f" for rows older than {REFRESH_AFTER_DAYS} days",
        batch_size,
    )

    with tqdm(desc="Refreshing Mapillary URLs", unit="img") as progress:
        for batch_rows in iter_image_batches(read_db.session, stmt, batch_size):
            if remaining is not None and remaining <= 0:
                break

            if remaining is not None:
                batch_rows = batch_rows[:remaining]

            if not batch_rows:
                break

            has_work = True
            batch_ids = [row.id for row in batch_rows]
            batch_source_ids = {
                row.id: row.id_from_source for row in batch_rows}
            images = write_db.session.query(Image).filter(
                Image.id.in_(batch_ids)).all()
            image_lookup = {image.id: image for image in images}
            num_workers = min(
                PipelineConfig.MAPILLARY_WORKERS, len(batch_rows))

            with create_session(num_workers) as session:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(
                            fetch_one,
                            api,
                            image_lookup[row.id],
                            session,
                        ): row.id
                        for row in batch_rows
                        if row.id in image_lookup
                    }
                    for future in as_completed(futures):
                        image_id = futures[future]
                        image = image_lookup[image_id]
                        try:
                            _, source_id, values = future.result()
                            thumb_url = values.get("url")
                            filled_fields += fill_empty_fields(image, values)
                            if not thumb_url:
                                skipped += 1
                                logger.warning(
                                    "No thumb_1024_url returned for image %s.", source_id
                                )
                            else:
                                image.url = thumb_url
                                image.url_fetched_at = datetime.now(
                                    timezone.utc)
                                refreshed += 1
                        except Exception as exc:
                            failed += 1
                            logger.warning(
                                "Failed to refresh Mapillary image %s: %s",
                                batch_source_ids.get(
                                    image_id, image.id_from_source),
                                exc,
                            )
                        finally:
                            processed += 1
                            progress.update(1)

            write_db.session.commit()
            if remaining is not None:
                remaining -= len(batch_rows)

    if not has_work:
        logger.info("No Mapillary image URLs need refreshing.")
        return

    logger.info(
        "Finished refreshing Mapillary URLs. processed=%s refreshed=%s filled_fields=%s skipped=%s failed=%s",
        processed,
        refreshed,
        filled_fields,
        skipped,
        failed,
    )


if __name__ == "__main__":
    main()
