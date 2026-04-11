import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from sqlalchemy import or_, select
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils import RateLimiter, setup_logger
from src.database.models import Image
from src.database import DatabaseManager
from src.config import PipelineConfig
from src.api import MapillaryAPI


logger = setup_logger(__name__)

REFRESH_AFTER_DAYS = 28
DEFAULT_BATCH_SIZE = 10000


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
    thumb_url = fetch_thumb_url(api, source_id, session=session)
    return image.id, source_id, thumb_url


def fetch_thumb_url(api: MapillaryAPI, image_id: str, session=None):
    response = api.send_request(
        image_id,
        params={"fields": "thumb_1024_url"},
        session=session,
    )
    return response.get("thumb_1024_url")


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
                            _, source_id, thumb_url = future.result()
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
        "Finished refreshing Mapillary URLs. processed=%s refreshed=%s skipped=%s failed=%s",
        processed,
        refreshed,
        skipped,
        failed,
    )


if __name__ == "__main__":
    main()
