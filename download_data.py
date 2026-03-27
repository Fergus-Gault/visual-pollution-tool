import argparse

from src.pipeline import DatasetManager
from src.database import DatabaseManager
from src.utils import setup_logger

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download all images from the database and export them as an NDJSON dataset."
    )
    parser.add_argument(
        "--download-path",
        dest="download_path",
        default=None,
        help="Base directory to write the dataset folder into.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    db = DatabaseManager()
    ds = DatasetManager(db, base_path=args.download_path)

    ds_path = ds.download_data()
    logger.info(f"Downloaded data to: {ds_path}")
