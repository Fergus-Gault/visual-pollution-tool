import argparse

from src.pipeline import DatasetManager
from src.database import DatabaseManager
from src.utils import setup_logger

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download all images from the database into tar shards with an NDJSON index."
    )
    parser.add_argument(
        "--download-path",
        dest="download_path",
        default=None,
        help="Base directory to write the dataset folder into.",
    )
    parser.add_argument(
        "--shard-size",
        dest="shard_size",
        type=int,
        default=10000,
        help="Number of samples per tar shard.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    db = DatabaseManager()
    ds = DatasetManager(
        db,
        base_path=args.download_path,
        shard_size=args.shard_size,
    )

    ds_path = ds.download_data()
    logger.info(f"Downloaded data to: {ds_path}")
