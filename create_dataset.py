from src.labelling import LabelStudioClient
from src.pipeline import DatasetManager
from src.database import DatabaseManager
from src.utils import setup_logger

logger = setup_logger(__name__)

if __name__ == "__main__":
    db = DatabaseManager()
    client = LabelStudioClient(db)
    ds = DatasetManager(db)

    annotations = client.fetch_annotations()
    ds_path = ds.create_dataset(annotations)
    logger.info(f"Created dataset at: {ds_path}")
