from database import DatabaseManager
from model import YoloModel
from utils import setup_logger

logger = setup_logger(__name__)


class Pipeline:
    def __init__(self):
        self.db = DatabaseManager()
        self.model = YoloModel()

    def scan_region(self, region_id=None, lng=None, lat=None):
        pass
