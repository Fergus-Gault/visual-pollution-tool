from src.database import DatabaseManager
from src.model import YoloModel
from src.utils import setup_logger, RegionManager

from .scanner import Scanner
from .inference import InferenceManager

logger = setup_logger(__name__)


class Pipeline:
    def __init__(self):
        self.db = DatabaseManager()
        self.model = YoloModel()
        self.scanner = Scanner(self.db)
        self.inference_manager = InferenceManager(self.db, self.model)

    def run(self, file_path=None, args=None):
        if file_path is not None:
            self._run_file(file_path)
        else:
            self._run_args(args)

    def get_lnglat(self, city, country=None):
        return RegionManager.geolocate_city(city, country)

    def scan_region(self, region_id=None, lng=None, lat=None):
        return self.scanner.scan_region(region_id=region_id, lng=lng, lat=lat)

    def run_inference(self, region):
        self.inference_manager.run_inference(region)

    def _run_file(self, file_path):
        with open(file_path, "r") as file:
            for line in file:
                city, country = line.split(',')
                lng, lat = self.get_lnglat(city, country)
                region = self.scan_region(lng=lng, lat=lat)
                if self.model.is_loaded():
                    self.run_inference(region)

    def _run_args(self, args):
        city = args[1]
        try:
            country = args[2]
        except:
            country = None
        lng, lat = self.get_lnglat(city, country)
        region = self.scan_region(lng=lng, lat=lat)
        if self.model.is_loaded():
            self.run_inference(region)
