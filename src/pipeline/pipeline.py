from time import perf_counter

from src.database import DatabaseManager
from src.model import YoloModel
from src.utils import setup_logger, RegionManager
from src.mapping import Mapper

from .scanner import Scanner
from .inference import InferenceManager

logger = setup_logger(__name__)


class Pipeline:
    def __init__(self):
        self.db = DatabaseManager()
        self.model = YoloModel()
        self.scanner = Scanner(self.db)
        self.mapper = Mapper(self.db)
        self.inference_manager = InferenceManager(self.db, self.model)

    def run(self, file_path=None, args=None, collect_only=False, override=False, region_method="shape", dense_scan=False, fetch_osm=False):
        start = perf_counter()
        if file_path is not None:
            self._run_file(file_path, collect_only, override,
                           region_method, dense_scan, fetch_osm)
        else:
            self._run_args(args, collect_only, override,
                           region_method, dense_scan, fetch_osm)
        end = perf_counter()
        logger.info(f"Completed pipeline in {(end-start):.2f} seconds.")

    def get_lnglat(self, city, country=None):
        return RegionManager.geolocate_city(city, country)

    def scan_region(self, region_id=None, lng=None, lat=None, override=False, region_method="shape", dense_scan=False, fetch_osm=False):
        return self.scanner.scan_region(region_id=region_id, lng=lng, lat=lat, override=override, region_method=region_method, dense_scan=dense_scan, fetch_osm=fetch_osm)

    def run_inference(self, region):
        self.inference_manager.run_inference(region)

    def _run_file(self, file_path, collect_only, override, region_method, dense_scan, fetch_osm):
        with open(file_path, "r") as file:
            for line in file:
                country = None
                try:
                    city, country = line.split(',')
                except:
                    city = line
                self._run_region(city, country, collect_only,
                                 override, region_method, dense_scan, fetch_osm)

    def _run_args(self, args, collect_only, override, region_method, dense_scan, fetch_osm):
        city = args[0]
        try:
            country = args[1]
        except:
            country = None
        self._run_region(city, country, collect_only,
                         override, region_method, dense_scan, fetch_osm)

    def _run_region(self, city, country=None, collect_only=False, override=False, region_method="shape", dense_scan=False, fetch_osm=False):
        coords = self.get_lnglat(city, country)
        if coords is None:
            return
        region = self.scan_region(
            lng=coords[0], lat=coords[1], override=override, region_method=region_method, dense_scan=dense_scan, fetch_osm=fetch_osm)
        if region is None:
            logger.warning(
                f"Region for {city.strip()} already exists. Skipping.")
            return
        region_map = self.mapper.map_region_images(region)
        self.mapper.save(region_map, region,
                         map_type="region_images", file_type="html")
        if self.model.is_loaded() and not collect_only:
            self.run_inference(region)
            detections_map = self.mapper.map_region_detections(region)
            self.mapper.save(detections_map, region,
                             map_type="region_detections", file_type="html")
