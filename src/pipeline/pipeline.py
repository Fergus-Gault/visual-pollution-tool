from time import perf_counter
import csv

from src.database import DatabaseManager
from src.model import YoloModel
from src.utils import setup_logger, RegionManager
from src.mapping import Mapper

from .score import Scorer
from .scanner import Scanner
from .inference import InferenceManager

logger = setup_logger(__name__)


class Pipeline:
    def __init__(self, apis=None):
        self.db = DatabaseManager()
        self.model = YoloModel()
        self.scanner = Scanner(self.db, apis=apis)
        self.mapper = Mapper(self.db)
        self.inference_manager = InferenceManager(self.db, self.model)
        self.scorer = Scorer(self.db)

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

    def scan_region(self, region_id=None, lng=None, lat=None, override=False, region_method="shape", dense_scan=False, fetch_osm=False, city=None, country=None, iso3=None, population=None, start_captured_at=None, end_captured_at=None):
        return self.scanner.scan_region(region_id=region_id, lng=lng, lat=lat, override=override, region_method=region_method, dense_scan=dense_scan, fetch_osm=fetch_osm, city=city, country=country, iso3=iso3, population=population, start_captured_at=start_captured_at, end_captured_at=end_captured_at)

    def score_region(self, region_id, method=None, save=True):
        if region_id is not None:
            score = self.scorer.score_region(region_id, method)
            if save:
                self.db.update_score(region_id, score)
            return score

    def run_inference(self, region):
        self.inference_manager.run_inference(region)

    def _run_file(self, file_path, collect_only, override, region_method, dense_scan, fetch_osm):
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if not row:
                    continue
                city = row[0].strip() if len(row) > 0 else None
                country = row[1].strip() if len(row) > 1 else None
                iso3 = row[2].strip().upper() if len(
                    row) > 2 and row[2].strip() else None
                start_captured_at = row[-2].strip() if len(
                    row) >= 4 and row[-2].strip() else None
                end_captured_at = row[-1].strip() if len(
                    row) >= 4 and row[-1].strip() else None
                self._run_region(
                    city=city,
                    country=country,
                    iso3=iso3,
                    collect_only=collect_only,
                    override=override,
                    region_method=region_method,
                    dense_scan=dense_scan,
                    fetch_osm=fetch_osm,
                    start_captured_at=start_captured_at,
                    end_captured_at=end_captured_at,
                )

    def _run_args(self, args, collect_only, override, region_method, dense_scan, fetch_osm):
        city = args[0]
        try:
            country = args[1]
        except:
            country = None
        self._run_region(
            city=city,
            country=country,
            collect_only=collect_only,
            override=override,
            region_method=region_method,
            dense_scan=dense_scan,
            fetch_osm=fetch_osm,
        )

    def _run_region(self, city, country=None, iso3=None, collect_only=False, override=False, region_method="shape", dense_scan=False, fetch_osm=False, start_captured_at=None, end_captured_at=None):
        coords = self.get_lnglat(city, country)
        if coords is None:
            return
        self._run_region_coords(lng=coords[0], lat=coords[1], city=city, country=country, iso3=iso3,
                                collect_only=collect_only, override=override,
                                region_method=region_method, dense_scan=dense_scan, fetch_osm=fetch_osm,
                                start_captured_at=start_captured_at, end_captured_at=end_captured_at)

    def _run_region_coords(self, lng, lat, city=None, country=None, iso3=None, population=None, collect_only=False, override=False, region_method="shape", dense_scan=False, fetch_osm=False, start_captured_at=None, end_captured_at=None):
        region = self.scan_region(
            lng=lng, lat=lat, override=override, region_method=region_method,
            dense_scan=dense_scan, fetch_osm=fetch_osm, city=city, country=country, iso3=iso3, population=population,
            start_captured_at=start_captured_at, end_captured_at=end_captured_at)
        if region is None:
            logger.warning(
                f"Region for {city or lng}, {country or lat} already exists. Skipping.")
            return
        region_map = self.mapper.map_region_images(region)
        self.mapper.save(region_map, region,
                         map_type="region_images", file_type="html")
        if self.model.is_loaded() and not collect_only:
            self.run_inference(region)
            detections_map = self.mapper.map_region_detections(region)
            self.score_region(region.id)
            self.mapper.save(detections_map, region,
                             map_type="region_detections", file_type="html")
