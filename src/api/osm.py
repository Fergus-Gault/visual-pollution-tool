import overpass
import json

from .models import BoundingBox, ImageRequest
from src.utils import setup_logger, RegionManager, RateLimiter
from src.config import OSMConfig, PipelineConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
logger = setup_logger(__name__)


class OSMApi:
    def __init__(self, rate_limiter: RateLimiter = None):
        self.endpoint = None
        self.rate_limiter = rate_limiter or RateLimiter(
            max_calls=PipelineConfig.OSM_RATE_LIMIT
        )
        self.api = self._connect()
        if self.api is None:
            logger.warning(
                "Failed to connect to any Overpass API. OSM collection will be skipped.")

    def _connect(self):
        for ep in OSMConfig.OSM_ENDPOINTS:
            try:
                logger.info(f"Attempting to connect to endpoint: {ep}")

                api = overpass.API(
                    endpoint=ep,
                    headers=OSMConfig.HEADERS,
                    timeout=OSMConfig.CONNECT_TIMEOUT,
                )
                response = api.get(
                    OSMConfig.CONNECT_TEST_QUERY,
                    responseformat="json",
                    build=False,
                )

                if not self._is_valid_connect_response(response):
                    raise Exception("Failed to connect to endpoint.")

                self.endpoint = ep
                logger.info(f"Successfully connected to endpoint.")
                return api

            except Exception as e:
                logger.warning(f"Failed to connect to {ep}: {e}")
                continue

    def _is_valid_connect_response(self, response):
        if isinstance(response, dict):
            return len(response.get("elements", [])) > 0

        if isinstance(response, str):
            text = response.strip()
            if not text:
                return False
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return "<osm" in text or "<remark" in text or "elements" in text
            return len(parsed.get("elements", [])) > 0

        return False

    def fetch_region(self, bbox: BoundingBox):
        return self.fetch_region_for_queries(bbox, OSMConfig.OSM_QUERIES)

    def fetch_region_for_queries(self, bbox: BoundingBox, queries):
        if self.api is None:
            return None
        subregions = RegionManager.get_subregions(
            bbox, OSMConfig.OSM_SUBREGIONS)
        data = {}
        data["features"] = []
        worker_count = min(len(subregions), OSMConfig.OSM_MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_fetch = {
                executor.submit(self._fetch_subregion, subregion, queries): subregion for subregion in subregions
            }
            with tqdm(total=len(subregions), desc="Fetching OSM data") as pbar:
                for future in as_completed(future_to_fetch):
                    osm_subregion = future.result()
                    if osm_subregion is not None:
                        data["features"].extend(
                            osm_subregion.get("features", []))
                    pbar.update(1)
        return data

    def _fetch_subregion(self, bbox: BoundingBox, queries=None):
        query = ImageRequest(bbox).to_osm_params(queries=queries)
        self.api.timeout = OSMConfig.QUERY_TIMEOUT
        for _ in range(OSMConfig.RETRIES):
            try:
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire()
                points = self.api.get(query)
                return points
            except Exception as e:
                logger.debug(f"OSM query failed, retrying: {e}")
        return None

    @staticmethod
    def extract_name(element):
        properties = element.get('properties', {})
        if properties:
            return next(iter(properties.values()))
        return ""


class OSMFeatureClassifier:
    BILLBOARD_TYPES = {
        'billboard',
        'poster_box',
        'column',
        'poster_panel',
        'ad_column',
        'totem',
        'signboard',
        'screen',
        'digital_display',
        'videowall',
    }
    SHOP_SIGN_TYPES = {
        'sign',
        'board',
        'banner',
        'flag',
        'wall_painting',
    }
    MOBILE_AD_TYPES = {
        'mobile',
        'vehicle',
        'transport',
    }
    BARRIER_TYPES = {
        'bollard',
        'wedge',
        'barrier_board',
        'jersey_barrier',
    }
    ROAD_SIGN_HIGHWAY_TYPES = {
        'give_way',
        'stop',
        'milestone',
        'motorway_junction',
        'speed_display',
    }

    @staticmethod
    def classify_advertising_type(ad_type: str) -> str:
        ad_type = str(ad_type).strip().lower()
        if ad_type in OSMFeatureClassifier.BILLBOARD_TYPES:
            return 'billboard'
        if ad_type in OSMFeatureClassifier.SHOP_SIGN_TYPES:
            return 'shop_sign'
        if ad_type in OSMFeatureClassifier.MOBILE_AD_TYPES:
            return 'mobile_advertisement'
        return 'advertising'

    @staticmethod
    def classify_barrier_type(barrier_type: str) -> str:
        barrier_type = str(barrier_type).strip().lower()
        if barrier_type in OSMFeatureClassifier.BARRIER_TYPES:
            return 'barrier'
        return 'other'

    @staticmethod
    def classify_highway_type(highway_type: str) -> str:
        highway_type = str(highway_type).strip().lower()
        if highway_type in OSMFeatureClassifier.ROAD_SIGN_HIGHWAY_TYPES:
            return 'road_sign'
        if highway_type == 'street_lamp':
            return 'street_light'
        if highway_type == 'traffic_signals':
            return 'traffic_light'
        return 'other'

    @staticmethod
    def determine_osm_type(properties: dict) -> str:
        if 'amenity' in properties:
            amenity = properties['amenity']
            if amenity in ['waste_basket', 'waste_disposal', 'recycling']:
                return 'bin'

        if 'power' in properties:
            power_type = properties['power']
            if power_type in ['pole', 'tower', 'portal', 'catenary_mast']:
                return 'power'

        if 'advertising' in properties:
            return OSMFeatureClassifier.classify_advertising_type(
                properties['advertising']
            )

        if 'barrier' in properties:
            return OSMFeatureClassifier.classify_barrier_type(
                properties['barrier']
            )

        if 'highway' in properties:
            return OSMFeatureClassifier.classify_highway_type(
                properties['highway']
            )

        if 'traffic_sign' in properties:
            return 'road_sign'

        return 'other'
