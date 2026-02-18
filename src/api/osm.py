import overpass

from .models import BoundingBox, ImageRequest
from src.utils import setup_logger
from src.config import OSMConfig

logger = setup_logger(__name__)


class OSMApi:
    def __init__(self):
        self.endpoint = None
        self.api = self._connect()
        if self.api is None:
            raise Exception("Failed to connect to any Overpass API.")

    def _connect(self):
        for ep in OSMConfig.OSM_ENDPOINTS:
            try:
                logger.info(f"Attempting to connect to endpoint: {ep}")

                api = overpass.API(endpoint=ep)
                response = api.get(
                    'node["name"="Edinburgh"]', verbosity="body")

                if len(response.get("features", [])) == 0:
                    raise Exception("Failed to connect to endpoint.")

                self.endpoint = ep
                logger.info(f"Successfully connected to endpoint.")
                return api

            except Exception as e:
                logger.warning(f"Failed to connect to {ep}: {e}")
                continue

    def fetch_region(self, bbox: BoundingBox):
        query = ImageRequest(bbox).to_osm_params()
        for _ in range(OSMConfig.RETRIES):
            try:
                points = self.api.get(query)
                return points
            except Exception as e:
                logger.warning(f"OSM query failed, retrying: {e}")
        return None

    @staticmethod
    def extract_name(element):
        properties = element.get('properties', {})
        if properties:
            return next(iter(properties.values()))
        return ""


class OSMFeatureClassifier:
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
            ad_type = properties['advertising']
            if ad_type in ['billboard', 'poster_box', 'column']:
                return 'billboard'

        if 'barrier' in properties:
            barrier_type = properties['barrier']
            if barrier_type in ['block', 'bollard', 'jersey_barrier']:
                return 'barrier'

        if 'highway' in properties:
            if properties['highway'] == 'street_lamp':
                return 'street_light'
            elif properties['highway'] == 'traffic_signals':
                return 'traffic_light'

        if 'traffic_sign' in properties:
            return 'traffic_sign'

        return 'other'
