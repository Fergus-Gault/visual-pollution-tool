import overpass

from api import BoundingBox, ImageRequest
from utils import setup_logger
from config import OSMConfig

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

                self.api = overpass.API(endpoint=ep)
                response = self.api.get(
                    'node["name"="Edinburgh"]', verbosity="body")

                if len(response.get("features", [])) == 0:
                    raise Exception("Failed to connect to endpoint.")

                self.endpoint = ep
                logger.info(f"Successfully connected to endpoint.")
                return

            except Exception as e:
                logger.warning(f"Failed to connect to {ep}: {e}")
                continue

    def fetch_region(self, bbox: BoundingBox):
        query = ImageRequest(bbox).to_osm_params()
        return self.api.get(query)

    @staticmethod
    def extract_name(element):
        properties = element.get('properties', {})
        if properties:
            return next(iter(properties.values()))
        return ""
