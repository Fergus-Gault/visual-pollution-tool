from abc import ABC, abstractmethod
from utils import setup_logger, load_config, RegionManager
from config import Config
from api import BoundingBox, HTTPClient

logger = setup_logger(__name__)
config = load_config()


class APIManager(ABC):
    def __init__(self, base_url, default_headers):
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.http_client = HTTPClient(
            base_url=base_url, headers=default_headers)

    @abstractmethod
    def send_request(self, endpoint, params=None):
        pass

    @abstractmethod
    def _fetch_subregion(self, subregion, **kwargs):
        pass

    def fetch_region(self, bbox, num_points=Config.DEFAULT_POINTS, num_subregions=Config.DEFAULT_SUBREGIONS, source=None):
        if source == "kartaview":
            return self._fetch_random_points(bbox, num_points)
        else:
            return self._fetch_subregion_points(bbox, num_subregions)

    def _fetch_subregion_points(self, bbox, num_subregions):
        subregions = RegionManager.get_subregions(bbox)
        logger.info(f"Generated {num_subregions} subregions for the region.")
        images = []
        for subregion in subregions:
            region_img = self._fetch_subregion(subregion)
            images.extend(region_img)
        logger.info(f"Retrieved {len(images)} images from region.")
        return images

    def _fetch_random_points(self, bbox, num_points):
        points = RegionManager.get_random_points(bbox)
        logger.info(f"Generated {num_points} random points ")
        images = []
        for lng, lat in points:
            region = BoundingBox.from_centre(lng, lat).to_tuple()
            region_img = self._fetch_subregion(region)
            images.extend(region_img)

        logger.info(f"Retrieved {len(images)} images from point.")
        return images
