from abc import ABC, abstractmethod
import random
from utils import setup_logger, load_config

logger = setup_logger(__name__)
config = load_config()


class APIManager(ABC):
    def __init__(self, access_token, base_url, default_headers):
        self.access_token = access_token
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.http_client = None  # TODO: Add HTTP client

    @abstractmethod
    def send_request(self, endpoint, params=None):
        pass

    @abstractmethod
    def _prepare_auth(self, params=None):
        pass

    @abstractmethod
    def _fetch_subregion(self, subregion, **kwargs):
        pass

    @abstractmethod
    def _fetch_point(self, lon, lat, **kwargs):
        pass

    def fetch_region(self, min_lon, min_lat, max_lon, max_lat, radius_km, img_per_point, num_points=config.DEFAULT_POINTS, num_subregions=config.DEFAULT_SUBREGIONS, method=None):
        if "random" in method:
            return self._fetch_random_points(min_lon, min_lat, max_lon, max_lat, img_per_point, radius_km, num_points)
        else:
            return self._fetch_subregion_points(min_lon, min_lat, max_lon, max_lat, img_per_point, num_subregions)

    def _fetch_subregion_points(self, min_lon, min_lat, max_lon, max_lat, img_per_point, num_subregions):
        subregions = None  # TODO: Add subregion generation
        logger.info(f"Generated {num_subregions} subregions for the region.")
        images = []
        for subregion in subregions:
            region_img = self._fetch_subregion(subregion)
            images.extend(region_img)
        logger.info(f"Retrieved {len(images)} images from region.")
        return images

    def _fetch_random_points(self, min_lon, min_lat, max_lon, max_lat, img_per_point, radius_km, num_points):
        points = None  # TODO: Add random point generation
        logger.info(f"Generated {num_points} random points ")
        images = []
        for lng, lat in points:
            region = None  # TODO: Generate bounding box from centre point
            region_img = self._fetch_subregion(region)
            if img_per_point > 0 and len(region_img) > img_per_point:
                region_img = random.sample(region_img, img_per_point)
            images.extend(region_img)

        logger.info(f"Retrieved {len(images)} images from point.")
        return images
