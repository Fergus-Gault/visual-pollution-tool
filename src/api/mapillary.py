import time
from dotenv import dotenv_values
from .manager import APIManager
from .models import BoundingBox, ImageRequest, ImageMetadata
from src.config import MapillaryConfig, Config
from src.utils import setup_logger

logger = setup_logger(__name__)


class MapillaryAPI(APIManager):
    def __init__(self):
        self.access_token = dotenv_values(
            Config.ENV_PATH).get("MAPILLARY_ACCESS_TOKEN")
        if not self.access_token:
            raise Exception("Mapillary access token not found.")
        super().__init__(base_url=MapillaryConfig.BASE_URL, default_headers={})

    def send_request(self, endpoint, params=None):
        params["access_token"] = self.access_token
        return self.http_client.get(endpoint, params=params)

    def fetch_region(self, bbox, num_points=Config.DEFAULT_POINTS, num_subregions=Config.DEFAULT_SUBREGIONS):
        return super().fetch_region(bbox, num_points, num_subregions, source="mapillary")

    def _fetch_subregion(self, subregion: BoundingBox, **kwargs):
        params = ImageRequest(subregion).to_mapillary_params()

        subregion_images = []
        next_cursor = None
        while True:
            if next_cursor:
                params["after"] = next_cursor
            try:
                response = self.send_request("images", params=params)
                if response is None:
                    logger.info("No response from request")
            except Exception as e:
                logger.warning(f"Request failed for subregion: {e}")
                break
            data = response.get("data", [])

            normalised_images = [ImageMetadata.from_mapillary(
                img).to_dict() for img in data]

            subregion_images.extend(normalised_images)

            next_cursor = response.get("paging", {}).get(
                "cursors", {}).get("after")
            if not next_cursor or not data:
                break
            time.sleep(MapillaryConfig.DEFAULT_DELAY)

        logger.info(
            f"Total {len(subregion_images)} from subregion {subregion.to_str()}.")
        return subregion_images
