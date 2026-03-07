from dotenv import dotenv_values
from .manager import APIManager
from .models import BoundingBox, ImageRequest, ImageMetadata
from src.config import KartaviewConfig, Config, PipelineConfig
from src.utils import setup_logger, RateLimiter
import time

logger = setup_logger(__name__)


class KartaviewAPI(APIManager):
    def __init__(self, access_token=None, rate_limiter: RateLimiter = None):
        self.access_token = access_token or dotenv_values(
            Config.ENV_PATH).get("KARTAVIEW_ACCESS_TOKEN")
        if not self.access_token:
            logger.warning("Kartaview access token not found, continuing...")
        self.rate_limiter = rate_limiter
        headers = {
            "Authorization": f"Bearer {self.access_token}"} if self.access_token else {}
        super().__init__(base_url=KartaviewConfig.BASE_URL, default_headers=headers)

    def send_request(self, endpoint, params=None, session=None):
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        return self.http_client.get(endpoint, params=params, session=session, headers=self.default_headers)

    def fetch_region(self, bbox, num_subregions=KartaviewConfig.SUBREGIONS, dense_scan=False):
        return super().fetch_region(bbox, num_subregions, dense_scan)

    def _num_workers(self):
        return PipelineConfig.KARTAVIEW_WORKERS

    def _fetch_subregion(self, subregion: BoundingBox, session=None):
        params = ImageRequest(subregion).to_kartaview_params()

        try:
            response = self.send_request(
                "photo/", params=params, session=session)
        except Exception as e:
            logger.debug(f"Request failed for subregion: {e}")
            return []

        raw_data = response.get("result", {}).get("data", [])

        normalised_data = [ImageMetadata.from_kartaview(
            img).to_dict() for img in raw_data]
        time.sleep(KartaviewConfig.DEFAULT_DELAY)
        return normalised_data
