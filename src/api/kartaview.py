from api import APIManager, BoundingBox, ImageRequest, ImageMetadata
from dotenv import load_dotenv
from config import KartaviewConfig, Config
from utils import setup_logger

logger = setup_logger(__name__)


class KartaviewAPI(APIManager):
    def __init__(self):
        self.access_token = load_dotenv().get("KARTAVIEW_ACCESS_TOKEN")
        if not self.access_token:
            logger.warning("Kartaview access token not found, continuing...")
        headers = {
            "Authorization": f"Bearer {self.access_token}"} if self.access_token else {}
        super().__init__(base_url=KartaviewConfig.BASE_URL, default_headers=headers)

    def send_request(self, endpoint, params):
        return self.http_client.get(endpoint, params)

    def fetch_region(self, bbox, num_points=Config.DEFAULT_POINTS, num_subregions=Config.DEFAULT_SUBREGIONS, source=None):
        return super().fetch_region(bbox, num_points, num_subregions, source="kartaview")

    def _fetch_subregion(self, subregion: BoundingBox):
        params = ImageRequest(subregion).to_kartaview_params()

        try:
            response = self.send_request("photo/", params=params)
        except Exception as e:
            logger.warning(f"Request failed for subregion: {e}")
            return []

        raw_data = response.get("result", {}).get("data", [])

        normalised_data = [ImageMetadata.from_kartaview(
            img).to_dict() for img in raw_data]

        logger.info(
            f"Total {len(normalised_data)} from subregion {subregion.to_str()}.")
        return normalised_data
