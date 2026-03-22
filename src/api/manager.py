from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests

from src.utils import setup_logger, RegionManager
from src.config import Config, PipelineConfig
from .models import BoundingBox
from .client import HTTPClient

logger = setup_logger(__name__)


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
    def _fetch_subregion(self, subregion, session=None, **kwargs):
        pass

    def _num_workers(self):
        return PipelineConfig.NUM_WORKERS

    def fetch_region(self, bbox: BoundingBox, num_subregions, dense_scan, **kwargs):
        if dense_scan:
            num_subregions = num_subregions * Config.DENSE_MULTIPLIER
        return self._fetch_subregion_points(bbox, num_subregions, **kwargs)

    def _fetch_subregion_points(self, bbox: BoundingBox, num_subregions, **kwargs):

        num_workers = self._num_workers()
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=num_workers,
            pool_maxsize=num_workers * 4,
            max_retries=0
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        subregions = RegionManager.get_subregions(
            bbox, num_subregions=num_subregions)
        logger.info(f"Generated {num_subregions} subregions for the region.")
        images = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_fetch = {
                executor.submit(self._fetch_subregion, subregion, session, **kwargs): subregion for subregion in subregions
            }
            with tqdm(total=len(subregions), desc="Fetching images from region") as pbar:
                for future in as_completed(future_to_fetch):
                    region_img = future.result()
                    images.extend(region_img)

                    pbar.update(1)
        logger.info(f"Retrieved {len(images)} images from region.")
        return images
