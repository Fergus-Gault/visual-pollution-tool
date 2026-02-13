from concurrent.futures import ThreadPoolExecutor, as_completed
from api import HTTPClient, OSMFeatureClassifier
from typing import List
from database import DatabaseManager, Region, Image
from api import KartaviewAPI, MapillaryAPI, OSMApi, APIManager, ImageMetadata
from utils import setup_logger, RegionManager
from config import PipelineConfig

logger = setup_logger(__name__)


class Scanner:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.apis: List[APIManager] = [KartaviewAPI(), MapillaryAPI()]
        self.osm = OSMApi()

    def scan_region(self, region_id=None, lng=None, lat=None, override=False):
        region = self._get_or_create_region(region_id, lng, lat)
        if not region and not override:
            return
        elif override:
            # TODO: self._delete_and_rescan
            pass
        else:
            self._scan_region(region)

    def _scan_region(self, region):
        image_count = 0
        self._fetch_osm_data(region.bbox)
        for api in self.apis:
            api_images = api.fetch_region(region.bbox)
            filter_images = self._filter_images(api_images)
            api_image_count = self._store_images(filter_images, region, api)
            image_count += api_image_count
            logger.info(
                f"Fetched and stored {api_image_count} from {api.__class__.__name__} for region {region.name}.")
        logger.info(
            f"Total {image_count} images fetched for region {region.name}.")

    def _fetch_osm_data(self, region):
        data = self.osm.fetch_region(region)
        if data is None:
            logger.warning("OSM did not return any data.")
            return []

        osm_points = []
        stored_osm_count = 0

        for vp in data['features']:
            try:
                geometry = vp.get('geometry', {})
                geom_type = geometry.get('type')
                coordinates = geometry.get('coordinates', [])

                if geom_type == 'Point' and len(coordinates) >= 2:
                    lng, lat = coordinates[0], coordinates[1]
                    feature_name = self.osm.extract_name(vp)
                    osm_points.append((lng, lat, feature_name))

                    properties = vp.get('properties', {})
                    osm_id = vp.get('id', str(properties))
                    osm_type = OSMFeatureClassifier.determine_osm_type(
                        properties)
                    self.db.add_osm_feature(region_id=region.id, osm_id=str(
                        osm_id), osm_type=osm_type, lng=lng, lat=lat, name=feature_name)

                    stored_osm_count += 1

            except (KeyError, IndexError, TypeError) as e:
                logger.warning(
                    f"Failed to extract coordinates from OSM feature: {e}")

        logger.info(f"Stored {stored_osm_count} OSM features.")
        return osm_points

    def _filter_images(self, region, images):
        filtered_images = []
        for image in images:
            if (region.bbox.min_lng <= image.lng <= region.bbox.max_lng
                    and region.bbox.min_lat <= image.lat <= region.bbox.max_lat):
                filtered_images.extend(image)
        return filtered_images

    def _store_images(self, images, region, api):
        if not images:
            return 0
        stored_count = 0
        chunk_size = PipelineConfig.IMAGE_STORAGE_CHUNK_SIZE

        for start in range(0, len(images), chunk_size):
            chunk = images[start:start + chunk_size]
            params_list = []
            urls = []
            for img_data in chunk:
                geometry = img_data.get('computed_geometry', {})
                coords = geometry.get('coordinates', [None, None])
                lng, lat = coords[0], coords[1]
                if lng is None or lat is None:
                    continue

                # TODO: Parse and add params
