from sqlalchemy.exc import IntegrityError
from api import OSMFeatureClassifier
from typing import List
from database import DatabaseManager, Image
from api import KartaviewAPI, MapillaryAPI, OSMApi, APIManager, ImageStoreMetadata, BoundingBox
from utils import setup_logger, RegionManager
from config import PipelineConfig
import requests
from PIL import Image as PILImage
from io import BytesIO

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
            self._delete_and_rescan(region_id)
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
                params = ImageStoreMetadata.convert_data(img_data, region, api)
                if params is None:
                    continue
                params_list.append(params)
                urls.append(params['url'])

            params_list = self._fetch_dimensions(params_list)

            images_to_add = []
            for params in params_list:
                image = self._create_image(params)
                images_to_add.append(image)

            try:
                self.db.session.add_all(images_to_add)
                self.db.session.commit()
                stored_count += len(images_to_add)
            except IntegrityError as e:
                logger.warning(f"Bulk insert failed: {e}")
                self.db.session.rollback()
                for params in params_list:
                    image = self.db.add_image(**params)
                    if image is not None:
                        stored_count += 1

        return stored_count

    def _create_image(self, params):
        return Image(
            region=params['region'],
            lng=params['lng'],
            lat=params['lat'],
            id_from_source=params['id_from_source'],
            source_captured_at=params['source_captured_at'],
            url=params['url'],
            source=params['source'],
            width=params['width'],
            height=params['height']
        )

    def _fetch_dimensions(self, params_list):
        cleaned_params = []
        for param in params_list:
            url = param['url']
            try:
                response = requests.get(url, timeout=0.5, stream=True)
                response.raise_for_status()

                img = PILImage.open(BytesIO(response.content))
                width, height = img.size
                param['width'] = width
                param['height'] = height
                cleaned_params.append(param)
            except Exception:
                continue

        return cleaned_params

    def _delete_and_rescan(self, region_id):
        region = self.db.get_region(region_id)
        bbox = BoundingBox(region.min_lng, region.min_lat,
                           region.max_lng, region.max_lat)
        lng, lat = RegionManager.get_region_mid(bbox)

        images = self.db.get_images_by_region(region_id)
        for image in images:
            self.db.delete_image(image.id)
        detections = self.db.get_detections_by_region(region_id)
        for det in detections:
            self.db.delete_detection(det.id)
        self.db.delete_region(region_id)

        self.scan_region(lng=lng, lat=lat)

    def _get_or_create_region(self, region_id, lng, lat):
        if region_id is not None:
            region = self.db.get_region(region_id)
            if region is not None:
                # We return none to indicate that the region already exists
                return None

        elif lng is not None and lat is not None:
            bbox = RegionManager.get_region_bbox(lng, lat)
            region = self.db.add_region(bbox)
            return region

        else:
            raise Exception(
                f"Tried to create a region where both region_id and lng and lat are None")
