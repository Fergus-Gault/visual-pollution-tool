from sqlalchemy.exc import IntegrityError
from typing import List
from datetime import datetime
from dateutil import parser as date_parser
from dateutil.parser import ParserError
from src.database import DatabaseManager, Image
from src.api import KartaviewAPI, MapillaryAPI, OSMApi, APIManager, ImageStoreMetadata, BoundingBox, OSMFeatureClassifier
from src.utils import setup_logger, RegionManager, Dimensioner
from src.config import PipelineConfig

logger = setup_logger(__name__)


class Scanner:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.apis: List[APIManager] = [KartaviewAPI(), MapillaryAPI()]
        self.osm = OSMApi()

    def scan_region(self, region_id=None, lng=None, lat=None, override=False):
        region = self._get_or_create_region(region_id, lng, lat)
        if region is None and not override:
            return None
        elif override:
            self._delete_and_rescan(region_id)
        else:
            self._scan_region(region)
        return region

    def _scan_region(self, region):
        image_count = 0
        region_bbox = BoundingBox(region.min_lng, region.min_lat,
                                  region.max_lng, region.max_lat)
        self._fetch_osm_data(region, region_bbox)
        for api in self.apis:
            api_images = api.fetch_region(region_bbox)
            filter_images = self._filter_images(region_bbox, api_images)
            api_image_count = self._store_images(filter_images, region, api)
            image_count += api_image_count
            logger.info(
                f"Fetched and stored {api_image_count} from {api.__class__.__name__} for region {region.name}.")
        logger.info(
            f"Total {image_count} images fetched for region {region.name}.")

    def _fetch_osm_data(self, region, region_bbox):
        data = self.osm.fetch_region(region_bbox)
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

    def _filter_images(self, region_bbox, images):
        filtered_images = []
        for image in images:
            geometry = image.get("geometry", {})
            coords = geometry.get("coordinates", [None, None])
            try:
                lng = float(coords[0])
                lat = float(coords[1])
            except (TypeError, ValueError, IndexError):
                continue

            if (region_bbox.min_lng <= lng <= region_bbox.max_lng
                    and region_bbox.min_lat <= lat <= region_bbox.max_lat):
                filtered_images.append(image)
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

            params_list = Dimensioner.update_dimensions(params_list)

            images_to_add = []
            for params in params_list:
                image = self._create_image(params)
                if image is not None:
                    images_to_add.append(image)

            try:
                self.db.session.add_all(images_to_add)
                self.db.session.commit()
                stored_count += len(images_to_add)
            except IntegrityError:
                self.db.session.rollback()
                for params in params_list:
                    image = self.db.add_image(**params)
                    if image is not None:
                        stored_count += 1

        return stored_count

    def _create_image(self, params):
        source_captured_at = params['source_captured_at']
        if isinstance(source_captured_at, int):
            captured_at = datetime.fromtimestamp(source_captured_at / 1000.0)
        elif isinstance(source_captured_at, str):
            try:
                captured_at = date_parser.parse(source_captured_at)
            except (ParserError, ValueError, TypeError):
                return None
        elif isinstance(source_captured_at, datetime):
            captured_at = source_captured_at
        else:
            return None

        return Image(
            region=params['region'],
            lng=params['lng'],
            lat=params['lat'],
            id_from_source=params['id_from_source'],
            source_captured_at=captured_at,
            url=params['url'],
            source=params['source'],
            width=params['width'],
            height=params['height']
        )

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
            city, country = RegionManager.geolocate_bbox(bbox)
            logger.info(f"Adding region for {city}, {country}.")
            region = self.db.add_region(bbox, city, country)
            return region
        else:
            raise Exception(
                f"Tried to create a region where both region_id and lng and lat are None")
