from typing import List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dateutil import parser as date_parser
from dateutil.parser import ParserError
from src.database import DatabaseManager, Image, OSMFeature
from src.api import KartaviewAPI, MapillaryAPI, OSMApi, APIManager, ImageStoreMetadata, BoundingBox, OSMFeatureClassifier
from src.utils import setup_logger, RegionManager, Dimensioner
from src.config import PipelineConfig, Config

logger = setup_logger(__name__)


class Scanner:
    def __init__(self, db: DatabaseManager, apis=None):
        self.db = db
        self.apis: List[APIManager] = apis or [KartaviewAPI(), MapillaryAPI()]
        self.osm = OSMApi()

    def scan_region(self, region_id=None, lng=None, lat=None, override=False, region_method="shape", dense_scan=False, fetch_osm=True, city=None, country=None, population=None):
        region, gdf = self._get_or_create_region(
            region_id, lng, lat, region_method, override=override, city=city, country=country, population=population)
        if region is None:
            return None
        self._scan_region(region, gdf, dense_scan, fetch_osm)
        return region

    def _scan_region(self, region, gdf, dense_scan, fetch_osm):
        region_bbox = BoundingBox(region.min_lng, region.min_lat,
                                  region.max_lng, region.max_lat)

        with ThreadPoolExecutor() as executor:
            osm_future = executor.submit(
                self.osm.fetch_region, region_bbox) if (fetch_osm and self.osm.api is not None) else None
            api_futures = [(api, executor.submit(
                api.fetch_region, region_bbox, dense_scan=dense_scan)) for api in self.apis]
            raw_osm = osm_future.result() if osm_future else None
            api_results = [(api, future.result())
                           for api, future in api_futures]

        if raw_osm is not None:
            osm_success = self._store_osm_data(region, raw_osm)
        else:
            osm_success = False
        self.db.update_osm_fetched(region.id, osm_success)

        image_count = 0
        for api, api_images in api_results:
            filtered = self._filter_images(region_bbox, api_images, gdf)
            api_image_count = self._store_images(filtered, region, api)
            image_count += api_image_count
            logger.info(
                f"Fetched and stored {api_image_count} from {api.__class__.__name__} for {region.city}, {region.country}.")
        logger.info(
            f"Total {image_count} images fetched for {region.city}, {region.country}.")

    def _fetch_osm_data(self, region, region_bbox):
        logger.info("Fetching OSM data")
        data = self.osm.fetch_region(region_bbox)
        if data is None:
            logger.warning("OSM did not return any data.")
            return
        self._store_osm_data(region, data)

    def _store_osm_data(self, region, data):
        to_add = []
        stored_osm_count = 0
        for vp in data['features']:
            try:
                geometry = vp.get('geometry', {})
                geom_type = geometry.get('type')
                coordinates = geometry.get('coordinates', [])

                if geom_type == 'Point' and len(coordinates) >= 2:
                    lng, lat = coordinates[0], coordinates[1]
                    feature_name = self.osm.extract_name(vp)

                    properties = vp.get('properties', {})
                    osm_id = vp.get('id', str(properties))
                    osm_type = OSMFeatureClassifier.determine_osm_type(
                        properties)
                    to_add.append(OSMFeature(region_id=region.id, osm_id=str(
                        osm_id), osm_type=osm_type, lng=lng, lat=lat, name=feature_name))

                    stored_osm_count += 1

            except (KeyError, IndexError, TypeError) as e:
                logger.warning(
                    f"Failed to extract coordinates from OSM feature: {e}")
        self.db.add_many_osm_features(to_add)
        logger.info(f"Stored {stored_osm_count} OSM features.")
        return stored_osm_count > 0

    def _filter_images(self, region_bbox, images, gdf):
        filtered_images = []
        for image in images:
            geometry = image.get("geometry", {})
            coords = geometry.get("coordinates", [None, None])
            try:
                lng = float(coords[0])
                lat = float(coords[1])
            except (TypeError, ValueError, IndexError):
                continue

            if gdf is None:
                if (region_bbox.min_lng <= lng <= region_bbox.max_lng
                        and region_bbox.min_lat <= lat <= region_bbox.max_lat):
                    filtered_images.append(image)
            else:
                if RegionManager.point_in_city(lng, lat, gdf):
                    filtered_images.append(image)
        return filtered_images

    def _store_images(self, images, region, api):
        if not images:
            return 0
        stored_count = 0
        chunk_size = PipelineConfig.IMAGE_STORAGE_CHUNK_SIZE
        session = Dimensioner._make_session()

        for start in range(0, len(images), chunk_size):
            chunk = images[start:start + chunk_size]
            params_list = []
            for img_data in chunk:
                params = ImageStoreMetadata.convert_data(img_data, region, api)
                if params is None:
                    continue
                params_list.append(params)

            params_list = Dimensioner.update_dimensions(
                params_list, session=session)

            images_to_add = []
            for params in params_list:
                image = self._create_image(params)
                if image is not None:
                    images_to_add.append(image)

            self.db.add_many_images(images_to_add)
            stored_count += len(images_to_add)

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
            region_id=params['region'].id,
            lng=params['lng'],
            lat=params['lat'],
            id_from_source=params['id_from_source'],
            source_captured_at=captured_at,
            url=params['url'],
            source=params['source'],
            width=params['width'],
            height=params['height']
        )

    def _get_or_create_region(self, region_id, lng, lat, region_method, override=False, city=None, country=None, population=None):
        gdf = None
        if region_id is not None:
            existing = self.db.get_region(region_id)
            if existing is None or not override:
                return None, None
            # override=True: extract coords from existing, delete it, then recreate
            bbox = BoundingBox(existing.min_lng, existing.min_lat,
                               existing.max_lng, existing.max_lat)
            lng, lat = RegionManager.get_region_mid(bbox)
            city = city or existing.city
            country = country or existing.country
            population = population or existing.population
            self.db.delete_region(region_id)
        elif lng is None or lat is None:
            raise Exception(
                "Tried to create a region where both region_id and lng and lat are None")

        bbox = RegionManager.get_region_bbox(lng, lat)
        if city is None or country is None:
            geocoded_city, geocoded_country = RegionManager.geolocate_bbox(
                bbox)
            city = city or geocoded_city
            country = country or geocoded_country
        if city is None and country is None:
            return None, None
        if region_method == "shape":
            gdf = RegionManager.get_shape_file(city, country)
            if gdf is not None:
                if lng is not None and lat is not None and not RegionManager.point_in_city(lng, lat, gdf):
                    gdf = None
                else:
                    shape_bbox = RegionManager.bbox_from_shape(gdf)
                    shape_area = (shape_bbox.max_lng - shape_bbox.min_lng) * \
                        (shape_bbox.max_lat - shape_bbox.min_lat)
                    if shape_area <= Config.MAX_SHAPE_BBOX_AREA:
                        bbox = shape_bbox
        existing = self.db.get_region_by_name(
            RegionManager.generate_region_name(bbox))
        if existing is not None:
            if not override:
                return None, None
            self.db.delete_region(existing.id)
        logger.info(f"Adding region for {city}, {country}.")
        region = self.db.add_region(bbox, city, country, population=population)
        if region is None:
            region = self.db.get_region_by_name(
                RegionManager.generate_region_name(bbox))
        return region, gdf
