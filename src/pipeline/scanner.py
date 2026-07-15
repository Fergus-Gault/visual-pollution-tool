from typing import List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dateutil import parser as date_parser
from dateutil.parser import ParserError
from src.database import DatabaseManager, Image, OSMFeature
from src.api import KartaviewAPI, MapillaryAPI, OSMApi, APIManager, ImageStoreMetadata, BoundingBox, OSMFeatureClassifier
from src.utils import setup_logger, RegionManager, Dimensioner
from src.config import PipelineConfig, Config, OSMConfig

logger = setup_logger(__name__)


class Scanner:
    def __init__(self, db: DatabaseManager, apis=None):
        self.db = db
        self.apis: List[APIManager] = apis or [KartaviewAPI(), MapillaryAPI()]
        self.osm = OSMApi()

    def scan_region(self, region_id=None, lng=None, lat=None, override=False, region_method="shape", dense_scan=False, fetch_osm=True, city=None, country=None, iso3=None, population=None, start_captured_at=None, end_captured_at=None):
        region, gdf = self._get_or_create_region(
            region_id, lng, lat, region_method, override=override, dense_scan=dense_scan, city=city, country=country, iso3=iso3, population=population, start_captured_at=start_captured_at, end_captured_at=end_captured_at)
        if region is None:
            return None
        effective_fetch_osm = fetch_osm and not dense_scan
        self._scan_region(region, gdf, dense_scan, effective_fetch_osm,
                          start_captured_at=start_captured_at, end_captured_at=end_captured_at)
        return region

    def scan_bbox(self, bbox, gdf=None, override=False, dense_scan=False, fetch_osm=True, city=None, country=None, iso3=None, population=None, start_captured_at=None, end_captured_at=None):
        region = self._get_or_create_region_from_bbox(
            bbox=bbox,
            override=override,
            dense_scan=dense_scan,
            city=city,
            country=country,
            iso3=iso3,
            population=population,
            start_captured_at=start_captured_at,
            end_captured_at=end_captured_at,
        )
        if region is None:
            return None
        effective_fetch_osm = fetch_osm and not dense_scan
        self._scan_region(region, gdf, dense_scan, effective_fetch_osm,
                          start_captured_at=start_captured_at, end_captured_at=end_captured_at)
        return region

    def _scan_region(self, region, gdf, dense_scan, fetch_osm, start_captured_at=None, end_captured_at=None):
        region_bbox = BoundingBox(region.min_lng, region.min_lat,
                                  region.max_lng, region.max_lat)
        eligible_apis = [
            api for api in self.apis
            if not (dense_scan and isinstance(api, KartaviewAPI))
        ]

        with ThreadPoolExecutor() as executor:
            osm_future = executor.submit(
                self.osm.fetch_region, region_bbox) if (fetch_osm and self.osm.api is not None) else None
            api_futures = [(api, executor.submit(
                api.fetch_region, region_bbox, dense_scan=dense_scan,
                start_captured_at=start_captured_at, end_captured_at=end_captured_at)) for api in eligible_apis]
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

    def rescan_osm_region(self, region):
        region_bbox = BoundingBox(
            region.min_lng,
            region.min_lat,
            region.max_lng,
            region.max_lat,
        )
        if self.osm.api is None:
            logger.warning("OSM API is unavailable. Skipping OSM rescan.")
            return False
        data = self.osm.fetch_region(region_bbox)
        if data is None:
            logger.warning(f"OSM did not return any data for region {region.id}.")
            return False
        osm_success = self._store_osm_data(region, data)
        self.db.update_osm_fetched(region.id, osm_success or bool(region.osm_fetched))
        return osm_success

    def rescan_targeted_features_region(self, region):
        region_bbox = BoundingBox(
            region.min_lng,
            region.min_lat,
            region.max_lng,
            region.max_lat,
        )
        if self.osm.api is None:
            logger.warning("OSM API is unavailable. Skipping targeted OSM rescan.")
            return False
        data = self.osm.fetch_region_for_queries(
            region_bbox,
            OSMConfig.TARGETED_RESCAN_QUERIES,
        )
        if data is None:
            logger.warning(
                f"OSM did not return any targeted OSM data for region {region.id}."
            )
            return False

        filtered_features = []
        for feature in data.get('features', []):
            properties = feature.get('properties', {})
            osm_type = OSMFeatureClassifier.determine_osm_type(properties)
            if 'advertising' in properties and osm_type != 'billboard':
                filtered_features.append(feature)
                continue
            if 'barrier' in properties and osm_type == 'barrier':
                filtered_features.append(feature)
                continue
            if ('highway' in properties or 'traffic_sign' in properties) and osm_type == 'road_sign':
                filtered_features.append(feature)
                continue

        if not filtered_features:
            return False

        osm_success = self._store_osm_data(region, {'features': filtered_features})
        self.db.update_osm_fetched(region.id, osm_success or bool(region.osm_fetched))
        return osm_success

    def _extract_osm_point(self, geometry):
        geom_type = geometry.get('type')
        coordinates = geometry.get('coordinates', [])

        if geom_type == 'Point' and len(coordinates) >= 2:
            return coordinates[0], coordinates[1]

        if geom_type == 'MultiPoint' and coordinates:
            return self._average_points(coordinates)

        if geom_type == 'LineString' and coordinates:
            return self._average_points(coordinates)

        if geom_type == 'MultiLineString' and coordinates:
            points = [point for line in coordinates for point in line]
            return self._average_points(points)

        if geom_type == 'Polygon' and coordinates:
            outer_ring = coordinates[0] if coordinates else []
            return self._average_points(outer_ring)

        if geom_type == 'MultiPolygon' and coordinates:
            points = []
            for polygon in coordinates:
                if polygon:
                    points.extend(polygon[0])
            return self._average_points(points)

        return (None, None)

    def _average_points(self, points):
        valid_points = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                lng = float(point[0])
                lat = float(point[1])
            except (TypeError, ValueError):
                continue
            valid_points.append((lng, lat))

        if not valid_points:
            return (None, None)

        lng = sum(point[0] for point in valid_points) / len(valid_points)
        lat = sum(point[1] for point in valid_points) / len(valid_points)
        return (lng, lat)

    def _store_osm_data(self, region, data):
        to_add = []
        stored_osm_count = 0
        existing_keys = self.db.get_osm_feature_keys_by_region(region.id)
        pending_keys = set()
        for vp in data['features']:
            try:
                geometry = vp.get('geometry', {})
                lng, lat = self._extract_osm_point(geometry)
                if lng is None or lat is None:
                    continue

                feature_name = self.osm.extract_name(vp)
                properties = vp.get('properties', {})
                osm_id = str(vp.get('id', str(properties)))
                osm_type = OSMFeatureClassifier.determine_osm_type(properties)
                feature_key = (osm_id, osm_type)

                if feature_key in existing_keys or feature_key in pending_keys:
                    continue

                pending_keys.add(feature_key)
                to_add.append(OSMFeature(
                    region_id=region.id,
                    osm_id=osm_id,
                    osm_type=osm_type,
                    lng=lng,
                    lat=lat,
                    name=feature_name,
                ))
                stored_osm_count += 1

            except (KeyError, IndexError, TypeError) as e:
                logger.warning(
                    f"Failed to extract coordinates from OSM feature: {e}")
        if to_add:
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

        id_from_source = params['id_from_source']
        if params['region'].dense_scan:
            id_from_source = f"{id_from_source}|dense|{params['region'].id}"

        return Image(
            region_id=params['region'].id,
            lng=params['lng'],
            lat=params['lat'],
            id_from_source=id_from_source,
            source_captured_at=captured_at,
            url=params['url'],
            source=params['source'],
            width=params['width'],
            height=params['height'],
            altitude=params.get('altitude'),
            atomic_scale=params.get('atomic_scale'),
            camera_parameters=params.get('camera_parameters'),
            camera_type=params.get('camera_type'),
            compass_angle=params.get('compass_angle'),
            computed_altitude=params.get('computed_altitude'),
            computed_compass_angle=params.get('computed_compass_angle'),
            computed_rotation=params.get('computed_rotation'),
            creator_id=params.get('creator_id'),
            creator_username=params.get('creator_username'),
            exif_orientation=params.get('exif_orientation'),
            is_pano=params.get('is_pano'),
            camera_make=params.get('camera_make'),
            camera_model=params.get('camera_model'),
            on_foot=params.get('on_foot'),
            organization_id=params.get('organization_id'),
            organization_name=params.get('organization_name'),
            organization_slug=params.get('organization_slug'),
            quality_score=params.get('quality_score'),
            sequence=params.get('sequence'),
            source_metadata=params.get('source_metadata'),
        )

    def _get_or_create_region(self, region_id, lng, lat, region_method, override=False, dense_scan=False, city=None, country=None, iso3=None, population=None, start_captured_at=None, end_captured_at=None):
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
            iso3 = iso3 or existing.iso3
            population = population or existing.population
            start_captured_at = start_captured_at or existing.start_captured_at
            end_captured_at = end_captured_at or existing.end_captured_at
            dense_scan = existing.dense_scan if dense_scan is None else dense_scan
            self.db.delete_region(region_id)
        elif lng is None or lat is None:
            raise Exception(
                "Tried to create a region where both region_id and lng and lat are None")

        bbox = RegionManager.get_region_bbox(lng, lat)
        if not override:
            initial_region_name = self.db.build_region_name(
                bbox,
                dense_scan=dense_scan,
                start_captured_at=start_captured_at,
                end_captured_at=end_captured_at,
            )
            initial_existing = self.db.get_region_by_name(initial_region_name)
            if initial_existing is not None:
                return None, None
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
        final_region_name = self.db.build_region_name(
            bbox,
            dense_scan=dense_scan,
            start_captured_at=start_captured_at,
            end_captured_at=end_captured_at,
        )
        existing = self.db.get_region_by_name(final_region_name)
        if existing is not None:
            if not override:
                return None, None
            self.db.delete_region(existing.id)
        logger.info(f"Adding region for {city}, {country}.")
        region = self.db.add_region(
            bbox,
            city,
            country,
            iso3=iso3,
            population=population,
            dense_scan=dense_scan,
            start_captured_at=start_captured_at,
            end_captured_at=end_captured_at,
        )
        if region is None:
            region = self.db.get_region_by_name(final_region_name)
        return region, gdf

    def _get_or_create_region_from_bbox(self, bbox, override=False, dense_scan=False, city=None, country=None, iso3=None, population=None, start_captured_at=None, end_captured_at=None):
        if city is None and country is None:
            geocoded_city, geocoded_country = RegionManager.geolocate_bbox(bbox)
            city = city or geocoded_city
            country = country or geocoded_country
        if city is None and country is None:
            return None

        region_name = self.db.build_region_name(
            bbox,
            dense_scan=dense_scan,
            start_captured_at=start_captured_at,
            end_captured_at=end_captured_at,
        )
        existing = self.db.get_region_by_name(region_name)
        if existing is not None:
            if not override:
                return None
            self.db.delete_region(existing.id)

        logger.info(f"Adding region for {city}, {country}.")
        region = self.db.add_region(
            bbox,
            city,
            country,
            iso3=iso3,
            population=population,
            dense_scan=dense_scan,
            start_captured_at=start_captured_at,
            end_captured_at=end_captured_at,
        )
        if region is None:
            region = self.db.get_region_by_name(region_name)
        return region
