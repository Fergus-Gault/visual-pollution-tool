from dateutil import parser as date_parser
from dateutil.parser import ParserError
from datetime import datetime, date
import unicodedata

from sqlalchemy import create_engine, delete, select, func
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker

from src.api.models import BoundingBox
from .models import Base, Region, Image, Detection, OSMFeature
from .repos import RegionRepo, ImageRepo, DetectionRepo, OSMFeatureRepo
from src.config import DatabaseConfig
from src.utils import setup_logger, RegionManager
import random

logger = setup_logger(__name__)


class DatabaseManager:
    def __init__(self):
        db_url = DatabaseConfig.get_postgres_url()

        self.engine = create_engine(db_url, poolclass=NullPool)
        Base.metadata.create_all(self.engine)

        SessionLocal = sessionmaker(bind=self.engine)
        self.session = SessionLocal()

        self.regions = RegionRepo(self.session)
        self.images = ImageRepo(self.session)
        self.detections = DetectionRepo(self.session)
        self.osm_features = OSMFeatureRepo(self.session)

    def get_all_regions(self):
        return self.regions.get_all()

    def get_region(self, region_id):
        return self.regions.get_by_id(region_id)

    def get_region_by_point(self, lng, lat):
        return self.regions.get_by_point(lng, lat)

    def get_region_by_city_and_country(self, city, country):
        return self.regions.get_by_city_and_country(city, country)

    def get_random_images(self, region_id, num_images):
        images = self.get_images_by_region(region_id)
        k = min(len(images), num_images)
        return random.sample(images, k=k)

    def get_random_images_by_country(self, num_images: int):
        subq = (
            select(
                Image.id.label("id"),
                func.row_number().over(
                    partition_by=Region.country,
                    order_by=func.random()
                ).label("rn")
            )
            .join(Region, Image.region_id == Region.id)
            .subquery()
        )
        sampled_ids = self.session.scalars(
            select(subq.c.id).where(subq.c.rn <= num_images)
        ).all()
        return self.session.scalars(
            select(Image).where(Image.id.in_(sampled_ids))
        ).all()

    def add_region(self, bbox: BoundingBox, city=None, country=None, population=None):
        name = RegionManager.generate_region_name(bbox)
        city = unicodedata.normalize('NFC', city) if city else city
        country = unicodedata.normalize('NFC', country) if country else country
        region = Region(name=name, min_lng=bbox.min_lng, min_lat=bbox.min_lat,
                        max_lng=bbox.max_lng, max_lat=bbox.max_lat, city=city, country=country, population=population)
        self.regions.add(region)
        return region

    def get_region_by_name(self, name):
        return self.regions.get_by_name(name)

    def update_osm_fetched(self, region_id, value: bool):
        region = self.get_region(region_id)
        if region:
            region.osm_fetched = value
            self.session.commit()

    def delete_region(self, region_id):
        image_ids_subq = select(Image.id).where(
            Image.region_id == region_id).scalar_subquery()
        self.session.execute(delete(Detection).where(
            Detection.image_id.in_(image_ids_subq)))
        self.session.execute(delete(Image).where(Image.region_id == region_id))
        self.session.execute(delete(OSMFeature).where(
            OSMFeature.region_id == region_id))
        self.session.commit()
        return self.regions.delete(region_id)

    def delete_regions(self, region_ids):
        if not region_ids:
            return 0

        image_ids_subq = select(Image.id).where(
            Image.region_id.in_(region_ids)).scalar_subquery()
        self.session.execute(delete(Detection).where(
            Detection.image_id.in_(image_ids_subq)))
        self.session.execute(delete(Image).where(
            Image.region_id.in_(region_ids)))
        self.session.execute(delete(OSMFeature).where(
            OSMFeature.region_id.in_(region_ids)))
        deleted_regions = self.session.execute(
            delete(Region).where(Region.id.in_(region_ids))
        )
        self.session.commit()
        return deleted_regions.rowcount or 0

    def get_region_bbox(self, region_id):
        region = self.get_region(region_id)
        if not region:
            return None
        return BoundingBox(region.min_lng, region.min_lat, region.max_lng, region.max_lat)

    def get_all_images(self):
        return self.images.get_all()

    def get_image_by_id(self, image_id):
        return self.images.get_by_id(image_id)

    def get_images_by_region(self, region_id):
        return self.images.get_by_region(region_id)

    def add_image(self, region, lng, lat, id_from_source, source_captured_at, url, source, width=None, height=None):
        if isinstance(source_captured_at, int):
            captured_at = datetime.fromtimestamp(source_captured_at / 1000.0)
        elif isinstance(source_captured_at, str):
            try:
                captured_at = date_parser.parse(source_captured_at)
            except (ParserError, ValueError, TypeError):
                return None
        elif isinstance(source_captured_at, datetime):
            captured_at = source_captured_at
        elif isinstance(source_captured_at, date):
            captured_at = datetime.combine(
                source_captured_at, datetime.min.time())
        else:
            return None
        image = Image(region=region, lng=lng, lat=lat, id_from_source=id_from_source,
                      source_captured_at=captured_at, url=url, source=source, width=width, height=height)
        try:
            self.images.add(image)
            return True
        except:
            return None

    def add_many_images(self, images):
        self.images.add_many(images)

    def update_image_status(self, image_id, status):
        return self.images.update_status(image_id, status)

    def get_images_by_status(self, status, region_id=None):
        return self.images.get_by_status(status, region_id)

    def delete_image(self, image_id):
        return self.images.delete(image_id)

    def get_images_with_detections(self, region_id=None):
        return self.images.get_with_detections(region_id)

    def update_image_dimensions(self, image_id, width, height):
        return self.images.update_dimensions(image_id, width, height)

    def get_all_detections(self):
        return self.detections.get_all()

    def get_detections_by_image(self, image_id):
        return self.detections.get_detections_by_image(image_id)

    def get_detection_by_id(self, detection_id):
        return self.detections.get_by_id(detection_id)

    def add_detection(self, image, label, confidence, bbox):
        detection = Detection(image=image, label=label,
                              confidence=confidence, bbox=bbox)
        self.detections.add(detection)

    def add_many_detections(self, to_add):
        self.detections.add_many(to_add)

    def bulk_update_image_status(self, image_ids, status):
        self.images.bulk_update_status(image_ids, status)

    def delete_detection(self, detection_id):
        return self.detections.delete(detection_id)

    def get_unreviewed_detections(self, region_id=None):
        return self.detections.get_unreviewed(region_id)

    def get_detections_by_region(self, region_id):
        return self.detections.get_by_region(region_id)

    def add_osm_feature(self, region_id, osm_id, osm_type, lng, lat, name=None):
        osm_feature = OSMFeature(
            region_id=region_id, osm_id=osm_id, osm_type=osm_type, lng=lng, lat=lat, name=name)
        self.osm_features.add(osm_feature)

    def add_many_osm_features(self, to_add):
        self.osm_features.add_all(to_add)

    def get_osm_features_by_region(self, region_id, _type=None):
        query = self.osm_features.get_by_region(region_id)
        if _type is not None:
            query = query.filter_by(type=_type)
        return query.all()
