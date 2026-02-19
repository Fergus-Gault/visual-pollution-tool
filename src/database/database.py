from dateutil import parser as date_parser
from dateutil.parser import ParserError
from datetime import datetime, date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.models import BoundingBox
from .models import Base, Region, Image, Detection, OSMFeature
from .repos import RegionRepo, ImageRepo, DetectionRepo, OSMFeatureRepo
from src.config import DatabaseConfig
from src.utils import setup_logger, RegionManager

logger = setup_logger(__name__)


class DatabaseManager:
    def __init__(self):
        db_url = DatabaseConfig.get_sqlite_url()

        self.engine = create_engine(db_url)
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

    def add_region(self, bbox: BoundingBox, city=None, country=None, override=False):
        name = RegionManager.generate_region_name(bbox)
        existing_region = self.regions.get_by_name(name)
        if existing_region and not override:
            return None
        elif existing_region and override:
            return region

        region = Region(name=name, min_lng=bbox.min_lng, min_lat=bbox.min_lat,
                        max_lng=bbox.max_lng, max_lat=bbox.max_lat, city=city, country=country,)
        self.regions.add(region)
        return region

    def delete_region(self, region_id):
        images = self.get_images_by_region(region_id)
        for image in images:
            self.images.delete(image.id)
        detections = self.get_detections_by_region(region_id)
        for det in detections:
            self.detections.delete(det.id)
        osm = self.get_osm_features_by_region(region_id)
        for osm_d in osm:
            self.osm_features.delete(osm_d.id)
        return self.regions.delete(region_id)

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

    def get_osm_features_by_region(self, region_id, _type=None):
        query = self.osm_features.get_by_region(region_id)
        if _type is not None:
            query = query.filter_by(type=_type)
        return query.all()
