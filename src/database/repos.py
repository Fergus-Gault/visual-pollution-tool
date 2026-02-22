from abc import ABC
from typing import Generic, TypeVar, Type, List
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .models import Region, Image, Detection, OSMFeature
from src.utils import setup_logger

T = TypeVar('T')

logger = setup_logger(__name__)


class BaseRepo(ABC, Generic[T]):
    def __init__(self, session: Session, model: Type[T]):
        self.session = session
        self.model = model

    def get_by_id(self, entity_id):
        return self.session.query(self.model).filter_by(id=entity_id).first()

    def get_all(self):
        return self.session.query(self.model).all()

    def add(self, entity: T):
        try:
            self.session.add(entity)
            self.commit()
            return entity
        except Exception:
            logger.debug(
                f"Failed to add entity (likely duplicate). Skipping.")
            self.session.rollback()
            return None

    def add_all(self, entities: List[T]):
        self.session.add_all(entities)
        self.commit()

    def delete(self, entity_id):
        entity = self.get_by_id(entity_id)
        if not entity:
            return False
        self.session.delete(entity)
        self.commit()
        return True

    def commit(self):
        self.session.commit()

    def rollbcak(self):
        self.session.rollback()


class RegionRepo(BaseRepo[Region]):
    def __init__(self, session: Session):
        super().__init__(session, Region)

    def get_by_name(self, name):
        return self.session.query(Region).filter_by(name=name).first()


class ImageRepo(BaseRepo[Image]):
    def __init__(self, session: Session):
        super().__init__(session, Image)

    def get_by_region(self, region_id):
        return self.session.query(Image).filter_by(region_id=region_id).all()

    def get_by_status(self, status, region_id=None):
        query = self.session.query(Image).filter_by(status=status)
        if region_id is not None:
            query = query.filter_by(region_id=region_id)
        return query.all()

    def update_status(self, image_id, status):
        image = self.get_by_id(image_id)
        if not image:
            return False
        image.status = status
        self.commit()
        return True

    def update_dimensions(self, image_id, width, height):
        image = self.get_by_id(image_id)
        if not image:
            return False
        image.width = width
        image.height = height
        self.commit()
        return True

    def get_with_detections(self, region_id=None):
        query = self.session.query(Image).join(
            Detection).filter(Image.status == "reviewed")
        if region_id is not None:
            query.filter(Image.region_id == region_id)
        query = query.group_by(Image.id).having(func.count(Detection.id) > 0)
        return query.all()


class DetectionRepo(BaseRepo[Detection]):
    def __init__(self, session: Session):
        super().__init__(session, Detection)

    def get_by_region(self, region_id):
        return self.session.query(Detection).join(Image).filter(Image.region_id == region_id).all()

    def get_unreviewed(self, region_id=None):
        query = self.session.query(Detection).join(
            Image).filter_by(Detection.manual_reviewed == 0)
        if region_id is not None:
            query = query.filter(Image.region_id == region_id)
        return query.all()

    def get_detections_by_image(self, image_id):
        return self.session.query(Detection).filter_by(image_id=image_id).all()


class OSMFeatureRepo(BaseRepo[OSMFeature]):
    def __init__(self, session: Session):
        super().__init__(session, OSMFeature)

    def get_by_region(self, region_id):
        return self.session.query(OSMFeature).filter(region_id == region_id)

    def add_many_osm(self, entities: List[dict]):
        stmt = sqlite_insert(OSMFeature.__table__).values(entities)
        stmt = stmt.on_conflict_do_nothing(index_elements=["osm_id"])

        self.session.execute(stmt)
        self.session.commit()
