
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Region(Base):

    __tablename__ = "regions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False)
    min_lng = Column(Float, nullable=False)
    min_lat = Column(Float, nullable=False)
    max_lng = Column(Float, nullable=False)
    max_lat = Column(Float, nullable=False)
    scanned_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    city = Column(String, nullable=True)
    country = Column(String, nullable=True)

    images = relationship("Image", back_populates="region",
                          cascade="all, delete-orphan")
    osm_features = relationship(
        "OSMFeature", back_populates="region", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    region_id = Column(String, ForeignKey('regions.id'))
    id_from_source = Column(String, nullable=True, unique=True)
    lng = Column(Float, nullable=True)
    lat = Column(Float, nullable=True)
    source_captured_at = Column(DateTime, nullable=False)
    url = Column(String, nullable=False)
    source = Column(String, nullable=True)
    status = Column(String, default="unreviewed", nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    region = relationship("Region", back_populates="images")
    detections = relationship(
        "Detection", back_populates="image", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    label = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    bbox = Column(String, nullable=True)
    time_of_detection = Column(
        DateTime, default=lambda: datetime.now(timezone.utc))
    manual_reviewed = Column(Integer, default=0, nullable=False)
    image = relationship("Image", back_populates="detections")
    image_id = Column(String, ForeignKey('images.id'))


class OSMFeature(Base):
    __tablename__ = "osm_features"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    region_id = Column(String, ForeignKey('regions.id'), nullable=False)
    osm_id = Column(String, unique=False, nullable=False)
    osm_type = Column(String, nullable=False)
    lng = Column(Float, nullable=False)
    lat = Column(Float, nullable=False)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    region = relationship("Region", back_populates="osm_features")
