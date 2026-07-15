
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String
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
    iso3 = Column(String, nullable=True)
    population = Column(Integer, nullable=True)
    start_captured_at = Column(DateTime, nullable=True, default=None)
    end_captured_at = Column(DateTime, nullable=True, default=None)
    dense_scan = Column(Boolean, nullable=False, default=False)
    osm_fetched = Column(Boolean, nullable=False, default=False)
    score = Column(Float, nullable=True)
    images_per_square_km = Column(Float, nullable=True)
    gdp = Column(Float, nullable=True)
    gdp_year = Column(Integer, nullable=True)
    gdppp = Column(Float, nullable=True)
    gdppp_year = Column(Integer, nullable=True)
    gni = Column(Float, nullable=True)
    gni_year = Column(Integer, nullable=True)
    urb = Column(Float, nullable=True)
    urb_year = Column(Integer, nullable=True)

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
    url_fetched_at = Column(DateTime, nullable=True)
    source = Column(String, nullable=True)
    status = Column(String, default="unreviewed", nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    score = Column(Float, nullable=True)
    altitude = Column(Float, nullable=True)
    atomic_scale = Column(Float, nullable=True)
    camera_parameters = Column(JSON, nullable=True)
    camera_type = Column(String, nullable=True)
    compass_angle = Column(Float, nullable=True)
    computed_altitude = Column(Float, nullable=True)
    computed_compass_angle = Column(Float, nullable=True)
    computed_rotation = Column(JSON, nullable=True)
    creator_id = Column(String, nullable=True)
    creator_username = Column(String, nullable=True)
    exif_orientation = Column(Integer, nullable=True)
    is_pano = Column(Boolean, nullable=True)
    camera_make = Column(String, nullable=True)
    camera_model = Column(String, nullable=True)
    on_foot = Column(Boolean, nullable=True)
    organization_id = Column(String, nullable=True)
    organization_name = Column(String, nullable=True)
    organization_slug = Column(String, nullable=True)
    quality_score = Column(Float, nullable=True)
    sequence = Column(String, nullable=True)
    source_metadata = Column(JSON, nullable=True)
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
