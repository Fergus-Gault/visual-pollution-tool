from __future__ import annotations
from .logger import setup_logger
from src.config import Config

import numpy as np
from typing import TYPE_CHECKING

from geopy.geocoders import Nominatim
from shapely.geometry import Point
import osmnx as ox

if TYPE_CHECKING:
    from src.api.models import BoundingBox

logger = setup_logger(__name__)


class RegionManager:
    region_size = max(0, min(Config.BBOX_OFFSET * 2,
                      Config.MAX_BBOX_AREA ** 0.5))
    geolocator = Nominatim(user_agent="visual_pollution")

    @staticmethod
    def get_region_bbox(lng, lat):
        from src.api.models import BoundingBox

        half = RegionManager.region_size / 2
        return BoundingBox(lng - half, lat - half, lng + half, lat + half)

    @staticmethod
    def get_subregions(bbox: BoundingBox, num_subregions):
        from src.api.models import BoundingBox

        n_total = max(1, num_subregions)
        nx = int(np.ceil(n_total ** 0.5))
        ny = int(np.ceil(n_total / nx))

        lng_span = max(0.0, bbox.max_lng - bbox.min_lng)
        lat_span = max(0.0, bbox.max_lat - bbox.min_lat)

        lng_step = lng_span / nx if lng_span > 0 else RegionManager.region_size
        lat_step = lat_span / ny if lat_span > 0 else RegionManager.region_size

        lng_steps = np.arange(
            bbox.min_lng, bbox.max_lng, lng_step) if lng_step > 0 else np.array([bbox.min_lng])
        lat_steps = np.arange(
            bbox.min_lat, bbox.max_lat, lat_step) if lat_step > 0 else np.array([bbox.min_lat])

        subregions = [
            BoundingBox(float(lng), float(lat), float(min(lng + lng_step, bbox.max_lng)),
                        float(min(lat + lat_step, bbox.max_lat)))
            for lng in lng_steps for lat in lat_steps
        ]

        if len(subregions) > n_total:
            subregions = subregions[:n_total]
        return subregions

    @staticmethod
    def get_region_mid(bbox: BoundingBox):
        mid_lng = (bbox.min_lng + bbox.max_lng) / 2
        mid_lat = (bbox.min_lat + bbox.max_lat) / 2

        return (mid_lng, mid_lat)

    @staticmethod
    def geolocate_bbox(bbox: BoundingBox):
        lng, lat = RegionManager.get_region_mid(bbox)
        location = RegionManager.geolocator.reverse(
            f"{lat}, {lng}", exactly_one=True, language="en", addressdetails=True)
        if location is None:
            return
        address = location.raw['address']
        return (address.get('city'), address.get('country'))

    @staticmethod
    def geolocate_city(city, country=None):
        location = RegionManager.geolocator.geocode(
            f"{city} {country if country is not None else ''}", language="en", exactly_one=True)
        if location is None:
            logger.warning(f"City {city} not found. Returning.")
            return None
        return (location.longitude, location.latitude)

    @staticmethod
    def get_shape_file(city, country=None):
        try:
            query = f"{city}" + (f", {country}" if country is not None else "")
            return ox.geocode_to_gdf(query)
        except:
            return None

    @staticmethod
    def bbox_from_shape(gdf):
        from src.api.models import BoundingBox
        min_lng, min_lat, max_lng, max_lat = gdf.total_bounds
        return BoundingBox(min_lng, min_lat, max_lng, max_lat)

    @staticmethod
    def point_in_city(lng, lat, gdf):
        point = Point(lng, lat)

        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf.geometry.contains(point).any()

    @staticmethod
    def generate_region_name(bbox: BoundingBox):
        return f"Region_{bbox.min_lng:.3f}_{bbox.min_lat:.3f}_{bbox.max_lng:.3f}_{bbox.max_lat:.3f}"

    @staticmethod
    def get_combined_bbox(regions):
        from src.api.models import BoundingBox
        min_lng = min(r.min_lng for r in regions)
        min_lat = min(r.min_lat for r in regions)
        max_lng = max(r.max_lng for r in regions)
        max_lat = max(r.max_lat for r in regions)
        bbox = BoundingBox(min_lng, min_lat, max_lng, max_lat)

        centre = RegionManager.get_region_mid(bbox)

        return bbox, centre
