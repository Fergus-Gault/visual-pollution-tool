import numpy as np
import random

import urllib.request
import json

from config import Config
from api import BoundingBox
from utils import setup_logger

logger = setup_logger(__name__)


class RegionManager:
    region_size = max(0, min(Config.BBOX_OFFSET * 2,
                      Config.MAX_BBOX_AREA ** 0.5))

    @staticmethod
    def get_region_bbox(lng, lat):
        region_x = int(lng // RegionManager.region_size) * \
            RegionManager.region_size
        region_y = int(lat // RegionManager.region_size) * \
            RegionManager.region_size

        min_lng = region_x
        min_lat = region_y
        max_lng = region_x + RegionManager.region_size
        max_lat = region_y + RegionManager.region_size

        return BoundingBox(min_lng, min_lat, max_lng, max_lat)

    @staticmethod
    def get_subregions(bbox: BoundingBox):
        n_total = max(1, Config.DEFAULT_SUBREGIONS)
        nx = int(np.ceil(n_total ** 0.5))
        ny = int(np.ceil(n_total / nx))

        lng_span = max(0.0, bbox.max_lng - bbox.min_lng)
        lat_span = max(0.0, bbox.max_lat - bbox.min_lat)

        lng_step = lng_span / nx if lng_span > 0 else RegionManager.region_size
        lat_step = lat_span / nx if lat_span > 0 else RegionManager.region_size

        lng_steps = np.arange(
            bbox.min_lng, bbox.max_lng, lng_step) if lng_step > 0 else np.array([bbox.min_lng])
        lat_steps = np.arange(
            bbox.min_lat, bbox.max_lat, lat_step) if lat_step > 0 else np.array([bbox.min_lat])

        subregions = [
            (float(lng), float(lat), float(min(lng + lng_step, bbox.max_lng)),
             float(min(lat + lat_step, bbox.max_lat)))
            for lng in lng_steps for lat in lat_steps
        ]

        if len(subregions) > n_total:
            subregions = subregions[:n_total]
        return subregions

    @staticmethod
    def get_random_points(bbox: BoundingBox, num_points):
        if num_points <= 0:
            return []

        min_lng, max_lng = sorted([bbox.min_lng, bbox.max_lng])
        min_lat, max_lat = sorted([bbox.min_lat, bbox.max_lat])

        return [(random.uniform(min_lng, max_lng), random.uniform(min_lat, max_lat)) for _ in range(num_points)]

    @staticmethod
    def get_region_mid(bbox: BoundingBox):
        mid_lng = (bbox.min_lng + bbox.max_lng) / 2
        mid_lat = (bbox.min_lat + bbox.max_lat) / 2

        return (mid_lng, mid_lat)

    @staticmethod
    def geolocate(bbox: BoundingBox):
        lng, lat = RegionManager.get_region_mid(bbox)
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&zoom=10&addressdetails=1&accept-language=en"
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'VisualPollutionDetection/1.0'}
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read().decode())
            if 'address' not in data:
                logger.warning(
                    f"No address found for coordinates ({lng}, {lat})")
                return None, None
            address = data['address']
            city = (
                address.get('city') or
                address.get('town') or
                address.get('village') or
                address.get('municipality') or
                address.get('county') or
                address.get('state')
            )
            country = address.get('country')

            return (country, city)

        except urllib.error.HTTPError as e:
            logger.error(f"HTTP error during geocoding ({lng}, {lat}): {e}")
            return None, None
        except urllib.error.URLError as e:
            logger.error(f"URL error during geocoding ({lng}, {lat}): {e}")
            return None, None
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON decode error during geocoding ({lng}, {lat}): {e}")
            return None, None
        except Exception as e:
            logger.error(
                f"Unexpected error during geocoding ({lng}, {lat}): {e}")
            return None, None
