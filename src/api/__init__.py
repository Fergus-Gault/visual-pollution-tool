from .models import Geometry, ImageMetadata, BoundingBox, ImageRequest, ImageStoreMetadata
from .manager import APIManager
from .client import HTTPClient
from .kartaview import KartaviewAPI
from .mapillary import MapillaryAPI
from .osm import OSMApi, OSMFeatureClassifier
from src.config import PipelineConfig
from src.utils import RateLimiter


def normalise_image_sources(value):
    if value is None:
        return ["mapillary", "kartaview"]
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = str(value).split(",")

    sources = []
    for raw_value in raw_values:
        token = str(raw_value).strip().lower()
        if not token:
            continue
        if token in {"both", "all"}:
            for source in ("mapillary", "kartaview"):
                if source not in sources:
                    sources.append(source)
            continue
        if token not in {"mapillary", "kartaview"}:
            raise ValueError(
                f"Unsupported image source '{raw_value}'. Use mapillary, kartaview, or both."
            )
        if token not in sources:
            sources.append(token)

    if not sources:
        raise ValueError("At least one image source must be specified.")
    return sources


def build_image_apis(image_sources=None, mapillary_token=None):
    sources = normalise_image_sources(image_sources)
    apis = []
    if "mapillary" in sources:
        apis.append(
            MapillaryAPI(
                access_token=mapillary_token,
                rate_limiter=RateLimiter(max_calls=PipelineConfig.MAPILLARY_RATE_LIMIT),
            )
        )
    if "kartaview" in sources:
        apis.append(
            KartaviewAPI(
                rate_limiter=RateLimiter(max_calls=PipelineConfig.KARTAVIEW_RATE_LIMIT)
            )
        )
    return apis

__all__ = ['APIManager', 'Geometry',
           'ImageMetadata', 'BoundingBox', 'ImageRequest', 'HTTPClient',
           'KartaviewAPI', 'MapillaryAPI', 'OSMApi', 'OSMFeatureClassifier', 'ImageStoreMetadata',
           'normalise_image_sources', 'build_image_apis']
