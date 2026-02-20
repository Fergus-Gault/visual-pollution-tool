from dataclasses import dataclass, field
from typing import Optional

from src.config import Config, KartaviewConfig, OSMConfig, MapillaryConfig


@dataclass
class Geometry:
    type: str = "Point"
    coordinates: tuple = field(default_factory=lambda: (0.0, 0.0))

    @property
    def lnggitude(self):
        return self.coordinates[0]

    @property
    def latitude(self):
        return self.coordinates[1]

    def to_dict(self):
        return {
            "type": self.type,
            "coordinates": list(self.coordinates)
        }


@dataclass
class ImageMetadata:
    # Required fields
    id: str
    geometry: Geometry
    thumb_1024_url: str
    captured_at: str
    source: str

    # Optional fields
    width: Optional[int] = None
    height: Optional[int] = None

    def to_dict(self):
        result = {
            "id": self.id,
            "geometry": self.geometry.to_dict(),
            "thumb_1024_url": self.thumb_1024_url,
            "captured_at": self.captured_at,
            "_source": self.source,
        }
        if self.width is not None:
            result["width"] = self.width
        if self.height is not None:
            result["height"] = self.height

        return result

    @classmethod
    def from_mapillary(cls, data):
        geometry_data = data.get("computed_geometry", {})
        coords = geometry_data.get("coordinates", [0.0, 0.0])

        return cls(
            id=str(data.get("id", "")),
            geometry=Geometry(
                coordinates=(coords[0], coords[1])
            ),
            thumb_1024_url=data.get("thumb_1024_url", ""),
            captured_at=data.get("captured_at", ""),
            source="mapillary",
            width=data.get("width"),
            height=data.get("height"),
        )

    @classmethod
    def from_kartaview(cls, data):
        file_url = data.get('fileurl', '')
        if '{{sizeprefix}}' in file_url:
            file_url = file_url.replace('{{sizeprefix}}', 'lth')

        return cls(
            id=str(data.get("id", "")),
            geometry=Geometry(
                coordinates=(data.get("lng", 0.0), data.get("lat", 0.0))
            ),
            thumb_1024_url=file_url,
            captured_at=data.get("shotDate", data.get("dateAdded", "")),
            source="kartaview",
            width=data.get("width"),
            height=data.get("height"),
        )


@dataclass
class BoundingBox:
    min_lng: float
    min_lat: float
    max_lng: float
    max_lat: float

    def to_str(self):
        return f"{self.min_lng:.6f},{self.min_lat:.6f},{self.max_lng:.6f},{self.max_lat:.6f}"

    def to_tuple(self):
        return (self.min_lng, self.min_lat, self.max_lng, self.max_lat)

    def to_json(self):
        return f"{'min_lng': {self.min_lng}, 'min+'}"

    @classmethod
    def from_centre(cls, lng, lat):
        lat_offset = Config.RADIUS_KM / 111.0
        lng_offset = Config.RADIUS_KM / \
            (111.0 * abs(lat) if lat != 0 else 111.0)
        return cls(
            min_lng=lng - lng_offset,
            min_lat=lat - lat_offset,
            max_lng=lng + lng_offset,
            max_lat=lat + lat_offset
        )


@dataclass
class ImageRequest:
    bbox: BoundingBox
    is_pano: bool = False
    fields: str = MapillaryConfig.DEFAULT_FIELDS

    # Kartaview specific
    zoom_level: int = KartaviewConfig.ZOOM_LEVEL

    def to_mapillary_params(self):
        return {
            "fields": self.fields,
            "bbox": self.bbox.to_str(),
            "is_pano": self.is_pano,
            "limit": MapillaryConfig.IMAGES_PER_POINT,
        }

    def to_kartaview_params(self):
        return {
            "nwLat": self.bbox.max_lat,
            "nwLng": self.bbox.min_lng,
            "seLat": self.bbox.min_lat,
            "seLng": self.bbox.max_lng,
            "zoomLevel": self.zoom_level,
            "join": "sequence",
            "itemsPerPage": KartaviewConfig.IMAGES_PER_POINT,
            "page": 1
        }

    def to_osm_params(self):
        query_parts = []
        bbox = f"{self.bbox.min_lat},{self.bbox.min_lng},{self.bbox.max_lat}, {self.bbox.max_lng}"
        for query in OSMConfig.OSM_QUERIES:
            query_parts.append(f"{query}[!'location']({bbox});")
            query_parts.append(
                f"{query}[location=outdoor]({bbox});")

        query = "\n".join(query_parts)

        return f"({query});out body;"


class ImageStoreMetadata:
    @staticmethod
    def convert_data(img_data, region, api):
        captured_at = img_data.get('captured_at')
        source_id = str(img_data.get('id'))
        geometry = img_data.get('geometry') or img_data.get(
            'computed_geometry', {})
        coords = geometry.get('coordinates', [None, None])
        lng, lat = coords[0], coords[1]
        if lng is None or lat is None:
            return None

        url = img_data.get('thumb_1024_url')
        source = img_data.get('_source')

        return {
            'region': region,
            'lng': lng,
            'lat': lat,
            'id_from_source': source_id,
            'source_captured_at': captured_at,
            'url': url,
            'source': source,
        }
