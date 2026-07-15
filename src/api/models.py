from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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
    altitude: Optional[float] = None
    atomic_scale: Optional[float] = None
    camera_parameters: Optional[list] = None
    camera_type: Optional[str] = None
    compass_angle: Optional[float] = None
    computed_altitude: Optional[float] = None
    computed_compass_angle: Optional[float] = None
    computed_rotation: Optional[list] = None
    creator_id: Optional[str] = None
    creator_username: Optional[str] = None
    exif_orientation: Optional[int] = None
    is_pano: Optional[bool] = None
    make: Optional[str] = None
    model: Optional[str] = None
    on_foot: Optional[bool] = None
    organization_id: Optional[str] = None
    organization_name: Optional[str] = None
    organization_slug: Optional[str] = None
    quality_score: Optional[float] = None
    sequence: Optional[str] = None
    mapillary_metadata: Optional[Dict[str, Any]] = None

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
        optional_fields = (
            "altitude",
            "atomic_scale",
            "camera_parameters",
            "camera_type",
            "compass_angle",
            "computed_altitude",
            "computed_compass_angle",
            "computed_rotation",
            "creator_id",
            "creator_username",
            "exif_orientation",
            "is_pano",
            "make",
            "model",
            "on_foot",
            "organization_id",
            "organization_name",
            "organization_slug",
            "quality_score",
            "sequence",
            "mapillary_metadata",
        )
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value

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
            altitude=data.get("altitude"),
            atomic_scale=data.get("atomic_scale"),
            camera_parameters=data.get("camera_parameters"),
            camera_type=data.get("camera_type"),
            compass_angle=data.get("compass_angle"),
            computed_altitude=data.get("computed_altitude"),
            computed_compass_angle=data.get("computed_compass_angle"),
            computed_rotation=data.get("computed_rotation"),
            creator_id=(data.get("creator") or {}).get("id"),
            creator_username=(data.get("creator") or {}).get("username"),
            exif_orientation=data.get("exif_orientation"),
            is_pano=data.get("is_pano"),
            make=data.get("make"),
            model=data.get("model"),
            on_foot=data.get("on_foot"),
            organization_id=(data.get("organization") or {}).get("id"),
            organization_name=(data.get("organization") or {}).get("name"),
            organization_slug=(data.get("organization") or {}).get("slug"),
            quality_score=data.get("quality_score"),
            sequence=data.get("sequence"),
            mapillary_metadata={
                key: data.get(key)
                for key in (
                    "creator",
                    "organization",
                    "sfm_cluster",
                    "mesh",
                    "detections",
                    "geometry",
                    "computed_geometry",
                    "thumb_256_url",
                    "thumb_512_url",
                    "thumb_2048_url",
                    "thumb_original_url",
                    "merge_cc",
                )
                if data.get(key) is not None
            },
        )

    @classmethod
    def from_kartaview(cls, data):
        image_url = data.get("imageLthUrl", "") or ""
        if not image_url:
            image_url = data.get("fileurl", "")
            if "{{sizeprefix}}" in image_url:
                image_url = image_url.replace("{{sizeprefix}}", "lth")

        return cls(
            id=str(data.get("id", "")),
            geometry=Geometry(
                coordinates=(data.get("lng", 0.0), data.get("lat", 0.0))
            ),
            thumb_1024_url=image_url,
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
    start_captured_at: Optional[str] = None
    end_captured_at: Optional[str] = None

    # Kartaview specific
    zoom_level: int = KartaviewConfig.ZOOM_LEVEL

    def to_mapillary_params(self):
        params = {
            "fields": self.fields,
            "bbox": self.bbox.to_str(),
            "is_pano": self.is_pano,
            "limit": MapillaryConfig.IMAGES_PER_POINT,
        }
        if self.start_captured_at is not None:
            params["start_captured_at"] = self.start_captured_at
        if self.end_captured_at is not None:
            params["end_captured_at"] = self.end_captured_at
        return params

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

    def to_osm_params(self, queries=None):
        query_parts = []
        bbox = f"{self.bbox.min_lat},{self.bbox.min_lng},{self.bbox.max_lat}, {self.bbox.max_lng}"
        if queries is None:
            queries = OSMConfig.OSM_QUERIES
        for query in queries:
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
            'width': img_data.get('width'),
            'height': img_data.get('height'),
            'altitude': img_data.get('altitude'),
            'atomic_scale': img_data.get('atomic_scale'),
            'camera_parameters': img_data.get('camera_parameters'),
            'camera_type': img_data.get('camera_type'),
            'compass_angle': img_data.get('compass_angle'),
            'computed_altitude': img_data.get('computed_altitude'),
            'computed_compass_angle': img_data.get('computed_compass_angle'),
            'computed_rotation': img_data.get('computed_rotation'),
            'creator_id': img_data.get('creator_id'),
            'creator_username': img_data.get('creator_username'),
            'exif_orientation': img_data.get('exif_orientation'),
            'is_pano': img_data.get('is_pano'),
            'camera_make': img_data.get('make'),
            'camera_model': img_data.get('model'),
            'on_foot': img_data.get('on_foot'),
            'organization_id': img_data.get('organization_id'),
            'organization_name': img_data.get('organization_name'),
            'organization_slug': img_data.get('organization_slug'),
            'quality_score': img_data.get('quality_score'),
            'sequence': img_data.get('sequence'),
            'source_metadata': img_data.get('mapillary_metadata'),
        }
