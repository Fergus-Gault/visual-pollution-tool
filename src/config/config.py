from pathlib import Path


class Config:
    DEFAULT_SUBREGIONS = 5000
    DEFAULT_POINTS = 500
    BBOX_OFFSET = 0.025
    MAX_BBOX_AREA = 0.01
    MAX_OFFSET = (MAX_BBOX_AREA ** 0.5) / 2
    RADIUS_KM = 0.7
    ENV_PATH = "./auth/.env"
    REQ_TIMEOUT = 5
    DEBUG = False


class MapillaryConfig:
    BASE_URL = "https://graph.mapillary.com"
    DEFAULT_FIELDS = "id,computed_geometry,thumb_1024_url,captured_at"
    DEFAULT_DELAY = 0.3
    IMAGES_PER_POINT = 2


class KartaviewConfig:
    BASE_URL = "https://api.openstreetcam.org/2.0"
    ZOOM_LEVEL = 20
    DEFAULT_DELAY = 0.3
    IMAGES_PER_POINT = 1


class OSMConfig:
    OSM_ENDPOINTS = [
        "https://overpass-api.de/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        "https://overpass.private.coffee/api/interpreter"
    ]
    BINS = 'node["amenity"="waste_basket"]'
    POWER_NODE = 'node["power"]'
    POWER_WAY = 'way["power"]'
    BILLBOARD_NODE = 'node["advertising"="billboard"]'
    BILLBOARD_WAY = 'way["advertising"="billboard"]'
    BARRIERS = 'way["barrier"]'
    TRAFFIC_SIGNS = 'node["highway"="traffic_signals"]'
    OSM_QUERIES = [BINS, POWER_NODE, POWER_WAY,
                   BILLBOARD_NODE, BILLBOARD_WAY, BARRIERS, TRAFFIC_SIGNS]
    RETRIES = 3


class DatabaseConfig:
    DEFAULT_DB_PATH = Path("data/db.sqlite3")

    @staticmethod
    def get_sqlite_url():
        path = DatabaseConfig.DEFAULT_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path}"


class YoloConfig:
    DEFAULT_MODEL_PATH = Path("data/model/best.pt")
    CONF_THRESHOLD = 0.50
    IMGSZ = 640
    IOU = 0.7
    STREAM = True


class PipelineConfig:
    IMAGE_STORAGE_CHUNK_SIZE = 500
    BATCH_SIZE = 16
    DOWNLOAD_TIMEOUT = 5
    NUM_WORKERS = 40


class MapConfig:
    MAPILLARY_COLOURS = "#088908"
    KARTAVIEW_COLOURS = "#0657A3"
    SOURCE_COLOURS = {
        "mapillary": MAPILLARY_COLOURS,
        "kartaview": KARTAVIEW_COLOURS,
    }
    BILLBOARDS_COLOUR = "#ca0a0a"
    BINS_COLOUR = "#1e43b3"
    UTILITY_POLE_COLOUR = "#158619"
    BARRIERS_COLOUR = "#5eb9e3"
    POTHOLES_COLOUR = "#A0750A"
    LITTER_COLOUR = "#8D05A1"
    GRAFFITI_COLOUR = "#f87ace"
    OTHER_COLOUR = "#203102"
    DETECTION_COLOURS = {
        "billboard": BILLBOARDS_COLOUR,
        "bin": BINS_COLOUR,
        "utility_pole": UTILITY_POLE_COLOUR,
        "barrier": BARRIERS_COLOUR,
        "pothole": POTHOLES_COLOUR,
        "litter": LITTER_COLOUR,
        "graffiti": GRAFFITI_COLOUR,
        "other": OTHER_COLOUR
    }
    TILES = "OpenStreetMap"
    ZOOM_START = 13
