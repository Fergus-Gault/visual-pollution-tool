from pathlib import Path


class Config:
    DEFAULT_SUBREGIONS = 72
    DEFAULT_POINTS = 72
    BBOX_OFFSET = 0.025
    MAX_BBOX_AREA = 0.01
    MAX_OFFSET = (MAX_BBOX_AREA ** 0.5) / 2
    RADIUS_KM = 0.7
    ENV_PATH = "./auth/.env"
    REQ_TIMEOUT = 5


class MapillaryConfig:
    BASE_URL = "https://graph.mapillary.com"
    DEFAULT_FIELDS = "id,computed_geometry,thumb_1024_url,captured_at"
    DEFAULT_DELAY = 0.3
    IMAGES_PER_POINT = 40


class KartaviewConfig:
    BASE_URL = "https://api.openstreetcam.org/2.0"
    ZOOM_LEVEL = 18
    DEFAULT_DELAY = 0.3
    IMAGES_PER_POINT = 20


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
    CONF_THRESHOLD = 0.25
    IMGSZ = 640
    IOU = 0.7
    STREAM = True


class PipelineConfig:
    IMAGE_STORAGE_CHUNK_SIZE = 200
    BATCH_SIZE = 16
    DOWNLOAD_TIMEOUT = 2
    NUM_WORKERS = 40


class MapConfig:
    MAPILLARY_COLOURS = "#016E01"
    KARTAVIEW_COLOURS = "#A53912"
    TILES = "OpenStreetMap"
