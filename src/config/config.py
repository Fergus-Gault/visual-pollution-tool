from pathlib import Path
import albumentations as A


class Config:
    BBOX_OFFSET = 0.025
    MAX_BBOX_AREA = 0.01
    MAX_OFFSET = (MAX_BBOX_AREA ** 0.5) / 2
    RADIUS_KM = 1.0
    ENV_PATH = "./auth/.env"
    REQ_TIMEOUT = 5
    DEBUG = False


class ArgsConfig:
    DEBUG = "--debug"
    COLLECT_ONLY = "--collect-only"
    OVERRIDE = "--override"
    REGION_COLLECT = "--region-collect"
    ARGS = [
        DEBUG,
        COLLECT_ONLY,
        OVERRIDE,
        REGION_COLLECT
    ]


class MapillaryConfig:
    BASE_URL = "https://graph.mapillary.com"
    DEFAULT_FIELDS = "id,computed_geometry,thumb_1024_url,captured_at"
    DEFAULT_DELAY = 0.3
    IMAGES_PER_POINT = 2
    SUBREGIONS = 10000


class KartaviewConfig:
    BASE_URL = "https://api.openstreetcam.org/2.0"
    ZOOM_LEVEL = 15
    DEFAULT_DELAY = 0.3
    IMAGES_PER_POINT = 1
    SUBREGIONS = 1000


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
    OSM_SUBREGIONS = 60


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
    IMAGE_STORAGE_CHUNK_SIZE = 2000
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


class LSConfig:
    BASE_URL = "http://localhost:8080"
    LABEL_CONFIG = """
        <View>
            <Image name="image" value="$image"/>
            <RectangleLabels name="label" toName="image">
                <Label value="graffiti"/>
                <Label value="barrier"/>
                <Label value="billboard"/>
                <Label value="utility_pole"/>
                <Label value="bin"/>
            </RectangleLabels>
        </View>
        """.strip()
    PROJECT_TITLE = "VP Detection"
    IMAGES_PER_REGION = 20
    MAX_RETRIES = 3
    BATCH_SIZE = 1000
    TIME_BETWEEN_BATCHES = 0.2
    REQ_TIMEOUT_S = 2
    MODEL_VERSION = "yolo_11_m"


class TrainConfig:
    BASE_MODEL = "yolo26l.pt"
    EPOCHS = 100
    IMGSZ = 640
    MODEL_VERSION = "v2"
    DATA_PATH = f"./data/datasets/{MODEL_VERSION}/data.yaml"
    DEVICE = "cuda"
    LABELS = {
        "barrier": "0",
        "billboard": "1",
        "bin": "2",
        "graffiti": "3",
        "mobile_advertisement": "4",
        "pothole": "5",
        "road_sign": "6",
        "shop_sign": "7",
        "utility_pole": "8"
    }
    LABELS_INV = {v: k for k, v in LABELS.items()}
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1
    AUGMENTATIONS = [
        A.Blur(blur_limit=5, p=0.3),
        A.CLAHE(clip_limit=3.0, p=0.3),
        A.Affine(rotate=(-30, 30), scale=(0.8, 1.2),
                 keep_ratio=True, rotate_method="largest_box"),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.2)
    ]
