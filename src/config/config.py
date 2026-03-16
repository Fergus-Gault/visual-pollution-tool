from pathlib import Path
import albumentations as A


class Config:
    BBOX_OFFSET = 0.04
    MAX_BBOX_AREA = 0.02
    MAX_OFFSET = (MAX_BBOX_AREA ** 0.5) / 2
    MAX_SHAPE_BBOX_AREA = 0.5
    RADIUS_KM = 1.0
    ENV_PATH = "./auth/.env"
    REQ_TIMEOUT = 3
    DEBUG = False
    DENSE_MULTIPLIER = 5
    MIN_POPULATION = 100000
    DEFAULT_CSV = "data/worldcities.csv"


class ArgsConfig:
    DEBUG = "--debug"


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
    @staticmethod
    def get_postgres_url():
        from dotenv import dotenv_values
        env = dotenv_values(Config.ENV_PATH)
        url = env.get("DATABASE_URL")
        if not url:
            raise ValueError(f"DATABASE_URL not set in {Config.ENV_PATH}")
        return url


class YoloConfig:
    DEFAULT_MODEL_PATH = Path("data/model/best.pt")
    CONF_THRESHOLD = 0.50
    IMGSZ = 1024
    IOU = 0.7
    STREAM = True


class PipelineConfig:
    IMAGE_STORAGE_CHUNK_SIZE = 2000
    BATCH_SIZE = 32
    DOWNLOAD_TIMEOUT = 5
    NUM_WORKERS = 40
    KARTAVIEW_WORKERS = 20
    MAPILLARY_WORKERS = 80
    INFERENCE_WORKERS = 20
    DIMENSION_WORKERS = 40
    REGION_WORKERS = 1
    MAPILLARY_RATE_LIMIT = 10000
    KARTAVIEW_RATE_LIMIT = 1000


class MapConfig:
    MAPILLARY_COLOURS = "#088908"
    KARTAVIEW_COLOURS = "#0657A3"
    SOURCE_COLOURS = {
        "mapillary": MAPILLARY_COLOURS,
        "kartaview": KARTAVIEW_COLOURS,
    }
    BILLBOARDS_COLOUR = "#ffab40"
    BINS_COLOUR = "#b551d8"
    UTILITY_POLE_COLOUR = "#63ac00"
    BARRIERS_COLOUR = "#f22750"
    POTHOLES_COLOUR = "#0185c7"
    GRAFFITI_COLOUR = "#ff7660"
    ROAD_SIGN_COLOUR = "#9abfff"
    SHOP_SIGN_COLOUR = "#4f5900"
    MOBILE_AD_COLOUR = "#ffabcd"
    OTHER_COLOUR = "#f0bd81"
    DETECTION_COLOURS = {
        "billboard": BILLBOARDS_COLOUR,
        "bin": BINS_COLOUR,
        "utility_pole": UTILITY_POLE_COLOUR,
        "barrier": BARRIERS_COLOUR,
        "pothole": POTHOLES_COLOUR,
        "graffiti": GRAFFITI_COLOUR,
        "road_sign": ROAD_SIGN_COLOUR,
        "shop_sign": SHOP_SIGN_COLOUR,
        "mobile_advertisement": MOBILE_AD_COLOUR,
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
                <Label value="mobile_advertisement"/>
                <Label value="pothole"/>
                <Label value="shop_sign"/>
                <Label value="road_sign"/>
            </RectangleLabels>
        </View>
        """.strip()
    PROJECT_TITLE = "VP Detection Fine-tuning"
    IMAGES_PER_COUNTRY = 10
    MAX_RETRIES = 3
    BATCH_SIZE = 1000
    TIME_BETWEEN_BATCHES = 0.2
    REQ_TIMEOUT_S = 2
    MODEL_VERSION = "yolo_26_m"


class TrainConfig:
    BASE_MODEL = "yolo26m.pt"
    WANDB_PROJECT = "dissertation"
    WANDB_NAME = "yolo_26_m"
    EPOCHS = 220
    IMGSZ = 1024
    LR0 = 0.003
    LRF = 0.01
    WARMUP_EPOCHS = 5
    MOSAIC = 0.5
    MIXUP = 0.0
    CLOSE_MOSAIC = 15
    FREEZE = 0
    PATIENCE = 50
    BATCH_SIZE = 16
    WORKERS = 8
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
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.30),
        A.HueSaturationValue(
            hue_shift_limit=6, sat_shift_limit=15, val_shift_limit=12, p=0.20),
        A.ImageCompression(quality_range=(55, 95), p=0.15),
        A.MotionBlur(blur_limit=(3, 5), p=0.10),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.10),
        A.RandomSizedBBoxSafeCrop(
            height=IMGSZ,
            width=IMGSZ,
            erosion_rate=0.0,
            p=0.30,
        ),
        A.Affine(
            scale=(0.9, 1.15),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-8, 8),
            shear={"x": (-4, 4), "y": (-2, 2)},
            interpolation=1,
            fit_output=False,
            keep_ratio=True,
            rotate_method="largest_box",
            p=0.35,
        ),
        A.HorizontalFlip(p=0.5),
    ]
