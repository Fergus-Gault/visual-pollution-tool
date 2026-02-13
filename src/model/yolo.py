from pathlib import Path
import torch
from ultralytics import YOLO
from config import YoloConfig

from utils import setup_logger

logger = setup_logger(__name__)


class YoloModel:
    def __init__(self, model_path=YoloConfig.DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._init_model()

    def _init_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file path not found: {self.model_path}.")
        model = YOLO(self.model_path)
        model.to(self.device)
        logger.info(f"Model successfully loaded on {self.device}.")
        return model

    def predict(self, source, conf=YoloConfig.CONF_THRESHOLD, iou=YoloConfig.IOU, imgsz=YoloConfig.IMGSZ, stream=YoloConfig.STREAM):
        source = self._normalise_source(source)

        predict_params = {
            "source": source,
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "device": self.device,
            "stream": stream,
        }
        return self.model.predict(**predict_params)

    def _normalise_source(source):
        if isinstance(source, list):
            return [str(s) if isinstance(s, Path) else s for s in source]
        if isinstance(source, Path):
            return str(source)
        return source

    def get_class_names(self):
        return self.model.names
