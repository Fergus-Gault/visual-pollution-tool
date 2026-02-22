from ultralytics import YOLO
from src.config import TrainConfig
from pathlib import Path


def train_model(path=None, base_model=None, epochs=None, imgsz=None, device=None, augmentations=None):
    data_path = Path(path or TrainConfig.DATA_PATH)

    model = YOLO(base_model or TrainConfig.BASE_MODEL)

    results = model.train(
        data=str(data_path),
        epochs=epochs or TrainConfig.EPOCHS,
        imgsz=imgsz or TrainConfig.IMGSZ,
        batch=-1,
        device=device or TrainConfig.DEVICE,
        pretrained=True,
        augmentations=augmentations or TrainConfig.AUGMENTATIONS,
    )

    metrics = model.val(data=str(data_path))
    return results, metrics
