from ultralytics import YOLO
from src.config import TrainConfig


def train_model(path=None, base_model=None, epochs=None, imgsz=None, device=None):
    model = YOLO(base_model or TrainConfig.BASE_MODEL)

    results = model.train(
        data=path or TrainConfig.DATA_PATH,
        epochs=epochs or TrainConfig.EPOCHS,
        imgsz=imgsz or TrainConfig.IMGSZ,
        batch=-1,
        device=device or TrainConfig.DEVICE,
        pretrained=True,
        cache=True
    )

    metrics = model.val(data=path or TrainConfig.DATA_PATH)
    return results, metrics
