from ultralytics import YOLO
from src.config import TrainConfig


def train_model():
    model = YOLO(TrainConfig.BASE_MODEL)

    results = model.train(
        data=TrainConfig.DATA_PATH,
        epochs=TrainConfig.EPOCHS,
        imgsz=TrainConfig.IMGSZ,
        batch=-1,
        device=TrainConfig.DEVICE,
        pretrained=True,
        cache=True
    )

    metrics = model.val(data=TrainConfig.DATA_PATH)
    return results, metrics
