from ultralytics import YOLO
from src.config import TrainConfig
from pathlib import Path


def train_model(path=None, base_model=None, epochs=None, imgsz=None, device=None, augmentations=None, project=None, name=None, batch=None, workers=None):
    data_path = Path(path or TrainConfig.DATA_PATH)

    model = YOLO(base_model or TrainConfig.BASE_MODEL)

    results = model.train(
        data=str(data_path),
        epochs=epochs or TrainConfig.EPOCHS,
        imgsz=imgsz or TrainConfig.IMGSZ,
        batch=batch or TrainConfig.BATCH_SIZE,
        device=device or TrainConfig.DEVICE,
        workers=workers or TrainConfig.WORKERS,
        lr0=TrainConfig.LR0,
        lrf=TrainConfig.LRF,
        warmup_epochs=TrainConfig.WARMUP_EPOCHS,
        mosaic=TrainConfig.MOSAIC,
        mixup=TrainConfig.MIXUP,
        close_mosaic=TrainConfig.CLOSE_MOSAIC,
        freeze=TrainConfig.FREEZE,
        patience=TrainConfig.PATIENCE,
        augmentations=augmentations or TrainConfig.AUGMENTATIONS,
        project=project or TrainConfig.WANDB_PROJECT,
        name=name or TrainConfig.WANDB_NAME,
    )
    return results


def validate_model(path=None, model_path=None, imgsz=None, device=None, split="test"):
    data_path = Path(path or TrainConfig.DATA_PATH)
    model = YOLO(model_path or "data/model/best.pt")

    results = model.val(
        data=str(data_path),
        split=split,
        imgsz=imgsz or TrainConfig.IMGSZ,
        device=device or TrainConfig.DEVICE,
    )
    return results
