from ultralytics import YOLO
from src.config import TrainConfig
from pathlib import Path
import wandb


def train_model(path=None, base_model=None, epochs=None, imgsz=None, device=None, augmentations=None, project=None, name=None):
    data_path = Path(path or TrainConfig.DATA_PATH)

    wandb.init(project=project or TrainConfig.WANDB_PROJECT,
               name=name, resume="allow")

    model = YOLO(base_model or TrainConfig.BASE_MODEL)

    results = model.train(
        data=str(data_path),
        epochs=epochs or TrainConfig.EPOCHS,
        imgsz=imgsz or TrainConfig.IMGSZ,
        batch=-1,
        device=device or TrainConfig.DEVICE,
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
        name=name,
    )
    return results
