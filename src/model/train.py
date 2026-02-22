from ultralytics import YOLO
from src.config import TrainConfig
from pathlib import Path
import asyncio
from ultralytics.data.converter import convert_ndjson_to_yolo
import yaml


def train_model(path=None, base_model=None, epochs=None, imgsz=None, device=None, augmentations=None):
    data_path = Path(path or TrainConfig.DATA_PATH)

    if data_path.suffix.lower() not in {".yaml", ".yml"}:
        converted = asyncio.run(
            convert_ndjson_to_yolo(
                data_path,
                output_path=data_path.parent
            )
        )

        yaml_path = Path(converted)
        dataset_root = yaml_path.parent

        data = yaml.safe_load(yaml_path.read_text())
        data["path"] = str(dataset_root)
        yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))

        if converted is not None:
            data_path = Path(converted)
        else:
            data_path = data_path.with_suffix(".yaml")

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
