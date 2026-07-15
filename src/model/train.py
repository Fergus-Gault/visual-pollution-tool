import json
import os
from pathlib import Path
import shutil

from ultralytics import YOLO

from src.config import TrainConfig


def _safe_stem_path(dataset_dir: Path, image_record: dict) -> Path | None:
    file_name = image_record.get("file")
    if file_name:
        candidate = dataset_dir / file_name
        if candidate.exists():
            return candidate

    image_id = image_record.get("image_id")
    if not image_id:
        return None

    matches = list(dataset_dir.glob(f"{image_id}.*"))
    for match in matches:
        if match.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            return match
    return None


def _write_yolo_label(label_path: Path, boxes: list) -> None:
    lines = []
    for box in boxes or []:
        if not isinstance(box, list) or len(box) != 5:
            continue
        cls_id, x_center, y_center, width, height = box
        lines.append(
            f"{int(cls_id)} {float(x_center):.10f} {float(y_center):.10f} {float(width):.10f} {float(height):.10f}"
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except FileExistsError:
        return
    except OSError:
        if dst.exists():
            return
        shutil.copy2(src, dst)


def _build_yolo_dataset_from_ndjson(ndjson_path: Path) -> Path:
    dataset_dir = ndjson_path.parent
    output_dir = dataset_dir / f"{ndjson_path.stem}_yolo"
    yaml_path = output_dir / "data.yaml"

    if yaml_path.exists() and yaml_path.stat().st_mtime >= ndjson_path.stat().st_mtime:
        return yaml_path

    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = None
    split_counts = {"train": 0, "val": 0, "test": 0}

    with ndjson_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            record = json.loads(line)
            if record.get("type") == "dataset":
                raw_names = record.get("class_names") or {}
                class_names = {
                    int(idx): name for idx, name in raw_names.items()
                }
                continue

            if record.get("type") != "image":
                continue

            split = record.get("split") or "train"
            if split not in split_counts:
                split = "train"

            src_image = _safe_stem_path(dataset_dir, record)
            if src_image is None:
                continue

            image_dir = output_dir / "images" / split
            label_dir = output_dir / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            dst_image = image_dir / src_image.name
            _link_or_copy(src_image, dst_image)

            label_path = label_dir / f"{src_image.stem}.txt"
            boxes = (record.get("annotations") or {}).get("boxes") or []
            _write_yolo_label(label_path, boxes)
            split_counts[split] += 1

    if not class_names:
        class_names = {int(idx): name for idx, name in TrainConfig.LABELS_INV.items()}

    val_split = "val" if split_counts["val"] else "test" if split_counts["test"] else "train"
    test_split = "test" if split_counts["test"] else val_split

    yaml_lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        f"val: images/{val_split}",
    ]
    if split_counts["test"] or test_split != val_split:
        yaml_lines.append(f"test: images/{test_split}")

    yaml_lines.append("names:")
    for idx in sorted(class_names):
        yaml_lines.append(f"  {idx}: {class_names[idx]}")

    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def _resolve_data_path(path=None) -> Path:
    data_path = Path(path or TrainConfig.DATA_PATH)
    if data_path.suffix.lower() == ".ndjson":
        return _build_yolo_dataset_from_ndjson(data_path)
    return data_path


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
    data_path = _resolve_data_path(path)
    model = YOLO(model_path or "data/model/best.pt")

    results = model.val(
        data=str(data_path),
        split=split,
        imgsz=imgsz or TrainConfig.IMGSZ,
        device=device or TrainConfig.DEVICE,
    )
    return results
