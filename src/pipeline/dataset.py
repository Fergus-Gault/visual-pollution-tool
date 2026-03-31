from src.utils import convert_ls_to_yolo, setup_logger
from src.config import TrainConfig, PipelineConfig
from collections import defaultdict, Counter
import random
import json
from pathlib import Path
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED, as_completed
from tqdm import tqdm
import requests

logger = setup_logger(__name__)


class DatasetManager:
    def __init__(self, db, name=uuid4(), base_path=None):
        self.db = db
        self.uuid = str(name)[:8]
        self.ds_name = f"dataset_{self.uuid}"
        base_dir = Path(base_path) if base_path is not None else Path("./datasets")
        self.folder_path = base_dir / self.ds_name
        self.out_path = Path(f"{self.folder_path}/{self.ds_name}.ndjson")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def create_dataset(self, annotated_imgs):
        yolo_format = convert_ls_to_yolo(annotated_imgs)
        dataset_records = self._build_dataset_records(yolo_format)
        downloaded_imgs = self._download_images(dataset_records)
        ds_split = self._split_dataset(downloaded_imgs)
        self._create_ndjson(ds_split)
        return self.ds_name
    
    def download_data(self):
        self._stream_download_data()
        return self.ds_name

    def _build_dataset_records(self, yolo_format):
        dataset_records = []
        for img_id, yolo_annos in yolo_format.items():
            img = self.db.get_image_by_id(img_id)
            dataset_records.append({
                "type": "image",
                "image_id": img_id,
                "url": img.url,
                "width": img.width,
                "height": img.height,
                "annotations": {
                    "boxes": yolo_annos
                },
            })
        return dataset_records

    def _build_db_image_record(self, img, region):
        predictions = self._serialise_predictions(img)
        boxes = []
        for pred in predictions:
            yolo_box = self._bbox_to_yolo_box(
                pred["bbox"],
                img.width,
                img.height,
                pred["label"],
            )
            if yolo_box is not None:
                boxes.append(yolo_box)

        record = {
            "type": "image",
            "image_id": img.id,
            "url": img.url,
            "width": img.width,
            "height": img.height,
            "region_id": img.region_id,
            "region_name": getattr(region, "name", None) if region is not None else None,
            "city": getattr(region, "city", None) if region is not None else None,
            "country": getattr(region, "country", None) if region is not None else None,
            "lng": img.lng,
            "lat": img.lat,
            "annotations": {
                "boxes": boxes,
            },
        }

        if predictions:
            record["predictions"] = predictions

        return record

    def _serialise_predictions(self, img):
        preds = list(getattr(img, "detections", []) or [])
        if not preds:
            preds = self.db.get_detections_by_image(img.id) or []

        serialised = []
        for pred in preds:
            bbox_coords = self._parse_bbox(pred.bbox)
            if bbox_coords is None:
                continue

            pred_record = {
                "label": pred.label,
                "class_id": self._get_class_id(pred.label),
                "confidence": float(pred.confidence) if pred.confidence is not None else None,
                "bbox": bbox_coords,
            }

            serialised.append(pred_record)

        for pred in preds:
            try:
                self.db.session.expunge(pred)
            except Exception:
                pass

        return serialised

    def _parse_bbox(self, bbox):
        try:
            coords = json.loads(bbox) if isinstance(bbox, str) else bbox
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

        if not coords or len(coords) != 4:
            return None

        try:
            return [float(coord) for coord in coords]
        except (TypeError, ValueError):
            return None

    def _bbox_to_yolo_box(self, bbox, img_width, img_height, label):
        class_id = self._get_class_id(label)
        if class_id is None or not img_width or not img_height:
            return None

        x1, y1, x2, y2 = bbox
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w == 0.0 or h == 0.0:
            return None

        xc = (x1 + (w / 2.0)) / float(img_width)
        yc = (y1 + (h / 2.0)) / float(img_height)
        w /= float(img_width)
        h /= float(img_height)
        return [class_id, xc, yc, w, h]

    def _get_class_id(self, label):
        class_id = TrainConfig.LABELS.get(label)
        if class_id is None:
            return None
        return int(class_id)

    def _split_dataset(self, dataset_records):
        n = len(dataset_records)
        if n == 0:
            return {"train": [], "val": [], "test": []}

        n_train = int(n * TrainConfig.TRAIN_SPLIT)
        n_val = int(n * TrainConfig.VAL_SPLIT)
        n_test = n - n_train - n_val

        target_sizes = {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        }
        split_ratios = {
            "train": TrainConfig.TRAIN_SPLIT,
            "val": TrainConfig.VAL_SPLIT,
            "test": TrainConfig.TEST_SPLIT,
        }

        rng = random.Random(42)
        data = list(dataset_records)
        rng.shuffle(data)
        data.sort(key=lambda rec: len(
            rec.get("annotations", {}).get("boxes", [])), reverse=True)

        total_class_counts = Counter()
        image_class_counts = []
        for rec in data:
            class_counts = Counter()
            for box in rec.get("annotations", {}).get("boxes", []):
                if not box:
                    continue
                class_counts[int(box[0])] += 1
            image_class_counts.append(class_counts)
            total_class_counts.update(class_counts)

        target_class_counts = {
            split: {
                cls: total * split_ratios[split]
                for cls, total in total_class_counts.items()
            }
            for split in ("train", "val", "test")
        }

        split_data = {"train": [], "val": [], "test": []}
        current_class_counts = {
            "train": Counter(),
            "val": Counter(),
            "test": Counter(),
        }

        for rec, class_counts in zip(data, image_class_counts):
            best_split = None
            best_score = None

            for split in ("train", "val", "test"):
                if len(split_data[split]) >= target_sizes[split]:
                    continue

                score = 0.0
                for cls, count in class_counts.items():
                    deficit = target_class_counts[split][cls] - \
                        current_class_counts[split][cls]
                    if deficit > 0:
                        score += min(deficit, count)

                remaining_capacity = target_sizes[split] - \
                    len(split_data[split])
                tiebreak = remaining_capacity / max(1, target_sizes[split])
                candidate = (score, tiebreak)

                if best_score is None or candidate > best_score:
                    best_score = candidate
                    best_split = split

            if best_split is None:
                best_split = min(("train", "val", "test"),
                                 key=lambda s: len(split_data[s]))

            split_data[best_split].append(rec)
            current_class_counts[best_split].update(class_counts)

        return split_data

    def _create_ndjson(self, ds_split):
        with open(self.out_path, "w", encoding="utf-8") as f:
            self._write_ndjson_meta(f)

            for split, records in ds_split.items():
                for rec in records:
                    rec = dict(rec)
                    rec["split"] = split
                    f.write(json.dumps(rec) + "\n")

    def _write_ndjson_meta(self, file_obj):
        meta = {
            "type": "dataset",
            "task": "detect",
            "name": "VP Detection",
            "description": "Dataset containing annotated images of visual pollution",
            "class_names": TrainConfig.LABELS_INV,
        }
        file_obj.write(json.dumps(meta) + "\n")

    def _stream_download_data(self):
        total_images = self.db.get_image_count()
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=PipelineConfig.NUM_WORKERS,
            pool_maxsize=PipelineConfig.NUM_WORKERS * 4,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        max_in_flight = max(1, PipelineConfig.NUM_WORKERS * 2)
        futures = set()
        stats = {"downloaded": 0, "failed": 0}

        with open(self.out_path, "w", encoding="utf-8") as f:
            self._write_ndjson_meta(f)

            with ThreadPoolExecutor(max_workers=PipelineConfig.NUM_WORKERS) as executor:
                with tqdm(total=total_images, desc="Downloading images", unit="img") as pbar:
                    for img in self.db.iter_all_images():
                        region = getattr(img, "region", None)
                        record = self._build_db_image_record(img, region)
                        country = getattr(region, "country", None) or "unknown"

                        future = executor.submit(
                            self._download_single,
                            record,
                            country,
                            session,
                        )
                        futures.add(future)

                        if len(futures) >= max_in_flight:
                            futures = self._flush_completed_downloads(
                                futures,
                                f,
                                pbar,
                                stats,
                                wait_for_all=False,
                            )

                        try:
                            self.db.session.expunge(img)
                        except Exception:
                            pass
                        if region is not None:
                            try:
                                self.db.session.expunge(region)
                            except Exception:
                                pass

                    while futures:
                        futures = self._flush_completed_downloads(
                            futures,
                            f,
                            pbar,
                            stats,
                            wait_for_all=True,
                        )

    def _flush_completed_downloads(self, futures, file_obj, pbar, stats, wait_for_all):
        if not futures:
            return futures

        return_when = ALL_COMPLETED if wait_for_all else FIRST_COMPLETED
        done, pending = wait(futures, return_when=return_when)

        for future in done:
            success, _, data = future.result()
            if success:
                record = dict(data)
                record["split"] = "all"
                file_obj.write(json.dumps(record) + "\n")
                stats["downloaded"] += 1
            else:
                stats["failed"] += 1
            pbar.update(1)

        pbar.set_postfix(downloaded=stats["downloaded"], failed=stats["failed"], queued=len(pending))
        file_obj.flush()
        return pending

    def _download_images(self, dataset_records):
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=PipelineConfig.NUM_WORKERS,
            pool_maxsize=PipelineConfig.NUM_WORKERS * 4,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        downloaded_imgs = []
        total_images = len(dataset_records)

        with ThreadPoolExecutor(max_workers=PipelineConfig.NUM_WORKERS) as executor:
            future_to_download = {
                executor.submit(self._download_single, data, session): data
                for data in dataset_records
            }

            with tqdm(total=total_images, desc="Downloading images.") as pbar:
                for future in as_completed(future_to_download):
                    success, data = future.result()
                    if success:
                        downloaded_imgs.append(data)
                    pbar.update(1)

        return downloaded_imgs

    def _download_single(self, data, session: requests.Session):
        img_id = data.get("image_id")
        url = data.get("url")

        if not img_id or not url:
            return False, data

        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()

            Path(self.folder_path).mkdir(parents=True, exist_ok=True)
            path = Path(self.folder_path) / f"{img_id}.jpg"

            with open(path, "wb") as f:
                f.write(resp.content)

            data["file"] = f"{img_id}.jpg"
            return True, data

        except Exception:
            return False, data
