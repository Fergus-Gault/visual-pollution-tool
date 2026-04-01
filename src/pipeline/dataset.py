from src.utils import convert_ls_to_yolo, setup_logger
from src.config import TrainConfig, PipelineConfig
from collections import defaultdict, Counter
import json
import mimetypes
import random
import tarfile
import tempfile
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import requests
from tqdm import tqdm

logger = setup_logger(__name__)


class DatasetManager:
    def __init__(self, db, name=uuid4(), base_path=None, shard_size=10000):
        self.db = db
        self.uuid = str(name)[:8]
        self.ds_name = f"dataset_{self.uuid}"
        base_dir = Path(base_path) if base_path is not None else Path("./datasets")
        self.folder_path = base_dir / self.ds_name
        self.out_path = Path(f"{self.folder_path}/{self.ds_name}.ndjson")
        self.index_path = self.folder_path / "index.ndjson"
        self.meta_path = self.folder_path / "dataset.json"
        self.shards_path = self.folder_path / "shards"
        self.tmp_path = self.folder_path / ".tmp"
        self.shard_size = max(1, int(shard_size))
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def create_dataset(self, annotated_imgs):
        yolo_format = convert_ls_to_yolo(annotated_imgs)
        sorted_countries = self._sort_countries(yolo_format)
        downloaded_imgs = self._download_images(sorted_countries)
        ds_split = self._split_dataset(downloaded_imgs)
        self._create_ndjson(ds_split)
        return self.ds_name
    
    def download_data(self):
        self._stream_download_data()
        return str(self.folder_path)

    def _sort_countries(self, yolo_format):
        sorted_by_country = defaultdict(list)
        for img_id, yolo_annos in yolo_format.items():
            img = self.db.get_image_by_id(img_id)
            if hasattr(img, "country"):
                country = img.country
            else:
                country = "unknown"
            sorted_by_country[country].append({
                "type": "image",
                "image_id": img_id,
                "url": img.url,
                "width": img.width,
                "height": img.height,
                "annotations": {
                    "boxes": yolo_annos
                },
            })
        return sorted_by_country

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

    def _split_dataset(self, sorted_countries):
        all_data = [item for country_data in sorted_countries.values()
                    for item in country_data]
        n = len(all_data)
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
        data = list(all_data)
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

    def _write_dataset_meta(self):
        meta = {
            "type": "dataset",
            "task": "detect",
            "name": "VP Detection",
            "description": "Dataset containing images of visual pollution",
            "class_names": TrainConfig.LABELS_INV,
            "format": {
                "storage": "tar-shards",
                "index": "ndjson",
                "preserves_original_bytes": True,
                "shard_size": self.shard_size,
            },
            "paths": {
                "index": self.index_path.name,
                "shards": self.shards_path.name,
            },
        }

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
            f.write("\n")

    def _stream_download_data(self):
        total_images = self.db.get_image_count()
        self.shards_path.mkdir(parents=True, exist_ok=True)
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        self._write_dataset_meta()

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
        stats = {"downloaded": 0, "failed": 0, "shards": 0}

        with open(self.index_path, "w", encoding="utf-8") as index_file:
            with self._open_shard(0) as shard:
                shard_state = {"idx": 0, "count": 0}

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
                                futures, shard = self._flush_completed_downloads(
                                    futures,
                                    index_file,
                                    shard,
                                    shard_state,
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
                            futures, shard = self._flush_completed_downloads(
                                futures,
                                index_file,
                                shard,
                                shard_state,
                                pbar,
                                stats,
                                wait_for_all=True,
                            )

                stats["shards"] = shard_state["idx"] + 1 if shard_state["count"] > 0 else shard_state["idx"]

        session.close()

    def _flush_completed_downloads(self, futures, index_file, shard, shard_state, pbar, stats, wait_for_all):
        if not futures:
            return futures, shard

        return_when = ALL_COMPLETED if wait_for_all else FIRST_COMPLETED
        done, pending = wait(futures, return_when=return_when)

        for future in done:
            success, _, data = future.result()
            if success:
                shard = self._write_downloaded_sample(
                    data,
                    shard,
                    shard_state,
                    index_file,
                )
                stats["downloaded"] += 1
            else:
                stats["failed"] += 1
            pbar.update(1)

        pbar.set_postfix(
            downloaded=stats["downloaded"],
            failed=stats["failed"],
            shard=shard_state["idx"],
            queued=len(pending),
        )
        index_file.flush()
        return pending, shard

    def _download_images(self, images):
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=PipelineConfig.NUM_WORKERS,
            pool_maxsize=PipelineConfig.NUM_WORKERS * 4,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        downloaded_imgs = defaultdict(list)
        total_images = sum(len(country_data) for country_data in images.values())

        with ThreadPoolExecutor(max_workers=PipelineConfig.NUM_WORKERS) as executor:
            future_to_download = {
                executor.submit(self._download_single, data, country, session):
                data for country, country_data in images.items()
                for data in country_data
            }

            with tqdm(total=total_images, desc="Downloading images.") as pbar:
                for future in as_completed(future_to_download):
                    success, country, data = future.result()
                    if success:
                        self._materialise_temp_download(data)
                        downloaded_imgs[country].append(data)
                    pbar.update(1)

        session.close()
        return downloaded_imgs

    def _download_single(self, data, country, session: requests.Session):
        img_id = data.get("image_id")
        url = data.get("url")

        if not img_id or not url:
            return False, country, data

        tmp_file = None
        resp = None
        try:
            resp = session.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            suffix = self._guess_image_suffix(url, resp.headers.get("Content-Type"))

            with tempfile.NamedTemporaryFile(
                dir=self.tmp_path,
                prefix=f"{img_id}_",
                suffix=suffix,
                delete=False,
            ) as tmp_file:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp_file.write(chunk)

            data["file_ext"] = suffix
            data["content_type"] = resp.headers.get("Content-Type")
            data["temp_file"] = tmp_file.name
            data["source_url"] = url
            return True, country, data

        except Exception:
            if tmp_file is not None:
                try:
                    Path(tmp_file.name).unlink(missing_ok=True)
                except Exception:
                    pass
            return False, country, data
        finally:
            try:
                resp.close()
            except Exception:
                pass

    def _open_shard(self, shard_idx):
        shard_name = self._shard_name(shard_idx)
        return tarfile.open(self.shards_path / shard_name, mode="w")

    def _shard_name(self, shard_idx):
        return f"shard-{shard_idx:06d}.tar"

    def _write_downloaded_sample(self, data, shard, shard_state, index_file):
        if shard_state["count"] >= self.shard_size:
            shard.close()
            shard_state["idx"] += 1
            shard_state["count"] = 0
            shard = self._open_shard(shard_state["idx"])

        img_id = data["image_id"]
        file_ext = data.get("file_ext") or ".jpg"
        image_member = f"{img_id}{file_ext}"
        meta_member = f"{img_id}.json"
        temp_path = Path(data["temp_file"])

        try:
            image_info = tarfile.TarInfo(name=image_member)
            image_info.size = temp_path.stat().st_size
            with open(temp_path, "rb") as img_file:
                shard.addfile(image_info, img_file)

            record = dict(data)
            record.pop("temp_file", None)
            record["file"] = image_member
            record["metadata_file"] = meta_member
            record["split"] = "all"
            record["shard"] = self._shard_name(shard_state["idx"])

            encoded_meta = json.dumps(record).encode("utf-8")
            meta_info = tarfile.TarInfo(name=meta_member)
            meta_info.size = len(encoded_meta)
            shard.addfile(meta_info, fileobj=self._bytes_io(encoded_meta))

            index_record = {
                "image_id": img_id,
                "shard": record["shard"],
                "file": image_member,
                "metadata_file": meta_member,
                "url": record.get("url"),
                "width": record.get("width"),
                "height": record.get("height"),
                "country": record.get("country"),
                "city": record.get("city"),
                "region_id": record.get("region_id"),
                "region_name": record.get("region_name"),
                "lat": record.get("lat"),
                "lng": record.get("lng"),
                "prediction_count": len(record.get("predictions", [])),
                "annotation_count": len(record.get("annotations", {}).get("boxes", [])),
                "split": "all",
            }
            index_file.write(json.dumps(index_record) + "\n")
            shard_state["count"] += 1
            return shard
        finally:
            temp_path.unlink(missing_ok=True)

    def _materialise_temp_download(self, data):
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        suffix = data.get("file_ext") or ".jpg"
        dest_path = Path(self.folder_path) / f"{data['image_id']}{suffix}"
        temp_path = Path(data["temp_file"])
        temp_path.replace(dest_path)
        data.pop("temp_file", None)
        data["file"] = dest_path.name

    def _bytes_io(self, data):
        from io import BytesIO

        return BytesIO(data)

    def _guess_image_suffix(self, url, content_type=None):
        parsed = urlparse(url)
        suffix = Path(parsed.path).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            return suffix

        guessed = mimetypes.guess_extension((content_type or "").split(";")[0].strip())
        if guessed == ".jpe":
            return ".jpg"
        if guessed:
            return guessed
        return ".jpg"
