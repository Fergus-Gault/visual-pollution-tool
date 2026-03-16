from src.utils import convert_ls_to_yolo, setup_logger
from src.config import TrainConfig, PipelineConfig
from collections import defaultdict, Counter
import random
import json
from pathlib import Path
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests

logger = setup_logger(__name__)


class DatasetManager:
    def __init__(self, db, name=uuid4()):
        self.db = db
        self.uuid = str(name)[:8]
        self.ds_name = f"dataset_{self.uuid}"
        self.folder_path = Path(f"./datasets/{self.ds_name}/")
        self.out_path = Path(f"{self.folder_path}/{self.ds_name}.ndjson")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def create_dataset(self, annotated_imgs):
        yolo_format = convert_ls_to_yolo(annotated_imgs)
        sorted_countries = self._sort_countries(yolo_format)
        downloaded_imgs = self._download_images(sorted_countries)
        ds_split = self._split_dataset(downloaded_imgs)
        self._create_ndjson(ds_split)
        return self.ds_name

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

        meta = {
            "type": "dataset",
            "task": "detect",
            "name": "VP Detection",
            "description": "Dataset containing annotated images of visual pollution",
            "class_names": TrainConfig.LABELS_INV,
        }

        with open(self.out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")

            for split, records in ds_split.items():
                for rec in records:
                    rec = dict(rec)
                    rec["split"] = split
                    f.write(json.dumps(rec) + "\n")

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

        with ThreadPoolExecutor(max_workers=PipelineConfig.NUM_WORKERS) as executor:
            future_to_download = {
                executor.submit(self._download_single, data, country, session):
                data for country, country_data in images.items()
                for data in country_data
            }

            with tqdm(total=len(images), desc="Downloading images.") as pbar:
                for future in as_completed(future_to_download):
                    success, country, data = future.result()
                    if success:
                        downloaded_imgs[country].append(data)
                    pbar.update(1)

        return downloaded_imgs

    def _download_single(self, data, country, session: requests.Session):
        img_id = data.get("image_id")
        url = data.get("url")

        if not img_id or not url:
            return False, country, data

        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()

            Path(self.folder_path).mkdir(parents=True, exist_ok=True)
            path = Path(self.folder_path) / f"{img_id}.jpg"

            with open(path, "wb") as f:
                f.write(resp.content)

            data["file"] = f"{img_id}.jpg"
            return True, country, data

        except Exception:
            return False, country, data
