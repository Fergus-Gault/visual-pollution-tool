from src.database import DatabaseManager
from src.config import LSConfig, Config, TrainConfig
from src.utils import setup_logger, get_prediction, convert_ls_to_yolo
from dotenv import dotenv_values
from label_studio_sdk import LabelStudio
import requests
from collections import defaultdict
import time
import random
from pathlib import Path
from uuid import uuid4
logger = setup_logger(__name__)


class LabelStudioClient:
    def __init__(self, db: DatabaseManager):
        self.api_key = dotenv_values(
            Config.ENV_PATH).get("LABEL_STUDIO_API_KEY")
        self.base_url = LSConfig.BASE_URL.rstrip("/")
        self.db = db
        self.client = self._init_client()
        self.project_id = self._get_or_create_project()

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {self.api_key.strip()}",
            "Content-Type": "application/json",
        })
        self.max_retries = LSConfig.MAX_RETRIES
        self.batch_size = LSConfig.BATCH_SIZE
        self.timeout = LSConfig.REQ_TIMEOUT_S

    def _init_client(self):
        return LabelStudio(base_url=self.base_url, api_key=self.api_key)

    def _get_or_create_project(self):
        existing_projects = self.client.projects.list()
        for project in existing_projects:
            if project.title == LSConfig.PROJECT_TITLE:
                return int(project.id)

        project = self.client.projects.create(
            title=LSConfig.PROJECT_TITLE, label_config=LSConfig.LABEL_CONFIG)
        return int(project.id)

    def upload(self):
        tasks = self._make_task_payload()
        self._import_tasks(tasks)

    def _import_tasks(self, tasks):
        if not tasks:
            return []

        url = f"{self.base_url}/api/projects/{self.project_id}/import"

        for batch_idx, batch in enumerate(self._chunk(tasks)):
            logger.info(f"Uploading batch {batch_idx}.")
            self._post_with_retries(url, json=batch)
            logger.info(f"Successfully uploaded batch {batch_idx}")
            if LSConfig.TIME_BETWEEN_BATCHES > 0:
                time.sleep(LSConfig.TIME_BETWEEN_BATCHES)

    def _post_with_retries(self, url, json):
        last_err = None
        for attempt in range(self.max_retries):
            logger.info(
                f"Sending request, attempt {attempt+1}/{self.max_retries}")
            try:
                response = self.session.post(
                    url, json=json, timeout=self.timeout)
                if response.status_code in (429, 502, 503, 504):
                    raise requests.HTTPError(
                        f"{response.status_code} transient", response=response)

                response.raise_for_status()

                return response.json() if response.content else None

            except Exception as e:
                last_err = e
                sleep_s = 2 ** attempt
                time.sleep(sleep_s)
        raise RuntimeError(
            f"Bulk import failed after {self.max_retries} retries: {last_err}")

    def _chunk(self, tasks):
        if self.batch_size <= 0:
            raise ValueError("Batch size must be > 0.")
        for i in range(0, len(tasks), self.batch_size):
            yield list(tasks[i: i + self.batch_size])

    def _make_task_payload(self):
        tasks = []
        regions = self.db.get_all_regions()
        logger.info("Creating tasks payload.")
        for region in regions:
            images = self.db.get_random_images(
                region.id, LSConfig.IMAGES_PER_REGION)

            for img in images:
                task = {"data": {"image": img.url, "image_id": img.id}}

                preds = self.db.get_detections_by_image(img.id) or []
                if preds:
                    results = [get_prediction(img, p) for p in preds]
                    task_score = float(max(float(p.confidence) for p in preds))

                    task["predictions"] = [{
                        "model_version": LSConfig.MODEL_VERSION,
                        "score": task_score,
                        "result": results,
                    }]

                tasks.append(task)

        return tasks

    def _get_annotated_image_ids(self):
        url = f"{self.base_url}/api/tasks/"
        headers = {
            "Authorization": f"Token {self.api_key}",
        }
        params = {
            "project": self.project_id,
            "completed": "true",
            "page_size": 100,
        }
        annotated = []
        while True:
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            annotated.extend(data["results"])

            if not data.get("next"):
                break

            url = data["next"]
            params = None

        out = []
        for task in annotated:
            image_id = task["data"].get("image_id")
            annotations = task.get("annotations", [])
            out.append({
                "image_id": image_id,
                "task_id": task["id"],
                "annotations": annotations,
            })
        return out

    def download(self):
        # TODO: Restructure and split up
        annotated_imgs = self._get_annotated_image_ids()
        yolo_format = convert_ls_to_yolo(annotated_imgs)

        sorted_by_countries = defaultdict(list)

        for img_id, yolo_annos in yolo_format.items():
            img = self.db.get_image_by_id(img_id)
            sorted_by_countries[img.country].append({
                "image_id": img_id,
                "annotations": yolo_annos,
            })

        train, val, test = [], [], []

        for _, data in sorted_by_countries.items():
            data = list(data)
            random.shuffle(data)
            n = len(data)
            n_train = int(n * TrainConfig.TRAIN_SPLIT)
            n_val = int(n * TrainConfig.VAL_SPLIT)

            c_train = data[:n_train]
            c_val = data[n_train:n_train + n_val]
            c_test = data[n_train + n_val:]

            train.extend(c_train)
            val.extend(c_val)
            test.extend(c_test)

        dataset_foldername = f"dataset_{uuid4()[:8]}"
        train_path = Path(f"{dataset_foldername}/train")
        train_path.parent.mkdir(parents=True, exist_ok=True)
        val_path = Path(f"{dataset_foldername}/val")
        val_path.parent.mkdir(parents=True, exist_ok=True)
        test_path = Path(f"{dataset_foldername}/test")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: download_images()
        # TODO: write_label_files()
        # TODO: create_yaml()
