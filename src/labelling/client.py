from src.database import DatabaseManager
from src.config import LSConfig, Config
from src.utils import setup_logger, get_prediction
from src.pipeline import DatasetManager
from dotenv import dotenv_values
from label_studio_sdk import LabelStudio
import requests
import time
import json
logger = setup_logger(__name__)


class LabelStudioClient:
    def __init__(self, db: DatabaseManager):
        self.api_key = dotenv_values(
            Config.ENV_PATH).get("LABEL_STUDIO_API_KEY")
        self.base_url = LSConfig.BASE_URL.rstrip("/")
        self.db = db
        self.client = self._init_client()
        self.project_id = self._get_or_create_project()
        self.dsm = DatasetManager(self.db)

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
        headers = {"Authorization": f"Token {self.api_key}"}

        page = 1
        page_size = 100
        out = []

        while True:
            params = {
                "project": self.project_id,
                "page": page,
                "page_size": page_size,
                "only_annotated": "true",
                "fields": "all",
            }

            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            batch = data.get("tasks") or data.get("results") or []
            if not batch:
                break

            for t in batch:
                task_id = t.get("id")
                task_data = t.get("data") or {}

                image_id = task_data.get("image_id")
                image_url = task_data.get("image_url") or task_data.get(
                    "image") or task_data.get("image_url")
                annos = t.get("annotations") or []
                if isinstance(annos, str):
                    try:
                        annos = json.loads(annos)
                    except json.JSONDecodeError:
                        annos = []
                results = []
                for a in annos:
                    results.extend(a.get("result") or [])

                if results:
                    out.append({
                        "image_id": image_id,
                        "image_url": image_url,
                        "task_id": task_id,
                        "results": results,
                    })

            # Stop condition
            if len(batch) < page_size:
                break
            page += 1

        return out

    def fetch_annotations(self):
        return self._get_annotated_image_ids()
