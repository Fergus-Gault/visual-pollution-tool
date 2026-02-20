from src.database import DatabaseManager
from src.config import LSConfig, Config, PipelineConfig
from src.utils import setup_logger, get_prediction
from dotenv import dotenv_values
from label_studio_sdk import LabelStudio

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
from typing import List

logger = setup_logger(__name__)

_thread = threading.local()


class LabelStudioClient:
    def __init__(self, db: DatabaseManager):
        self.api_key = dotenv_values(
            Config.ENV_PATH).get("LABEL_STUDIO_API_KEY")
        self.base_url = LSConfig.BASE_URL
        self.db = db
        self.client = self._init_client()
        self.project_id = self._get_or_create_project()

    def _init_client(self):
        return LabelStudio(base_url=self.base_url, api_key=self.api_key)

    def _get_or_create_project(self):
        existing_projects = self.client.projects.list()
        for project in existing_projects:
            if project.title == LSConfig.PROJECT_TITLE:
                return project.id

        project = self.client.projects.create(
            title=LSConfig.PROJECT_TITLE, label_config=LSConfig.LABEL_CONFIG)
        return project.id

    def upload(self):
        regions = self.db.get_all_regions()
        images = []
        for region in regions:
            images.extend(self.db.get_random_images(
                region.id, LSConfig.IMAGES_PER_REGION))

        tasks_payload = [
            {"data": {"image": img.url, "image_id": img.id}}
            for img in images
        ]

        tasks = self._create_tasks(tasks_payload)
        prediction_payload = []
        for img, task in zip(images, tasks):
            dets = self.db.get_detections_by_image(img.id) or []
            predictions = []
            if dets and (not img.width or not img.height):
                logger.warning("No dimensions found for image, skipping.")
                dets = []

            for det in dets:
                pred = get_prediction(img, det)
                predictions.append(pred)
            prediction_payload.append((task.id, predictions))

        self._attach_predictions(prediction_payload)

    def _create_tasks(self, tasks):
        tasks_list = list(tasks)
        data_list = [
            t["data"] if isinstance(t, dict) and "data" in t else t
            for t in tasks_list
        ]

        created_tasks = [None] * len(data_list)

        with ThreadPoolExecutor(max_workers=PipelineConfig.NUM_WORKERS) as ex:
            futures = {ex.submit(self._create_one_task, d): i for i, d in enumerate(data_list)}
            with tqdm(total=len(data_list), desc="Creating tasks") as pbar:
                for future in as_completed(futures):
                    i = futures[future]
                    created_tasks[i] = future.result()
                    pbar.update(1)

        return created_tasks

    def _get_client(self):
        if not hasattr(_thread, "client"):
            _thread.client = self._init_client()
        return _thread.client

    def _create_one_task(self, data):
        client = self._get_client()
        return client.tasks.create(project=self.project_id, data=data)

    def _attach_predictions(self, task_predictions: List):
        created_predictions = []

        with ThreadPoolExecutor(max_workers=PipelineConfig.NUM_WORKERS) as ex:
            futures = [ex.submit(self._attach_single, tp)
                       for tp in task_predictions]
            with tqdm(total=len(futures), desc="Attaching predictions") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res is not None:
                        if isinstance(res, list):
                            created_predictions.extend(res)
                        else:
                            created_predictions.append(res)
                    pbar.update(1)

        logger.info(
            f"Successfully attached {len(created_predictions)} predictions.")

    def _attach_single(self, task_prediction):
        client = self._get_client()
        task_id, results = task_prediction

        payload = {"task": task_id, "result": results}
        try:
            return client.predictions.create(**payload)
        except Exception as e:
            logger.warning(
                f"Failed to attach prediction for task {task_id}: {e}")
            return None
