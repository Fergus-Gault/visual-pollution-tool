import tempfile
import os
import urllib.request
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.model import YoloModel
from src.utils import setup_logger
from src.database import DatabaseManager, Detection
from src.config import PipelineConfig

logger = setup_logger(__name__)


class InferenceManager:
    def __init__(self, db: DatabaseManager, model: YoloModel):
        self.db = db
        self.model = model

    def run_inference(self, region):
        images = self.db.get_images_by_status("unreviewed", region.id)
        logger.info(f"Found {len(images)} images for inference.")

        if not images:
            return

        total_detections = self._batch_process(images)

        logger.info(
            f"Processed {len(images)} images, found {total_detections} detections.")

    def _batch_process(self, images):
        batch_size = PipelineConfig.BATCH_SIZE
        num_batches = (len(images) + batch_size - 1) // batch_size
        batches = []

        all_results = []
        all_image_paths = []
        all_image_mappings = {}
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batches.append(batch)
        with ThreadPoolExecutor(max_workers=PipelineConfig.NUM_WORKERS) as executor:
            future_to_detection = {
                executor.submit(self._process_batch, batch, idx): (idx, batch) for idx, batch in enumerate(batches)
            }
            with tqdm(total=num_batches, desc="Running inference on batches") as pbar:
                for future in as_completed(future_to_detection):
                    results, image_paths, image_mappings = future.result()
                    all_results.extend(results)
                    all_image_paths.extend(image_paths)
                    all_image_mappings.update(image_mappings)
                    pbar.update(1)

        total_detections = self._process_results(
            all_results, all_image_paths, all_image_mappings)
        return total_detections

    def _temp_download(self, batch, idx):
        image_paths = []
        # Image mappings maps the image path to the image object
        image_mappings = {}
        temp_dir = tempfile.mkdtemp(prefix=f"inference_batch_{idx}")
        for idx, image in enumerate(batch):
            try:
                image_path = os.path.join(
                    temp_dir, f"img_{idx}_{image.id}.jpg")

                request = urllib.request.Request(
                    image.url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                )

                with urllib.request.urlopen(request, timeout=PipelineConfig.DOWNLOAD_TIMEOUT) as response:
                    with open(image_path, "wb") as out_file:
                        out_file.write(response.read())
                    image_paths.append(image_path)
                    image_mappings[image_path] = image

            except Exception as e:
                logger.debug(
                    f"Failed to download image {image.id} for batch: {e}")

        return image_paths, image_mappings

    def _process_batch(self, batch, idx):
        image_paths, image_mappings = self._temp_download(batch, idx)
        results = self.model.predict(source=image_paths)
        return results, image_paths, image_mappings

    def _process_results(self, results, image_paths, image_mapping):
        all_detections = []
        reviewed_ids = []
        num_detections = 0
        for result, path in zip(results, image_paths):
            image = image_mapping.get(path)
            if image:
                try:
                    detections = self._extract_det_info(result)
                    det_objects = self._build_detections(image, detections)
                    all_detections.extend(det_objects)
                    num_detections += len(det_objects)
                    reviewed_ids.append(image.id)
                except Exception as e:
                    logger.error(f"Failed to process image {image.id}: {e}")
                    self.db.update_image_status(image.id, "failed")

        self.db.add_many_detections(all_detections)
        for image_id in reviewed_ids:
            self.db.update_image_status(image_id, "reviewed")

        return num_detections

    def _build_detections(self, image, detections):
        to_add = []
        for det in detections:
            class_id = det["class_id"]
            label = self.model.get_class_names().get(class_id, str(class_id))
            to_add.append(Detection(
                image_id=image.id,
                label=label,
                confidence=det["confidence"],
                bbox=json.dumps(det["bbox"])
            ))
        return to_add

    def _extract_det_info(self, result):
        detections = []
        try:
            if result.boxes is None or len(result.boxes) == 0:
                return detections

            for *box, conf, cls_id in result.boxes.data.tolist():
                detections.append({
                    "class_id": int(cls_id),
                    "confidence": float(conf),
                    "bbox": [float(coord) for coord in box]
                })
        except Exception as e:
            logger.error(f"Failed to extract detections: {e}")
        return detections
