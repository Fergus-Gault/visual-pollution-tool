import tempfile
import os
import urllib.request
import json

from src.model import YoloModel
from src.utils import setup_logger
from src.database import DatabaseManager
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
        total_detections = 0
        batch_size = PipelineConfig.BATCH_SIZE
        num_batches = (len(images) + batch_size - 1) // batch_size

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_num = i // batch_size + 1
            logger.info(
                f"Processing batch {batch_num}/{num_batches} ({len(batch)}) images)")
            image_paths, image_mappings = self._temp_download(batch)
            results = self._process_batch(image_paths)
            num_detections = self._process_results(
                results, image_paths, image_mappings)
            total_detections += num_detections
            logger.info(
                f"Batch complete. Total detections so far: {total_detections}")
        return total_detections

    def _temp_download(self, batch):
        image_paths = []
        # Image mappings maps the image path to the image object
        image_mappings = []
        temp_dir = tempfile.mkdtemp(prefix="inference_batch_")
        for idx, image in enumerate(batch):
            try:
                image_path = os.path.join(
                    temp_dir, f"img_{idx}_{image.id}.jpg")

                with urllib.request.urlopen(image.url, timeout=PipelineConfig.DOWNLOAD_TIMEOUT) as response:
                    with open(image_path, "wb") as out_file:
                        out_file.write(response.read())
                    image_paths.append(image_path)
                    image_mappings[image_path] = image

            except Exception as e:
                logger.warning(
                    f"Failed to download image {image.id} for batch: {e}")
                self.db.update_image_status(image.id, "failed")

        return image_paths, image_mappings

    def _process_batch(self, image_paths):
        return self.model.predict(source=image_paths)

    def _process_results(self, results, image_paths, image_mapping):
        num_detections = 0
        for result, path in zip(results, image_paths):
            image = image_mapping.get(path)
            if image:
                try:
                    detections = self._extract_det_info(result)
                    self._store_detection(image, detections)
                    self.db.update_image_status(image.id, "reviewed")
                    num_detections += len(detections)
                except Exception as e:
                    logger.error(f"Failed to process image {image.id}: {e}")
                    self.db.update_image_status(image.id, "failed")

        return num_detections

    def _store_detection(self, image, detections):
        for det in detections:
            class_id = det["class_id"]
            label = self.model.get_class_names().get(class_id, str(class_id))
            self.db.add_detection(
                image=image,
                label=label,
                confidence=det["confidence"],
                bbox=json.dump(det["bbox"])
            )

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
