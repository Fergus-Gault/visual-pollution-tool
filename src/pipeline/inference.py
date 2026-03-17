import tempfile
import os
import shutil
import urllib.request
import json
from PIL import Image, UnidentifiedImageError
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
        all_failed_ids = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batches.append(batch)
        with ThreadPoolExecutor(max_workers=PipelineConfig.INFERENCE_WORKERS) as executor:
            future_to_detection = {
                executor.submit(self._process_batch, batch, idx): (idx, batch) for idx, batch in enumerate(batches)
            }
            with tqdm(total=num_batches, desc="Running inference on batches") as pbar:
                for future in as_completed(future_to_detection):
                    results, image_paths, image_mappings, failed_ids = future.result()
                    all_results.extend(results)
                    all_image_paths.extend(image_paths)
                    all_image_mappings.update(image_mappings)
                    all_failed_ids.extend(failed_ids)
                    pbar.update(1)

        total_detections = self._process_results(
            all_results, all_image_paths, all_image_mappings)

        if all_failed_ids:
            self.db.bulk_update_image_status(all_failed_ids, "failed")

        temp_dirs = {os.path.dirname(p) for p in all_image_paths}
        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)

        return total_detections

    def _temp_download(self, batch, idx):
        image_paths = []
        failed_ids = []
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

                    try:
                        with Image.open(image_path) as img:
                            img.verify()
                    except (UnidentifiedImageError, OSError) as e:
                        logger.debug(
                            f"Invalid image {image.id} for batch: {e}")
                        failed_ids.append(image.id)
                        if os.path.exists(image_path):
                            os.remove(image_path)
                        continue

                    image_paths.append(image_path)
                    image_mappings[image_path] = image

            except Exception as e:
                logger.debug(
                    f"Failed to download image {image.id} for batch: {e}")
                failed_ids.append(image.id)

        return image_paths, image_mappings, failed_ids

    def _process_batch(self, batch, idx):
        image_paths, image_mappings, failed_ids = self._temp_download(
            batch, idx)
        if not image_paths:
            return [], [], {}, failed_ids

        try:
            results = list(self.model.predict(source=image_paths))
            return results, image_paths, image_mappings, failed_ids
        except Exception as e:
            logger.warning(
                f"Batch {idx} inference failed ({e}). Falling back to per-image inference.")

        fallback_results = []
        fallback_paths = []
        fallback_mappings = {}

        for path in image_paths:
            image = image_mappings.get(path)
            if image is None:
                continue

            try:
                single_result = list(self.model.predict(source=[path]))
                if single_result:
                    fallback_results.append(single_result[0])
                    fallback_paths.append(path)
                    fallback_mappings[path] = image
                else:
                    failed_ids.append(image.id)
            except Exception as single_error:
                logger.debug(
                    f"Failed inference for image {image.id} in batch {idx}: {single_error}")
                failed_ids.append(image.id)

        return fallback_results, fallback_paths, fallback_mappings, failed_ids

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
        self.db.bulk_update_image_status(reviewed_ids, "reviewed")

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
