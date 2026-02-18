from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from io import BytesIO
from PIL import Image as PILImage
from tqdm import tqdm

from src.config import PipelineConfig


class Dimensioner:

    @staticmethod
    def _update_single(param, session):
        url = param['url']
        try:
            response = session.get(url, timeout=0.5, stream=True)
            response.raise_for_status()

            img = PILImage.open(BytesIO(response.content))
            width, height = img.size
            param['width'] = width
            param['height'] = height
            return (param, True)
        except Exception:
            return (param, False)

    @staticmethod
    def update_dimensions(params_list):
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=PipelineConfig.NUM_WORKERS,
            pool_maxsize=PipelineConfig.NUM_WORKERS*4,
            max_retries=0
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        cleaned_params = []

        with ThreadPoolExecutor(max_workers=PipelineConfig.NUM_WORKERS) as executor:
            future_to_image = {
                executor.submit(Dimensioner._update_single, param, session): param for param in params_list
            }

            with tqdm(total=len(params_list), desc="Fetching image dimensions") as pbar:
                for future in as_completed(future_to_image):
                    updated_params, success = future.result()

                    if success and updated_params.get("width", None) is not None:
                        cleaned_params.append(updated_params)
                    else:
                        continue

                    pbar.update(1)

        return cleaned_params
