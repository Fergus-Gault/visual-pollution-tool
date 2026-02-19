from PIL import Image as PILImage
from io import BytesIO
from pathlib import Path
import folium
from src.database import DatabaseManager
from src.utils import setup_logger
from .region_images import RegionImages
from .region_detections import RegionDetections

logger = setup_logger(__name__)


class Mapper:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def save(self, m, region, file_type, map_type="region_images"):
        if file_type.lower() == "png":
            self._save_as_png(m, region, map_type, file_type)
        elif file_type.lower() == "html":
            self._save_as_html(m, region, map_type, file_type)
        else:
            logger.warning(
                "Invalid file type, cannot save. Please use either png or html.")

    def _save_as_png(self, m, region, map_type, file_type="png"):
        if m is None:
            logger.warning("No map to save.")
            return
        img_data = m._to_png(5)
        img = PILImage.open(BytesIO(img_data))
        file_name = self._generate_filename(region, map_type, file_type)
        img.save(file_name)

    def _save_as_html(self, m, region, map_type, file_type="html"):
        if m is None:
            logger.warning("No map to save.")
            return
        file_name = self._generate_filename(region, map_type, file_type)
        m.save(file_name)

    def _generate_filename(self, region, map_type, file_type):
        output_file = f"maps/{region.country}/{region.city}/{map_type}/{region.id}"
        output_file += ".png" if file_type == "png" else ".html"
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def map_region_images(self, regions):
        return RegionImages.map_region_images(self.db, regions)

    def map_region_detections(self, regions):
        return RegionDetections.map_region_detections(self.db, regions)
