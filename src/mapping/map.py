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

    def save(self, m, region, file_type):
        if file_type.lower() == "png":
            self._save_as_png(m, region, file_type)
        elif file_type.lower() == "html":
            self._save_as_html(m, region, file_type)
        else:
            logger.warning(
                "Invalid file type, cannot save. Please use either png or html.")

    def _save_as_png(self, m, region, file_type="png"):
        img_data = m._to_png(5)
        img = PILImage.open(BytesIO(img_data))
        file_name = self._generate_filename(m, region, file_type)
        img.save(file_name)

    def _save_as_html(self, m, region, file_type="html"):
        file_name = self._generate_filename(m, region, file_type)
        m.save(file_name)

    def _generate_filename(self, region, file_type):
        output_file = f"maps/region_images/{region.country}/{region.city}/{region.id}"
        output_file += ".png" if file_type == "png" else ".html"
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def map_region_images(self, regions):
        return RegionImages.map_region_images(self.db, regions)

    def map_region_detections(self, regions):
        return RegionDetections.map_region_detections(self.db, regions)


class MapHelper:
    @staticmethod
    def draw_region_bounds(m, regions):
        for region in regions:
            region_bounds = [
                [region.min_lat, region.min_lng],
                [region.min_lat, region.max_lng],
                [region.max_lat, region.max_lng],
                [region.max_lat, region.min_lng],
                [region.min_lat, region.min_lng]
            ]
            folium.PolyLine(
                region_bounds,
                color='black',
                weight=2,
                opacity=0.8,
                dash_array='5, 10',
                popup=region.city
            ).add_to(m)
        return m
