import folium
from src.utils import setup_logger, RegionManager
from src.config import MapConfig
from src.database import DatabaseManager
from src.api import BoundingBox

logger = setup_logger(__name__)


class WorldImages:
    @staticmethod
    def map_world_images(db: DatabaseManager):
        all_regions = db.get_all_regions()

        m = folium.Map(location=[20, 0],
                       zoom_start=2, tile=MapConfig.TILES)

        for region in all_regions:
            bbox = BoundingBox(region.min_lng, region.min_lat,
                               region.max_lng, region.max_lat)
            lng, lat = RegionManager.get_region_mid(bbox)

            folium.CircleMarker(location=[lat, lng],
                                radius=4,
                                color="#3388ff",
                                fill=True,
                                fillColor="#3388ff",
                                fillOpacity=0.7,
                                weight=1
                                ).add_to(m)

        folium.LayerControl().add_to(m)
        return m
