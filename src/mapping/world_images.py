import folium
import math
from branca.colormap import LinearColormap
from sqlalchemy import func
from src.utils import setup_logger, RegionManager
from src.config import MapConfig
from src.database import DatabaseManager, Image
from src.api import BoundingBox

logger = setup_logger(__name__)


class WorldImages:
    @staticmethod
    def map_world_images(db: DatabaseManager):
        all_regions = db.get_all_regions()
        coords = []

        m = folium.Map(location=[20, 0],
                       zoom_start=2,
                       tiles=MapConfig.get_tiles_url(),
                       attr=MapConfig.TILES_ATTR,
                       prefer_canvas=True)

        for region in all_regions:
            bbox = BoundingBox(region.min_lng, region.min_lat,
                               region.max_lng, region.max_lat)
            lng, lat = RegionManager.get_region_mid(bbox)
            coords.append([lat, lng])

            folium.CircleMarker(location=[lat, lng],
                                radius=4,
                                color="#3388ff",
                                fill=True,
                                fillColor="#3388ff",
                                fillOpacity=0.7,
                                weight=1
                                ).add_to(m)

        if coords:
            lats = [coord[0] for coord in coords]
            lngs = [coord[1] for coord in coords]
            m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]],
                         padding=(20, 20), max_zoom=6)

        folium.LayerControl().add_to(m)
        return m

    @staticmethod
    def map_world_images_scaled_by_count(db: DatabaseManager, min_radius=1, max_radius=4):
        all_regions = db.get_all_regions()
        coords = []

        colour_scale = LinearColormap(
            colors=["#d7191c", "#fa7d00", "#007425"],
            vmin=0,
            vmax=1,
            caption="Log-normalized image count",
        )

        m = folium.Map(location=[20, 0],
                       zoom_start=2,
                       tiles=MapConfig.get_tiles_url(),
                       attr=MapConfig.TILES_ATTR,
                       prefer_canvas=True)

        rows = db.session.query(
            Image.region_id,
            func.count(Image.id).label("image_count")
        ).group_by(Image.region_id).all()
        image_counts = {row.region_id: int(row.image_count) for row in rows}

        if not image_counts:
            folium.LayerControl().add_to(m)
            return m

        max_count = max(image_counts.values())
        max_log = math.log1p(max_count) if max_count > 0 else 0

        for region in all_regions:
            bbox = BoundingBox(region.min_lng, region.min_lat,
                               region.max_lng, region.max_lat)
            lng, lat = RegionManager.get_region_mid(bbox)
            coords.append([lat, lng])
            count = image_counts.get(region.id, 0)

            if max_log > 0:
                norm = math.log1p(count) / max_log
                radius = min_radius + (norm * (max_radius - min_radius))
            else:
                norm = 0
                radius = min_radius

            colour = colour_scale(norm)

            folium.CircleMarker(location=[lat, lng],
                                radius=radius,
                                color=colour,
                                fill=True,
                                fillColor=colour,
                                fillOpacity=1.0,
                                weight=1,
                                ).add_to(m)

        if coords:
            lats = [coord[0] for coord in coords]
            lngs = [coord[1] for coord in coords]
            m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]],
                         padding=(20, 20), max_zoom=6)

        m.add_child(colour_scale)

        folium.LayerControl().add_to(m)
        return m
