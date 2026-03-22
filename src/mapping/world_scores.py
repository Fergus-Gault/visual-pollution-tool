import folium
from branca.colormap import LinearColormap
from src.utils import setup_logger, RegionManager
from src.config import MapConfig
from src.database import DatabaseManager
from src.api import BoundingBox

logger = setup_logger(__name__)


class WorldScores:
    @staticmethod
    def map_world_scores_scaled_by_value(db: DatabaseManager, min_radius=1, max_radius=4):
        all_regions = db.get_all_regions()
        coords = []

        colour_scale = LinearColormap(
            colors=["#007425", "#fa7d00", "#d7191c"],
            vmin=0,
            vmax=1,
            caption="Normalized region score",
        )

        m = folium.Map(location=[20, 0],
                       zoom_start=2,
                       tiles=MapConfig.get_tiles_url(),
                       attr=MapConfig.TILES_ATTR,
                       prefer_canvas=True)

        scores = [
            float(region.score)
            for region in all_regions
            if region.score is not None and float(region.score) > 0.0
        ]
        if not scores:
            folium.LayerControl().add_to(m)
            return m

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        for region in all_regions:
            score = float(region.score) if region.score is not None else 0.0
            if score <= 0.0:
                continue

            bbox = BoundingBox(region.min_lng, region.min_lat,
                               region.max_lng, region.max_lat)
            lng, lat = RegionManager.get_region_mid(bbox)
            coords.append([lat, lng])

            if score_range > 0:
                norm = (score - min_score) / score_range
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
