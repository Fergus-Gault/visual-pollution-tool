import folium
import math
from branca.colormap import LinearColormap
from sqlalchemy import func
from src.utils import RegionManager
from src.config import MapConfig
from src.database import DatabaseManager, OSMFeature
from src.api import BoundingBox


class WorldOSM:
    @staticmethod
    def map_world_osm(db: DatabaseManager):
        all_regions = [r for r in db.get_all_regions() if not r.dense_scan]
        coords = []

        m = folium.Map(location=[20, 0],
                       zoom_start=2,
                       tiles=MapConfig.get_tiles_url(),
                       attr=MapConfig.get_tiles_attr(),
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
    def map_osm_scaled_by_count(db: DatabaseManager, min_radius=1, max_radius=4):
        all_regions = [r for r in db.get_all_regions() if not r.dense_scan]
        included_region_ids = {region.id for region in all_regions}
        coords = []

        colour_scale = LinearColormap(
            colors=["#d7191c", "#fa7d00", "#007425"],
            vmin=0,
            vmax=1,
            caption="Log-normalized OSM feature count",
        )

        m = folium.Map(location=[20, 0],
                       zoom_start=2,
                       tiles=MapConfig.get_tiles_url(),
                       attr=MapConfig.get_tiles_attr(),
                       prefer_canvas=True)

        rows = db.session.query(
            OSMFeature.region_id,
            func.count(OSMFeature.id).label("osm_count")
        ).group_by(OSMFeature.region_id).all()
        osm_counts = {
            row.region_id: int(row.osm_count)
            for row in rows
            if row.region_id in included_region_ids
        }

        if not osm_counts:
            folium.LayerControl().add_to(m)
            return m

        max_count = max(osm_counts.values())
        max_log = math.log1p(max_count) if max_count > 0 else 0

        for region in all_regions:
            bbox = BoundingBox(region.min_lng, region.min_lat,
                               region.max_lng, region.max_lat)
            lng, lat = RegionManager.get_region_mid(bbox)
            coords.append([lat, lng])
            count = osm_counts.get(region.id, 0)

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
                                weight=1
                                ).add_to(m)

        if coords:
            lats = [coord[0] for coord in coords]
            lngs = [coord[1] for coord in coords]
            m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]],
                         padding=(20, 20), max_zoom=6)

        m.add_child(colour_scale)

        folium.LayerControl().add_to(m)
        return m
