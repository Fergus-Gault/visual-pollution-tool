import folium
from src.utils import setup_logger, RegionManager
from src.config import MapConfig
from src.database import DatabaseManager
from .map import MapHelper

logger = setup_logger(__name__)


class RegionDetections:
    @staticmethod
    def map_region_detections(db: DatabaseManager, regions):
        all_detections = []
        if isinstance(regions, list):
            for region in regions:
                detections = db.get_detections_by_region(region.id)
                all_detections.extend(detections)
        else:
            regions = list(regions)
            detections = db.get_detections_by_region(region.id)
            all_detections.extend(detections)

        if not all_detections:
            logger.warning(
                f"No detections found in any of the {len(regions)} regions.")
            return

        _, centre = RegionManager.get_combined_bbox(regions)

        m = folium.Map(location=[centre[1], centre[0]],
                       zoom_start=MapConfig.ZOOM_START, tile=MapConfig.TILES)

        detection_counts = {}
        for det in all_detections:
            det_t = det.label
            detection_counts[det_t] = detection_counts.get(det_t, 0) + 1

        for det in all_detections:
            colour = MapConfig.DETECTION_COLOURS.get(
                det.label, MapConfig.OTHER_COLOUR)

            folium.CircleMarker(location=[det.lat, det.lng],
                                radius=4,
                                tooltip=f"{det.label}",
                                color=colour,
                                fill=True,
                                fillColor=colour,
                                fillOpacity=0.7,
                                weight=1
                                ).add_to(m)

        m = MapHelper.draw_region_bounds(regions)

        # TODO: add legend html
        # det_legend_items = ...
        # legend_html = ...

        # m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(m)

        return m
