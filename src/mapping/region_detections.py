import folium
from .helper import MapHelper
from src.utils import setup_logger, RegionManager
from src.config import MapConfig
from src.database import DatabaseManager

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
            regions = [regions]
            detections = db.get_detections_by_region(regions[0].id)
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

            popup_html = f"""
                        <div style="font-family: Arial; width: 250px;">
                        <p style="margin: 10px 0 5px 0;"><a href="{det.image.url}" target="_blank">View Image</a></p>
                        <p><b>Detection: {det.label}</b></p>
                        <p>Confidence: {det.confidence}</p>
                        <p>Location: ({det.image.lat:.6f}, {det.image.lng:.6f})</p>
                        <p>Image Source: {det.image.source}</p>
                        </div>"""

            folium.CircleMarker(location=[det.image.lat, det.image.lng],
                                radius=4,
                                tooltip=f"{det.label}",
                                popup=folium.Popup(popup_html, max_width=300),
                                color=colour,
                                fill=True,
                                fillColor=colour,
                                fillOpacity=0.7,
                                weight=1
                                ).add_to(m)

        m = MapHelper.draw_region_bounds(m, regions)

        legend_html = """
            <div style="position: fixed;
                bottom: 50px; right: 50px; width: 250px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px; border-radius: 5px;">
            <b>Detection Labels</b><br>
            """
        for label, color in sorted(MapConfig.DETECTION_COLOURS.items()):
            legend_html += f'<i style="background:{color}; width: 18px; height: 18px; float: left; margin-right: 8px; border-radius: 50%;"></i>{label}<br>'
        legend_html += '</div>'

        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(m)

        return m
