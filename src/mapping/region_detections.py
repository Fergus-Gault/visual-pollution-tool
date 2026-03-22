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
                       zoom_start=MapConfig.ZOOM_START,
                       tiles=MapConfig.get_tiles_url(),
                       attr=MapConfig.TILES_ATTR)
        detection_counts = {}
        for det in all_detections:
            det_t = det.label
            detection_counts[det_t] = detection_counts.get(det_t, 0) + 1

        grouped = {}
        for det in all_detections:
            grouped.setdefault(det.image_id, []).append(det)

        for _, dets in grouped.items():
            image = dets[0].image
            best = max(dets, key=lambda d: d.confidence)
            colour = MapConfig.DETECTION_COLOURS.get(
                best.label, MapConfig.OTHER_COLOUR)
            opacity = 1.0 if len(dets) > 1 else 0.7

            det_rows = "".join(
                f"<p style='margin:2px 0'><b>{d.label}</b>: {d.confidence:.2f}</p>"
                for d in sorted(dets, key=lambda d: d.confidence, reverse=True)
            )
            popup_html = f"""
                        <div style="font-family: Arial; width: 250px;">
                        <p style="margin: 10px 0 5px 0;"><a href="{image.url}" target="_blank">View Image</a></p>
                        {det_rows}
                        <p>Location: ({image.lat:.6f}, {image.lng:.6f})</p>
                        <p>Image Source: {image.source}</p>
                        </div>"""

            tooltip = ", ".join(
                f"{d.label} ({d.confidence:.2f})"
                for d in sorted(dets, key=lambda d: d.confidence, reverse=True)
            )

            folium.CircleMarker(location=[image.lat, image.lng],
                                radius=4,
                                tooltip=tooltip,
                                popup=folium.Popup(popup_html, max_width=300),
                                color=colour,
                                fill=True,
                                fillColor=colour,
                                fillOpacity=opacity,
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
