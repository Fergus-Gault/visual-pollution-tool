import folium
from src.utils import setup_logger, RegionManager
from .helper import MapHelper
from src.config import MapConfig

logger = setup_logger(__name__)


class RegionImages:
    @staticmethod
    def map_region_images(db, regions):
        all_images = []
        if isinstance(regions, list):
            for region in regions:
                images = db.get_images_by_region(region.id)
                all_images.extend(images)
        else:
            regions = [regions]
            images = db.get_images_by_region(regions[0].id)
            all_images.extend(images)

        if not all_images:
            logger.warning(
                f"No images found in any of the {len(regions)} regions.")
            return

        _, centre = RegionManager.get_combined_bbox(regions)

        m = folium.Map(location=[centre[1], centre[0]],
                       zoom_start=MapConfig.ZOOM_START, tiles=MapConfig.TILES)

        source_counts = {}
        for img in all_images:
            source = img.source
            source_counts[source] = source_counts.get(source, 0) + 1

        for img in all_images:
            colour = MapConfig.SOURCE_COLOURS.get(img.source)

            popup_html = f"""
                        <div style="font-family: Arial; width: 250px;">
                        <p style="margin: 10px 0 5px 0;"><a href="{img.url}" target="_blank">View Image</a></p>
                        </div>"""

            folium.CircleMarker(
                location=[img.lat, img.lng],
                radius=4,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{img.source}",
                color=colour,
                fill=True,
                fillColor=colour,
                fillOpacity=0.7,
                weight=1
            ).add_to(m)

        m = MapHelper.draw_region_bounds(m, regions)

        source_legend_items = ''.join(
            [f'<i class="fa fa-circle" style="color:{MapConfig.SOURCE_COLOURS.get(source)}"></i> {source.capitalize()}: {count}<br>'for source, count in source_counts.items()])

        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 240px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
            <p style="margin: 0 0 10px 0;"><strong>Combined Regions ({len(regions)})</strong></p>
            <hr style="margin: 5px 0;">
            <p style="margin: 5px 0; font-size: 12px;">
                <strong>Total Images:</strong> {len(all_images)}<br>
            </p>
            <p style="margin: 10px 0 5px 0; font-size: 12px;">
                <strong>By Source:</strong><br>
                {source_legend_items}
            </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(m)
        return m
