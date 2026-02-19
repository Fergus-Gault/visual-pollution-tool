import folium
from PIL import Image as PILImage
from io import BytesIO
from src.database import DatabaseManager
from src.utils import setup_logger, RegionManager
from src.api import BoundingBox
from src.config import MapConfig

logger = setup_logger(__name__)


class Mapper:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def save_as_png(self, map, file_name):
        img_data = map._to_png(5)
        img = PILImage.open(BytesIO(img_data))
        img.save(file_name)

    def save_as_html(self, map, file_name):
        map.save(file_name)

    def map_region_images(self, regions):
        all_images = []
        if isinstance(regions, list):
            for region in regions:
                images = self.db.get_images_by_region(region.id)
                all_images.extend(images)
        else:
            images = self.db.get_images_by_region(regions.id)
            all_images.extend(images)

        if not all_images:
            logger.warning(
                f"No images found in any of the {len(regions)} regions.")
            return

        min_lng = min(r.min_lng for r in regions)
        min_lat = min(r.min_lat for r in regions)
        max_lng = max(r.max_lng for r in regions)
        max_lat = max(r.max_lat for r in regions)
        bbox = BoundingBox(min_lng, min_lat, max_lng, max_lat)

        centre = RegionManager.get_region_mid(bbox)

        m = folium.Map(location=[centre[1], centre[0]],
                       zoom_start=14, tiles=MapConfig.TILES)

        source_counts = {}
        for img in all_images:
            source = img.source
            source_counts[source] = source_counts.get(source, 0) + 1

        source_colours = {
            'mapillary': MapConfig.MAPILLARY_COLOURS,
            'kartaview': MapConfig.KARTAVIEW_COLOURS,
        }

        for img in all_images:
            color = source_colours.get(img.source)

            folium.CircleMarker(
                location=[img.lat, img.lng],
                radius=4,
                tooltip=f"{source}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=1
            ).add_to(m)

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

        source_legend_items = ''.join(
            [f'<i class="fa fa-circle" style="color:{source_colours.get(source)}"></i> {source.capitalize()}: {count}<br>'for source, count in source_counts.items()])

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
