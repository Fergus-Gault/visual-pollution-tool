import folium
from html import escape
from branca.colormap import LinearColormap

from src.api import BoundingBox
from src.config import MapConfig
from src.database import DatabaseManager
from src.pipeline.score import Scorer
from src.utils import RegionManager


class WorldScoreDifferences:
    @staticmethod
    def map_vpi_osm_differences(
        db: DatabaseManager,
        min_radius=1,
        max_radius=4,
        exclude_zero_scores=False,
    ):
        all_regions = db.get_all_regions()
        region_ids = [region.id for region in all_regions]
        scorer = Scorer(db)
        vpi_scores = scorer.score_regions(region_ids=region_ids)
        osm_scores = scorer.score_regions_with_osm_only(region_ids=region_ids)

        colour_scale = LinearColormap(
            colors=["#007425", "#fa7d00", "#d7191c"],
            vmin=0,
            vmax=1,
            caption="Normalized absolute VPI vs OSM score difference",
        )

        m = folium.Map(
            location=[20, 0],
            zoom_start=2,
            tiles=MapConfig.get_tiles_url(),
            attr=MapConfig.get_tiles_attr(),
            prefer_canvas=True,
        )

        rows = []
        for region in all_regions:
            vpi_score = vpi_scores.get(region.id, 0.0)
            osm_score = osm_scores.get(region.id, 0.0)
            if exclude_zero_scores and (vpi_score == 0.0 or osm_score == 0.0):
                continue
            difference = abs(osm_score - vpi_score)
            rows.append((region, vpi_score, osm_score, difference))

        positive_differences = [
            difference for _, _, _, difference in rows if difference > 0.0
        ]
        if not positive_differences:
            folium.LayerControl().add_to(m)
            return m

        max_difference = max(positive_differences)
        coords = []

        for region, vpi_score, osm_score, difference in rows:
            if difference <= 0.0:
                continue

            bbox = BoundingBox(
                region.min_lng,
                region.min_lat,
                region.max_lng,
                region.max_lat,
            )
            lng, lat = RegionManager.get_region_mid(bbox)
            coords.append([lat, lng])

            norm = difference / max_difference if max_difference > 0 else 0.0
            radius = min_radius + (norm * (max_radius - min_radius))
            colour = colour_scale(norm)
            city = escape(region.city or "Unknown region").replace("`", "&#96;")
            country = escape(region.country or "Unknown country").replace(
                "`", "&#96;")

            folium.CircleMarker(
                location=[lat, lng],
                radius=radius,
                color=colour,
                fill=True,
                fillColor=colour,
                fillOpacity=1.0,
                weight=1,
                tooltip=(
                    f"{city}, {country}"
                    f"<br>VPI score: {vpi_score:.6f}"
                    f"<br>OSM score: {osm_score:.6f}"
                    f"<br>Absolute difference: {difference:.6f}"
                ),
            ).add_to(m)

        if coords:
            lats = [coord[0] for coord in coords]
            lngs = [coord[1] for coord in coords]
            m.fit_bounds(
                [[min(lats), min(lngs)], [max(lats), max(lngs)]],
                padding=(20, 20),
                max_zoom=6,
            )

        m.add_child(colour_scale)
        folium.LayerControl().add_to(m)
        return m
