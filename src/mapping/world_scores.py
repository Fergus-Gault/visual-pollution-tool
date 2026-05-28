import math
import unicodedata
from csv import DictReader
from pathlib import Path
from statistics import quantiles

import folium
from branca.colormap import StepColormap
from src.utils import setup_logger, RegionManager
from src.config import MapConfig
from src.database import DatabaseManager
from src.api import BoundingBox

logger = setup_logger(__name__)


class WorldScores:
    @staticmethod
    def percentile(sorted_values, fraction):
        if not sorted_values:
            return None
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        position = max(0.0, min(1.0, float(fraction))) * \
            (len(sorted_values) - 1)
        lower_index = int(math.floor(position))
        upper_index = int(math.ceil(position))
        if lower_index == upper_index:
            return float(sorted_values[lower_index])
        lower_value = float(sorted_values[lower_index])
        upper_value = float(sorted_values[upper_index])
        return lower_value + (upper_value - lower_value) * (position - lower_index)

    @staticmethod
    def score_palette_color(fraction):
        green_rgb = (0x00, 0x74, 0x25)
        amber_rgb = (0xFA, 0x7D, 0x00)
        red_rgb = (0xD7, 0x19, 0x1C)

        clamped_fraction = max(0.0, min(1.0, float(fraction)))
        if clamped_fraction <= 0.5:
            mix = clamped_fraction / 0.5
            start_rgb, end_rgb = green_rgb, amber_rgb
        else:
            mix = (clamped_fraction - 0.5) / 0.5
            start_rgb, end_rgb = amber_rgb, red_rgb

        rgb = tuple(
            int(round(start_channel + (end_channel - start_channel) * mix))
            for start_channel, end_channel in zip(start_rgb, end_rgb)
        )
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    @staticmethod
    def build_score_scale(scores, band_count=10):
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            max_score = min_score + 1e-9

        sorted_scores = sorted(scores)
        effective_band_count = max(2, min(int(band_count), len(sorted_scores)))
        thresholds = [min_score]
        for band_index in range(1, effective_band_count):
            thresholds.append(
                float(WorldScores.percentile(sorted_scores,
                      band_index / effective_band_count))
            )
        thresholds.append(max_score)

        for index in range(1, len(thresholds)):
            if thresholds[index] <= thresholds[index - 1]:
                thresholds[index] = thresholds[index - 1] + 1e-9

        colors = [
            WorldScores.score_palette_color(
                band_index / max(1, effective_band_count - 1)
            )
            for band_index in range(effective_band_count)
        ]
        colour_scale = StepColormap(
            colors=colors,
            index=thresholds,
            vmin=min_score,
            vmax=max_score,
        )
        colour_scale.max_labels = 8
        colour_scale.text_color = "#111111"
        colour_scale.width = 420
        colour_scale.caption = (
            f"Region score percentile bands ({effective_band_count} bands, green = low, red = high)"
        )
        return colour_scale

    @staticmethod
    def normalize_place_name(value):
        text = (value or "").strip()
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text)
        text = "".join(
            char for char in text if not unicodedata.combining(char))
        return " ".join(text.casefold().split())

    @staticmethod
    def load_ghs_density_lookup():
        csv_path = Path(__file__).resolve(
        ).parents[2] / "data" / "GHS_UCDB_GLOBE_R2024A.csv"
        if not csv_path.exists():
            logger.warning(
                f"GHS density CSV not found at {csv_path}; falling back to estimated densities.")
            return {}, {}

        by_city_country = {}
        by_city = {}
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = DictReader(handle)
            for row in reader:
                city_key = WorldScores.normalize_place_name(
                    row.get("GC_UCN_MAI_2025"))
                country_key = WorldScores.normalize_place_name(
                    row.get("GC_CNT_GAD_2025"))
                density_raw = row.get("GH_XST_D30_2025")
                try:
                    density = float(density_raw)
                except (TypeError, ValueError):
                    continue
                if density <= 0.0 or not city_key:
                    continue

                by_city_country[(city_key, country_key)] = density
                by_city.setdefault(city_key, set()).add((country_key, density))
        return by_city_country, by_city

    @staticmethod
    def estimate_bbox_area_km2(region):
        min_lng = float(region.min_lng)
        max_lng = float(region.max_lng)
        min_lat = float(region.min_lat)
        max_lat = float(region.max_lat)

        lat_height_km = max(0.0, max_lat - min_lat) * 111.32
        mid_lat = (min_lat + max_lat) / 2.0
        lng_width_km = max(0.0, max_lng - min_lng) * \
            111.32 * math.cos(math.radians(mid_lat))
        return max(lat_height_km * max(lng_width_km, 0.0), 0.0)

    @staticmethod
    def map_world_scores_scaled_by_value(db: DatabaseManager, min_radius=1.0, max_radius=5.8):
        all_regions = db.get_all_regions()
        coords = []
        ghs_density_by_city_country, ghs_density_by_city = WorldScores.load_ghs_density_lookup()

        m = folium.Map(location=[20, 0],
                       zoom_start=2,
                       tiles=MapConfig.get_tiles_url(),
                       attr=MapConfig.get_tiles_attr(),
                       prefer_canvas=True)

        scores = [
            float(region.score)
            for region in all_regions
            if region.score is not None and float(region.score) > 0.0
        ]
        if not scores:
            folium.LayerControl().add_to(m)
            return m

        colour_scale = WorldScores.build_score_scale(scores)

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        density_by_region = {}
        densities = []

        for region in all_regions:
            city_key = WorldScores.normalize_place_name(region.city)
            country_key = WorldScores.normalize_place_name(region.country)
            density = ghs_density_by_city_country.get((city_key, country_key))

            if density is None and city_key:
                city_matches = ghs_density_by_city.get(city_key, set())
                if len(city_matches) == 1:
                    density = next(iter(city_matches))[1]

            if density is None:
                population = (
                    float(region.population)
                    if region.population is not None and float(region.population) > 0.0
                    else None
                )
                if population is None:
                    continue
                area_km2 = WorldScores.estimate_bbox_area_km2(region)
                if area_km2 <= 0.0:
                    continue
                density = population / area_km2

            density_by_region[region.id] = density
            if density > 0.0:
                densities.append(density)

        density_floor = None
        density_ceiling = None
        density_range = None
        if densities:
            if len(densities) >= 20:
                ventiles = quantiles(densities, n=20, method="inclusive")
                density_floor = ventiles[0]
                density_ceiling = ventiles[-1]
            else:
                density_floor = min(densities)
                density_ceiling = max(densities)

            if density_ceiling <= density_floor:
                density_floor = min(densities)
                density_ceiling = max(densities)
            density_range = density_ceiling - density_floor

        for region in all_regions:
            score = float(region.score) if region.score is not None else 0.0
            if score <= 0.0:
                continue

            bbox = BoundingBox(region.min_lng, region.min_lat,
                               region.max_lng, region.max_lat)
            lng, lat = RegionManager.get_region_mid(bbox)
            coords.append([lat, lng])

            if density_range is not None and density_range > 0:
                density = density_by_region.get(region.id, density_floor)
                clipped_density = min(
                    max(density, density_floor), density_ceiling)
                norm_radius = (clipped_density - density_floor) / density_range
                # Expand visible differences in the dense mid-range without letting
                # extreme megacities dominate the scale.
                norm_radius = norm_radius ** 0.75
                radius = min_radius + (norm_radius * (max_radius - min_radius))
            elif density_range == 0 and densities:
                radius = (min_radius + max_radius) / 2.0
            elif score_range > 0:
                norm = (score - min_score) / score_range
                radius = min_radius + (norm * (max_radius - min_radius))
            else:
                radius = min_radius

            if score_range > 0:
                norm = (score - min_score) / score_range
            else:
                norm = 0

            colour = colour_scale(score)

            folium.CircleMarker(location=[lat, lng],
                                radius=radius,
                                color=colour,
                                fill=True,
                                fillColor=colour,
                                fillOpacity=0.55,
                                weight=1,
                                opacity=0.65,
                                tooltip=(
                                    f"{region.city or 'Unknown'} | "
                                    f"score={score:.4f} | "
                                    f"density={density_by_region.get(region.id, 0.0):.1f}"
            ),
            ).add_to(m)

        if coords:
            lats = [coord[0] for coord in coords]
            lngs = [coord[1] for coord in coords]
            m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]],
                         padding=(20, 20), max_zoom=6)

        m.add_child(colour_scale)

        folium.LayerControl().add_to(m)
        return m
