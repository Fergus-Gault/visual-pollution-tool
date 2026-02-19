import folium


class MapHelper:
    @staticmethod
    def draw_region_bounds(m, regions):
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
        return m
