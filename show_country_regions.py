import argparse
import csv
import re
from pathlib import Path

import folium

from src.config import Config, MapConfig
from src.utils import RegionManager, setup_logger

logger = setup_logger(__name__)


def slugify(value):
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return text.lower() or "country"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Country region preview",
        description="Preview the exact connected subregions that would be created for a country.",
    )
    parser.add_argument("country", help="Country name to geocode via OSMnx.")
    parser.add_argument("--subregions", "-sr", type=int, default=1000)
    parser.add_argument("--region-mode", choices=["land", "uniform"], default="land",
                        help="Use land-aware subdivision or a uniform full-bbox grid.")
    parser.add_argument("--min-land-fraction", type=float, default=0.01,
                        help="In land mode, skip grid cells where less than this fraction overlaps the country shape.")
    parser.add_argument("--land-filter", choices=["center", "overlap"], default="center",
                        help="In land mode, use fast centre-point filtering or slower area-overlap filtering.")
    parser.add_argument("--output-html", default=None,
                        help="Optional HTML output path.")
    parser.add_argument("--output-csv", default=None,
                        help="Optional CSV output path.")
    parser.add_argument("--debug", "-d", action="store_true")
    return parser.parse_args()


def build_map(country, gdf, subregions, country_bbox, region_mode):
    centre_lng, centre_lat = RegionManager.get_region_mid(country_bbox)
    m = folium.Map(
        location=[centre_lat, centre_lng],
        zoom_start=MapConfig.ZOOM_START,
        tiles=MapConfig.get_tiles_url(),
        attr=MapConfig.get_tiles_attr(),
    )

    folium.GeoJson(
        gdf.__geo_interface__,
        name="Country shape",
        style_function=lambda _: {
            "color": "#0b6e4f",
            "weight": 2,
            "fillColor": "#52b788",
            "fillOpacity": 0.12,
        },
        tooltip=country,
    ).add_to(m)

    for idx, bbox in enumerate(subregions, start=1):
        tooltip = (
            f"{country} grid {idx:04d}<br>"
            f"{bbox.min_lng:.6f}, {bbox.min_lat:.6f}, "
            f"{bbox.max_lng:.6f}, {bbox.max_lat:.6f}"
        )
        folium.Rectangle(
            bounds=[[bbox.min_lat, bbox.min_lng], [bbox.max_lat, bbox.max_lng]],
            color="#1d3557",
            weight=1,
            fill=True,
            fill_opacity=0.04,
            tooltip=tooltip,
        ).add_to(m)

    legend_html = f"""
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 260px; height: auto;
                background-color: white; border: 2px solid grey; z-index: 9999;
                font-size: 14px; padding: 10px;">
        <p style="margin: 0 0 10px 0;"><strong>{country}</strong></p>
        <p style="margin: 0; font-size: 12px;">
            Region mode: {region_mode}<br>
            Subregions: {len(subregions)}<br>
            Hover a box to inspect its bbox.
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m


def write_csv(csv_path, country, subregions, region_mode):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "name", "country", "region_mode", "min_lng", "min_lat", "max_lng", "max_lat"])
        for idx, bbox in enumerate(subregions, start=1):
            writer.writerow([
                idx,
                f"{country} grid {idx:04d}",
                country,
                region_mode,
                bbox.min_lng,
                bbox.min_lat,
                bbox.max_lng,
                bbox.max_lat,
            ])


def main():
    args = parse_args()
    if args.debug:
        Config.DEBUG = True

    gdf = RegionManager.get_shape_file(args.country)
    if gdf is None:
        raise ValueError(f"Could not resolve a shape for country '{args.country}'.")

    country_bbox = RegionManager.bbox_from_shape(gdf)
    if args.region_mode == "uniform":
        subregions = RegionManager.get_connected_grid_subregions(
            country_bbox,
            args.subregions,
        )
    else:
        subregions = RegionManager.get_land_aware_subregions(
            gdf,
            args.subregions,
            min_land_fraction=args.min_land_fraction,
            country=args.country,
            land_filter=args.land_filter,
        )

    slug = slugify(args.country)
    html_path = Path(args.output_html) if args.output_html else Path(
        f"maps/country_previews/{slug}_{args.region_mode}_{len(subregions)}_regions.html"
    )
    csv_path = Path(args.output_csv) if args.output_csv else Path(
        f"data/country_previews/{slug}_{args.region_mode}_{len(subregions)}_regions.csv"
    )

    preview_map = build_map(args.country, gdf, subregions, country_bbox, args.region_mode)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    preview_map.save(html_path)
    write_csv(csv_path, args.country, subregions, args.region_mode)

    logger.info(f"Saved country region preview map to {html_path}")
    logger.info(f"Saved country region bounding boxes to {csv_path}")


if __name__ == "__main__":
    main()
