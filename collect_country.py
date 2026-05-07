import argparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.config import Config
from src.pipeline import Pipeline
from src.utils import RegionManager, setup_logger

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Country collection",
        description="Split a country's bounds into connected subregions, scan each one, and run inference.",
    )
    parser.add_argument("country", help="Country name to geocode via OSMnx.")
    parser.add_argument("--subregions", "-sr", type=int, default=1000)
    parser.add_argument("--region-mode", choices=["land", "uniform"], default="land",
                        help="Use land-aware subdivision or a uniform full-bbox grid.")
    parser.add_argument("--image-sources", "-is", default="mapillary",
                        help="Comma-separated image sources: mapillary, kartaview, or both.")
    parser.add_argument("--collect-only", "-co", action="store_true")
    parser.add_argument("--override", "-or", action="store_true")
    parser.add_argument("--dense", "-dn", action="store_true")
    parser.add_argument("--map", action="store_true",
                        help="Generate region image/detection maps for each subregion.")
    parser.add_argument("--no-fetch-osm", action="store_true",
                        help="Disable OSM feature collection for each subregion.")
    parser.add_argument("--start-captured-at", default=None)
    parser.add_argument("--end-captured-at", default=None)
    parser.add_argument("--debug", "-d", action="store_true")
    return parser.parse_args()


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
        )

    logger.info(
        f"Scanning {len(subregions)} {args.region_mode} subregions for {args.country}.")
    pipeline = Pipeline(image_sources=args.image_sources)
    for idx, subregion in enumerate(tqdm(subregions, desc="Scanning country grid"), start=1):
        pipeline.run_bbox(
            bbox=subregion,
            gdf=gdf,
            city=f"{args.country} grid {idx:04d}",
            country=args.country,
            collect_only=args.collect_only,
            override=args.override,
            dense_scan=args.dense,
            fetch_osm=not args.no_fetch_osm,
            start_captured_at=args.start_captured_at,
            end_captured_at=args.end_captured_at,
            make_maps=args.map,
        )


if __name__ == "__main__":
    main()
