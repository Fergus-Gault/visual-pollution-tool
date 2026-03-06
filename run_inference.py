from src.model import YoloModel
from src.database import DatabaseManager
from src.pipeline import InferenceManager
from src.utils import setup_logger
from src.mapping import Mapper
import argparse


if __name__ == "__main__":
    logger = setup_logger(__name__)
    db = DatabaseManager()
    model = YoloModel()
    pipeline = InferenceManager(db, model)
    mapper = Mapper(db)

    parser = argparse.ArgumentParser(
        prog="Inference", description="Runs inference either on city, country, or all regions")
    parser.add_argument("--city", required=False, default=None)
    parser.add_argument("--country", required=False, default=None)

    args = parser.parse_args()

    if args.city is None and args.country is None:
        regions = db.get_all_regions()
    else:
        city = args.city.title() if args.city is not None else None
        country = args.country.title() if args.country is not None else None
        regions = db.get_region_by_city_and_country(city, country)

    if regions == [] or regions is None:
        logger.warning("No regions found.")

    for region in regions:
        logger.info(f"Running inference for {region.city}, {region.country}.")
        pipeline.run_inference(region)
        detections_map = mapper.map_region_detections(region)
        mapper.save(detections_map, region,
                    map_type="region_detections", file_type="html")
