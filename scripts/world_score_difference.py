import argparse
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager
from src.mapping import WorldScoreDifferences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map the absolute difference between VPI-only and OSM-only scores."
    )
    parser.add_argument(
        "--exclude-zero-scores",
        action="store_true",
        help="Exclude regions where either the VPI-only or OSM-only score is zero.",
    )
    args = parser.parse_args()

    db = DatabaseManager()
    m = WorldScoreDifferences.map_vpi_osm_differences(
        db,
        exclude_zero_scores=args.exclude_zero_scores,
    )
    suffix = "_nonzero" if args.exclude_zero_scores else ""
    m.save(Path(f"./maps/world_score_differences{suffix}.html"))
    img_data = m._to_png(5)
    img = PILImage.open(BytesIO(img_data))
    img.save(f"./maps/world_score_differences{suffix}.png")
