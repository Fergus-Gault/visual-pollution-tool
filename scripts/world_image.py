import argparse
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager
from src.mapping import WorldImages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled", action="store_true")
    args = parser.parse_args()

    db = DatabaseManager()
    if args.scaled:
        m = WorldImages.map_world_images_scaled_by_count(db)
    else:
        m = WorldImages.map_world_images(db)

    m.save(Path("./maps/world_images.html"))
    img_data = m._to_png(5)
    img = PILImage.open(BytesIO(img_data))
    img.save("./maps/world_images.png")
