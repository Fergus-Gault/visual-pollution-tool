from src.mapping import WorldOSM
from src.database import DatabaseManager
from pathlib import Path
from PIL import Image as PILImage
from io import BytesIO
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled", action="store_true")
    args = parser.parse_args()

    db = DatabaseManager()
    if args.scaled:
        m = WorldOSM.map_osm_scaled_by_count(db)
    else:
        m = WorldOSM.map_world_osm(db)
    m.save(Path("./maps/world_osm.html"))
    img_data = m._to_png(5)
    img = PILImage.open(BytesIO(img_data))
    img.save("./maps/world_osm.png")
