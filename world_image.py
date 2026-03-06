from src.mapping import WorldImages
from src.database import DatabaseManager
from pathlib import Path

if __name__ == "__main__":
    db = DatabaseManager()
    m = WorldImages.map_world_images(db)
    m.save(Path("./maps/world_images.html"))
