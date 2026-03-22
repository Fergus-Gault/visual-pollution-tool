from src.mapping import WorldScores
from src.database import DatabaseManager
from pathlib import Path
from PIL import Image as PILImage
from io import BytesIO


if __name__ == "__main__":
    db = DatabaseManager()
    m = WorldScores.map_world_scores_scaled_by_value(db)
    m.save(Path("./maps/world_scores.html"))
    img_data = m._to_png(5)
    img = PILImage.open(BytesIO(img_data))
    img.save("./maps/world_scores.png")
