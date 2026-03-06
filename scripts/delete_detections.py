import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.database.models import Detection, Image
from src.database.database import DatabaseManager
from sqlalchemy import delete, update


if __name__ == "__main__":
    db = DatabaseManager()
    db.session.execute(delete(Detection))
    db.session.execute(update(Image).where(
        Image.status == "reviewed").values(status="unreviewed"))
    db.session.commit()
