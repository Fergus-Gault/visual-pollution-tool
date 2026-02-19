import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.database.database import DatabaseManager


if __name__ == "__main__":
    db = DatabaseManager()
    args = sys.argv

    region_id = args[1]
    db.delete_region(region_id)
