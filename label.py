from src.database import DatabaseManager
from src.labelling import LabelStudioClient

if __name__ == "__main__":
    db = DatabaseManager()
    ls = LabelStudioClient(db)
    ls.upload()
