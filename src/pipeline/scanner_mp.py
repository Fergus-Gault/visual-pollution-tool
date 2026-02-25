from .scanner import Scanner
from src.database import DatabaseManager
from src.config import Config
from uuid import uuid4
import multiprocessing
from pathlib import Path
from dotenv import dotenv_values
import csv


class ScannerMP:
    def __init__(self, file_path, db_base_path=None):
        self.file_path = file_path
        self.db_base_path = db_base_path or Path(f"data/dbs_{uuid4()[:8]}/")
        self.dotenv = dotenv_values(Config.ENV_PATH)
        self.tokens = [self.dotenv.get("MAPILLARY_ACCESS_TOKEN"), self.dotenv.get(
            "EXTRA_TOKEN_1"), self.dotenv.get("EXTRA_TOKEN_2"), self.dotenv.get("EXTRA_TOKEN_3")]

    def start_mp(self):
        file_chunks = self._split_file()
        for idx, chunk in enumerate(file_chunks):
            db = self._create_db(idx)
        pass

    def _split_file(self):
        num_splits = len(self.tokens)
        if self.file_path.endswith(".csv"):
            with csv.reader(self.file_path) as reader:
                pass

    def _create_db(self, idx):
        return DatabaseManager(f"{self.db_base_path}/db_{idx}.sqlite3")
