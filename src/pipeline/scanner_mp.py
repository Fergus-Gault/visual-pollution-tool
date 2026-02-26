from .scanner import Scanner
from src.database import DatabaseManager
from src.config import Config
from uuid import uuid4
import multiprocessing
from pathlib import Path
from dotenv import dotenv_values
from src.api import MapillaryAPI, KartaviewAPI
import csv


class ScannerMP:
    def __init__(self, file_path, db_base_path=None):
        self.file_path = file_path
        self.db_base_path = db_base_path or Path(f"data/dbs_{uuid4()[:8]}/")
        self.dotenv = dotenv_values(Config.ENV_PATH)
        self.tokens = [self.dotenv.get("MAPILLARY_ACCESS_TOKEN", None), self.dotenv.get(
            "EXTRA_TOKEN_1", None), self.dotenv.get("EXTRA_TOKEN_2", None), self.dotenv.get("EXTRA_TOKEN_3", None)]
        self.num_tokens = len(self.tokens) - self.tokens.count(None)

    def start_mp(self):
        file_chunks = self._split_file()
        processes = []
        for idx, chunk in enumerate(file_chunks):
            db = self._create_db(idx)
            new_mapillary = MapillaryAPI(self.tokens[idx])
            new_kartaview = KartaviewAPI()
            new_scanner = Scanner(db, [new_mapillary, new_kartaview])
            # TODO: This should probably point to a pipeline_mp version, so it runs the file chunk instead of region at a time
            # Might be easier to just make a pipeline_mp instead of scanner_mp
            new_process = multiprocessing.Process()

        pass

    def _split_file(self):
        if self.file_path.endswith(".csv"):
            with csv.reader(self.file_path) as reader:
                pass

    def _create_db(self, idx):
        return DatabaseManager(f"{self.db_base_path}/db_{idx}.sqlite3")
