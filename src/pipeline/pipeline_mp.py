from src.config import Config, PipelineConfig
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import dotenv_values
from src.utils import RegionManager, RateLimiter
import csv


def _run_region(row, pipeline, collect_only, override, region_method, dense_scan, fetch_osm):
    r = row
    city = r.get("city_ascii", "").strip() or None
    country = r.get("country", "").strip() or None
    population_raw = r.get("population", "").strip()
    population = int(float(population_raw)) if population_raw else None
    try:
        lng = float(r.get("lng", ""))
        lat = float(r.get("lat", ""))
    except (ValueError, TypeError):
        lng, lat = None, None

    if city and country:
        coords = RegionManager.geolocate_city(city, country)
        if coords is not None:
            lng, lat = coords

    if lng is None or lat is None:
        return
    pipeline._run_region_coords(
        lng=lng, lat=lat, city=city, country=country, population=population,
        collect_only=collect_only, override=override, region_method=region_method,
        dense_scan=dense_scan, fetch_osm=fetch_osm)


def _run_chunk(chunk, header, token, collect_only=False, override=False, region_method="shape", dense_scan=False, fetch_osm=True):
    from .pipeline import Pipeline
    from src.api import MapillaryAPI, KartaviewAPI
    rate_limiter = RateLimiter(max_calls=PipelineConfig.MAPILLARY_RATE_LIMIT)
    kv_rate_limiter = RateLimiter(
        max_calls=PipelineConfig.KARTAVIEW_RATE_LIMIT)
    mapillary = MapillaryAPI(token, rate_limiter=rate_limiter)
    pipeline = Pipeline(
        apis=[mapillary, KartaviewAPI(rate_limiter=kv_rate_limiter)])
    rows = [dict(zip(header, row)) for row in chunk]
    with ThreadPoolExecutor(max_workers=PipelineConfig.REGION_WORKERS) as executor:
        futures = [
            executor.submit(_run_region, row, pipeline, collect_only,
                            override, region_method, dense_scan, fetch_osm)
            for row in rows
        ]
        for future in as_completed(futures):
            future.result()


class PipelineMP:
    def __init__(self, file_path, collect_only=False, override=False, region_method="shape", dense_scan=False, fetch_osm=True):
        self.file_path = file_path
        self.collect_only = collect_only
        self.override = override
        self.region_method = region_method
        self.dense_scan = dense_scan
        self.fetch_osm = fetch_osm
        self.dotenv = dotenv_values(Config.ENV_PATH)
        self.tokens = [self.dotenv.get("MAPILLARY_ACCESS_TOKEN", None), self.dotenv.get(
            "EXTRA_TOKEN_1", None), self.dotenv.get("EXTRA_TOKEN_2", None), self.dotenv.get("EXTRA_TOKEN_3", None)]
        self.num_tokens = len(self.tokens) - self.tokens.count(None)

    def start_mp(self):
        file_chunks, header = self._split_file()
        processes = []
        for idx, chunk in enumerate(file_chunks):
            p = multiprocessing.Process(
                target=_run_chunk, args=(chunk, header, self.tokens[idx]),
                kwargs=dict(collect_only=self.collect_only, override=self.override,
                            region_method=self.region_method, dense_scan=self.dense_scan,
                            fetch_osm=self.fetch_osm))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

    def _split_file(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]
            rows = list(reader)
        lines_per_chunk = (len(rows) + self.num_tokens - 1) // self.num_tokens
        chunks = [rows[i:i + lines_per_chunk]
                  for i in range(0, len(rows), lines_per_chunk)]
        return chunks, header
