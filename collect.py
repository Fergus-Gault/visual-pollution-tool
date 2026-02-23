from src.pipeline import Pipeline
from src.config import Config, ArgsConfig
import sys

if __name__ == "__main__":
    pipeline = Pipeline()
    args = sys.argv
    collect_only = False
    override = False
    region_method = "shape"
    dense_scan = False
    if ArgsConfig.DEBUG in args:
        Config.DEBUG = True
    if ArgsConfig.COLLECT_ONLY in args:
        collect_only = True
    if ArgsConfig.OVERRIDE in args:
        override = True
    if ArgsConfig.REGION_COLLECT in args:
        region_method = "region"
    if ArgsConfig.DENSE in args:
        dense_scan = True
    if ".csv" in args[1] or ".txt" in args[1]:
        pipeline.run(
            file_path=args[1], collect_only=collect_only, override=override, region_method=region_method, dense_scan=dense_scan)
    else:
        pipeline.run(args=args, collect_only=collect_only,
                     override=override, region_method=region_method, dense_scan=dense_scan)
