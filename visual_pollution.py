from src.pipeline import Pipeline
import sys

if __name__ == "__main__":
    pipeline = Pipeline()
    args = sys.argv
    if ".csv" in args[1] or ".txt" in args[1]:
        pipeline.run(file_path=args[1])
    else:
        pipeline.run(args=args)
