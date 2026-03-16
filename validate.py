from src.model import validate_model
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Model validation", description="Runs YOLO validation on the test split")
    parser.add_argument("--path", "-p", default=None)
    parser.add_argument("--imgsz", "-i", default=None)
    parser.add_argument("--device", "-d", default=None)
    parser.add_argument("--model-path", "-m", default="data/model/best.pt")
    parser.add_argument("--split", "-s", default="test")
    args = parser.parse_args()

    validate_model(path=args.path, model_path=args.model_path,
                   imgsz=args.imgsz, device=args.device, split=args.split)
