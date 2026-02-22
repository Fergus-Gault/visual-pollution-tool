from src.model import train_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Model training", description="Trains a YOLO model")
    parser.add_argument("--path", "-p", default=None)
    parser.add_argument("--epochs", "-e", default=None)
    parser.add_argument("--imgsz", "-i", default=None)
    parser.add_argument("--device", "-d", default=None)
    parser.add_argument("--base-model", "-b", default=None)
    args = parser.parse_args()

    train_model(path=args.path, base_model=args.base_model,
                epochs=args.epochs, imgsz=args.imgsz, device=args.device)
