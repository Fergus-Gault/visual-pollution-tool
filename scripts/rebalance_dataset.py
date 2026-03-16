import sys
import json
import random
import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TrainConfig

def load_ndjson(path: Path):
    meta = None
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "dataset":
                meta = obj
            else:
                records.append(obj)
    return meta, records


def split_dataset(records: list[dict]) -> dict[str, list[dict]]:
    n = len(records)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    n_train = int(n * TrainConfig.TRAIN_SPLIT)
    n_val = int(n * TrainConfig.VAL_SPLIT)
    n_test = n - n_train - n_val

    target_sizes = {"train": n_train, "val": n_val, "test": n_test}
    split_ratios = {
        "train": TrainConfig.TRAIN_SPLIT,
        "val": TrainConfig.VAL_SPLIT,
        "test": TrainConfig.TEST_SPLIT,
    }

    rng = random.Random(42)
    data = list(records)
    rng.shuffle(data)
    data.sort(
        key=lambda rec: len(rec.get("annotations", {}).get("boxes", [])),
        reverse=True,
    )

    total_class_counts = Counter()
    image_class_counts = []
    for rec in data:
        class_counts = Counter()
        for box in rec.get("annotations", {}).get("boxes", []):
            if not box:
                continue
            class_counts[int(box[0])] += 1
        image_class_counts.append(class_counts)
        total_class_counts.update(class_counts)

    target_class_counts = {
        split: {
            cls: total * split_ratios[split]
            for cls, total in total_class_counts.items()
        }
        for split in ("train", "val", "test")
    }

    split_data: dict[str, list] = {"train": [], "val": [], "test": []}
    current_class_counts = {
        "train": Counter(), "val": Counter(), "test": Counter()}

    for rec, class_counts in zip(data, image_class_counts):
        best_split = None
        best_score = None

        for split in ("train", "val", "test"):
            if len(split_data[split]) >= target_sizes[split]:
                continue

            score = 0.0
            for cls, count in class_counts.items():
                deficit = (
                    target_class_counts[split][cls]
                    - current_class_counts[split][cls]
                )
                if deficit > 0:
                    score += min(deficit, count)

            remaining_capacity = target_sizes[split] - len(split_data[split])
            tiebreak = remaining_capacity / max(1, target_sizes[split])
            candidate = (score, tiebreak)

            if best_score is None or candidate > best_score:
                best_score = candidate
                best_split = split

        if best_split is None:
            best_split = min(
                ("train", "val", "test"), key=lambda s: len(split_data[s])
            )

        split_data[best_split].append(rec)
        current_class_counts[best_split].update(class_counts)

    return split_data


def write_ndjson(path: Path, meta: dict, split_data: dict[str, list[dict]]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for split, records in split_data.items():
            for rec in records:
                rec = dict(rec)
                rec["split"] = split
                f.write(json.dumps(rec) + "\n")


def print_class_distribution(split_data: dict[str, list[dict]], class_names: dict):
    for split, records in split_data.items():
        counts = Counter()
        for rec in records:
            for box in rec.get("annotations", {}).get("boxes", []):
                if box:
                    counts[int(box[0])] += 1
        label_counts = {class_names.get(str(cls), str(
            cls)): n for cls, n in sorted(counts.items())}
        print(f"  {split} ({len(records)} images): {label_counts}")


def main():
    parser = argparse.ArgumentParser(
        description="Rebalance an existing ndjson dataset file according to configured train/val/test splits."
    )
    parser.add_argument("input", type=Path, help="Path to input .ndjson file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write rebalanced .ndjson (defaults to overwriting input)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    out_path = args.output or args.input

    meta, records = load_ndjson(args.input)
    if meta is None:
        print("Error: no dataset metadata line found in ndjson.")
        sys.exit(1)

    print(f"Loaded {len(records)} records from {args.input}")
    print(
        f"Rebalancing with splits: train={TrainConfig.TRAIN_SPLIT}, "
        f"val={TrainConfig.VAL_SPLIT}, test={TrainConfig.TEST_SPLIT}"
    )

    split_data = split_dataset(records)

    class_names = meta.get("class_names") or TrainConfig.LABELS_INV
    print("\nClass distribution after rebalancing:")
    print_class_distribution(split_data, class_names)

    write_ndjson(out_path, meta, split_data)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
