import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))


DEFAULT_OUTPUT = Path("plots/pollutant_shapes_and_locations.png")
DEFAULT_OUTPUT_CSV = Path("data/pollutant_shapes_and_locations.csv")

FALLBACK_COLORS = {
    "barrier": "#eab308",
    "billboard": "#ef4444",
    "bin": "#22c55e",
    "graffiti": "#a855f7",
    "mobile_advertisement": "#f97316",
    "pothole": "#78350f",
    "road_sign": "#3b82f6",
    "shop_sign": "#ec4899",
    "utility_pole": "#64748b",
    "other": "#6b7280",
}


def normalize(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def slugify(value):
    cleaned = []
    for char in value.strip().lower():
        cleaned.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(cleaned).split("_") if part)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="PlotPollutantShapesAndLocations",
        description=(
            "Plot each pollutant's detected bounding-box shape and location "
            "on a normalized image canvas."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output atlas image path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV path for normalized detection boxes. Use an empty string to skip.",
    )
    parser.add_argument(
        "--source",
        default="",
        help="Optional image source filter, e.g. mapillary. Empty includes all sources.",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Optional country filter applied through the parent region.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Optional city filter applied through the parent region.",
    )
    parser.add_argument(
        "--region-id",
        action="append",
        dest="region_ids",
        default=None,
        help="Optional region ID filter. Pass multiple times to include several regions.",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=None,
        help="Optional pollutant label to include. Pass multiple times for several labels.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Optional minimum detection confidence.",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=1,
        help="Minimum detections required for a pollutant to appear in the atlas.",
    )
    parser.add_argument(
        "--max-boxes-per-label",
        type=int,
        default=600,
        help="Maximum boxes drawn per pollutant label. Summary stats still use all rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling boxes for display.",
    )
    parser.add_argument(
        "--individual-dir",
        type=Path,
        default=None,
        help="Optional directory for one full-size plot per pollutant.",
    )
    return parser.parse_args()


def parse_bbox(value):
    if value is None:
        return None
    try:
        coords = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(coord) for coord in coords]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(coord) for coord in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def build_detection_query(db, args):
    from src.database import Detection, Image, Region

    query = db.session.query(
        Detection.id.label("detection_id"),
        Detection.label.label("label"),
        Detection.confidence.label("confidence"),
        Detection.bbox.label("bbox"),
        Image.id.label("image_id"),
        Image.width.label("image_width"),
        Image.height.label("image_height"),
        Image.source.label("source"),
        Image.region_id.label("region_id"),
        Region.city.label("city"),
        Region.country.label("country"),
    ).join(Image, Detection.image_id == Image.id).outerjoin(Region, Image.region_id == Region.id)

    query = query.filter(Detection.bbox.isnot(None))
    query = query.filter(Image.width.isnot(None), Image.height.isnot(None))
    query = query.filter(Image.width > 0, Image.height > 0)
    if args.source:
        query = query.filter(func.lower(func.coalesce(Image.source, "")) == normalize(args.source))
    if args.country:
        query = query.filter(func.lower(func.coalesce(Region.country, "")) == normalize(args.country))
    if args.city:
        query = query.filter(func.lower(func.coalesce(Region.city, "")) == normalize(args.city))
    if args.region_ids:
        query = query.filter(Image.region_id.in_(args.region_ids))
    if args.labels:
        labels = [normalize(label) for label in args.labels]
        query = query.filter(func.lower(func.coalesce(Detection.label, "")).in_(labels))
    if args.min_confidence is not None:
        query = query.filter(Detection.confidence >= args.min_confidence)

    return query.order_by(Detection.label, Detection.id)


def rows_to_frame(rows):
    records = []
    for row in rows:
        record = dict(row._mapping)
        width = float(record["image_width"])
        height = float(record["image_height"])
        bbox = parse_bbox(record["bbox"])
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        x1 = min(max(x1 / width, 0.0), 1.0)
        x2 = min(max(x2 / width, 0.0), 1.0)
        y1 = min(max(y1 / height, 0.0), 1.0)
        y2 = min(max(y2 / height, 0.0), 1.0)
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0.0 or box_height <= 0.0:
            continue

        record["label"] = (record["label"] or "other").strip() or "other"
        record["x1"] = x1
        record["y1"] = y1
        record["x2"] = x2
        record["y2"] = y2
        record["box_width"] = box_width
        record["box_height"] = box_height
        record["box_area"] = box_width * box_height
        record["aspect_ratio"] = box_width / box_height
        record["center_x"] = x1 + box_width / 2.0
        record["center_y"] = y1 + box_height / 2.0
        records.append(record)

    return pd.DataFrame.from_records(records)


def load_label_colors():
    try:
        from src.config import MapConfig
    except Exception:
        return FALLBACK_COLORS
    colors = dict(FALLBACK_COLORS)
    colors.update(getattr(MapConfig, "DETECTION_COLOURS", {}) or {})
    return colors


def label_order(frame, min_detections):
    counts = frame["label"].value_counts()
    labels = [label for label, count in counts.items() if count >= min_detections]
    if not labels:
        raise SystemExit("No pollutant labels met --min-detections.")
    return labels


def sample_for_display(subset, max_boxes, seed):
    if max_boxes is None or max_boxes <= 0 or len(subset) <= max_boxes:
        return subset
    return subset.sample(n=max_boxes, random_state=seed)


def display_alpha(display_count, base_alpha, minimum_alpha):
    if display_count <= 0:
        return base_alpha
    scaled = base_alpha * math.sqrt(400.0 / float(display_count))
    return min(base_alpha, max(minimum_alpha, scaled))


def draw_label_canvas(ax, subset, label, color, max_boxes, seed):
    display = sample_for_display(subset, max_boxes, seed)
    box_alpha = display_alpha(len(display), base_alpha=0.045, minimum_alpha=0.008)
    center_alpha = display_alpha(len(display), base_alpha=0.12, minimum_alpha=0.025)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f8fafc", edgecolor="#d1d5db", linewidth=1.0))

    for row in display.itertuples(index=False):
        ax.add_patch(
            Rectangle(
                (row.x1, row.y1),
                row.box_width,
                row.box_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.45,
                alpha=box_alpha,
            )
        )

    median = subset[["x1", "y1", "box_width", "box_height", "center_x", "center_y"]].median()
    ax.add_patch(
        Rectangle(
            (median["x1"], median["y1"]),
            median["box_width"],
            median["box_height"],
            facecolor="none",
            edgecolor="#111827",
            linewidth=2.0,
        )
    )
    ax.scatter(
        display["center_x"],
        display["center_y"],
        s=4,
        alpha=center_alpha,
        color="#111827",
        linewidths=0,
    )
    ax.scatter([median["center_x"]], [median["center_y"]], s=42, color="#111827", zorder=5)

    area_median = subset["box_area"].median() * 100.0
    aspect_median = subset["aspect_ratio"].median()
    ax.set_title(
        f"{label.replace('_', ' ').title()}\n"
        f"n={len(subset):,} | drawn={len(display):,} | median area={area_median:.2f}% | aspect={aspect_median:.2f}",
        fontsize=10,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=7, length=2)
    ax.set_xlabel("left to right", fontsize=8)
    ax.set_ylabel("top to bottom", fontsize=8)


def plot_atlas(frame, labels, output_path, max_boxes, seed):
    colors = load_label_colors()
    column_count = min(3, len(labels))
    row_count = math.ceil(len(labels) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.6 * column_count, 4.9 * row_count),
        squeeze=False,
    )

    for index, label in enumerate(labels):
        ax = axes.flat[index]
        subset = frame[frame["label"] == label].copy()
        color = colors.get(label, colors.get("other", "#6b7280"))
        draw_label_canvas(ax, subset, label, color, max_boxes, seed + index)

    for ax in axes.flat[len(labels):]:
        ax.axis("off")

    fig.suptitle(
        "Pollutant Detection Shape And Location Fingerprints",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_individuals(frame, labels, output_dir, max_boxes, seed):
    if output_dir is None:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = load_label_colors()
    for index, label in enumerate(labels):
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        subset = frame[frame["label"] == label].copy()
        color = colors.get(label, colors.get("other", "#6b7280"))
        draw_label_canvas(ax, subset, label, color, max_boxes, seed + index)
        fig.tight_layout()
        fig.savefig(output_dir / f"{slugify(label)}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    return len(labels)


def export_csv(frame, output_path):
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "detection_id",
        "image_id",
        "label",
        "confidence",
        "source",
        "region_id",
        "city",
        "country",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "box_width",
        "box_height",
        "box_area",
        "aspect_ratio",
    ]
    frame[columns].to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def main():
    from src.database import DatabaseManager

    args = parse_args()
    if args.min_detections < 1:
        raise SystemExit("--min-detections must be at least 1.")
    if args.max_boxes_per_label < 1:
        raise SystemExit("--max-boxes-per-label must be at least 1.")
    if args.min_confidence is not None and not 0.0 <= args.min_confidence <= 1.0:
        raise SystemExit("--min-confidence must be between 0 and 1.")

    db = DatabaseManager()
    rows = build_detection_query(db, args).all()
    if not rows:
        raise SystemExit("No detections matched the requested filters.")

    frame = rows_to_frame(rows)
    if frame.empty:
        raise SystemExit("No detections had usable image dimensions and bounding boxes.")

    labels = label_order(frame, args.min_detections)
    plotted = frame[frame["label"].isin(labels)].copy()
    output_csv = Path(args.output_csv) if str(args.output_csv).strip() else None

    export_csv(plotted, output_csv)
    plot_atlas(plotted, labels, args.output, args.max_boxes_per_label, args.seed)
    individual_count = plot_individuals(
        plotted,
        labels,
        args.individual_dir,
        args.max_boxes_per_label,
        args.seed,
    )

    print(f"Loaded {len(frame):,} usable detection(s).")
    print(f"Plotted {len(labels)} pollutant label(s): {', '.join(labels)}")
    print(f"Saved atlas to {args.output}")
    if output_csv is not None:
        print(f"Saved normalized boxes to {output_csv}")
    if individual_count:
        print(f"Saved {individual_count} individual pollutant plot(s) to {args.individual_dir}")


if __name__ == "__main__":
    main()
