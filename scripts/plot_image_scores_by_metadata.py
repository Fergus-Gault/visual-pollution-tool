import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))


DEFAULT_OUTPUT = Path("plots/image_scores_by_metadata.png")
DEFAULT_OUTPUT_CSV = Path("data/image_scores_by_metadata.csv")

METRIC_LABELS = {
    "altitude": "Altitude",
    "computed_altitude": "Computed altitude",
    "quality_score": "Quality score",
    "compass_angle": "Compass angle",
    "computed_compass_angle": "Computed compass angle",
    "rotation_x": "Computed rotation[0]",
    "rotation_y": "Computed rotation[1]",
    "rotation_z": "Computed rotation[2]",
    "rotation_norm": "Computed rotation norm",
}


def normalize(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def parse_args():
    parser = argparse.ArgumentParser(
        prog="PlotImageScoresByMetadata",
        description=(
            "Plot individual image VPI scores against image metadata such as "
            "altitude, computed rotation, and quality score. Images with "
            "zero scores are ignored by default."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output plot image path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV path for the source image rows. Use an empty string to skip.",
    )
    parser.add_argument(
        "--source",
        default="mapillary",
        help="Image source filter. Use an empty string to include all sources.",
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
        "--include-zero-score",
        action="store_true",
        help="Include images with a stored score of zero. By default only positive scores are plotted.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of images to plot after filtering.",
    )
    parser.add_argument(
        "--metrics",
        default="altitude,quality_score,computed_compass_angle,rotation_norm",
        help=(
            "Comma-separated metadata metrics to plot. Available: "
            f"{', '.join(METRIC_LABELS)}."
        ),
    )
    parser.add_argument(
        "--no-trend",
        action="store_true",
        help="Do not draw simple least-squares trend lines.",
    )
    parser.add_argument(
        "--outlier-filter",
        choices=["iqr", "quantile", "none"],
        default="iqr",
        help=(
            "How to remove obvious per-metric outliers before plotting. "
            "Use none to plot every value."
        ),
    )
    parser.add_argument(
        "--outlier-iqr-multiplier",
        type=float,
        default=3.0,
        help="IQR fence multiplier used when --outlier-filter iqr is selected.",
    )
    parser.add_argument(
        "--outlier-quantile",
        type=float,
        default=0.01,
        help=(
            "Tail fraction removed by --outlier-filter quantile, and used as "
            "the fallback when an IQR cannot be estimated."
        ),
    )
    return parser.parse_args()


def parse_metrics(value):
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(metrics) - set(METRIC_LABELS))
    if unknown:
        raise SystemExit(f"Unknown metric(s): {', '.join(unknown)}")
    if not metrics:
        raise SystemExit("At least one metric must be requested.")
    return metrics


def build_image_query(db, args):
    from src.database import Image, Region

    query = db.session.query(
        Image.id.label("image_id"),
        Image.region_id,
        Region.city,
        Region.country,
        Image.source,
        Image.id_from_source.label("source_image_id"),
        Image.source_captured_at.label("captured_at"),
        Image.score,
        Image.altitude,
        Image.computed_altitude,
        Image.quality_score,
        Image.compass_angle,
        Image.computed_compass_angle,
        Image.computed_rotation,
    ).outerjoin(Region, Image.region_id == Region.id)

    query = query.filter(Image.score.isnot(None))
    if not args.include_zero_score:
        query = query.filter(Image.score > 0.0)
    if args.source:
        query = query.filter(func.lower(func.coalesce(Image.source, "")) == normalize(args.source))
    if args.country:
        query = query.filter(func.lower(func.coalesce(Region.country, "")) == normalize(args.country))
    if args.city:
        query = query.filter(func.lower(func.coalesce(Region.city, "")) == normalize(args.city))
    if args.region_ids:
        query = query.filter(Image.region_id.in_(args.region_ids))

    query = query.order_by(Image.id)
    if args.max_images is not None:
        if args.max_images < 1:
            raise SystemExit("--max-images must be at least 1.")
        query = query.limit(args.max_images)
    return query


def float_or_none(value):
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def rotation_component(rotation, index):
    if not isinstance(rotation, (list, tuple)) or len(rotation) <= index:
        return None
    return float_or_none(rotation[index])


def rotation_norm(rotation):
    if not isinstance(rotation, (list, tuple)):
        return None
    values = [float_or_none(value) for value in rotation]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values))


def rows_to_frame(rows):
    records = []
    for row in rows:
        record = dict(row._mapping)
        rotation = record.pop("computed_rotation", None)
        record["rotation_x"] = rotation_component(rotation, 0)
        record["rotation_y"] = rotation_component(rotation, 1)
        record["rotation_z"] = rotation_component(rotation, 2)
        record["rotation_norm"] = rotation_norm(rotation)
        record["computed_rotation"] = rotation
        records.append(record)
    return pd.DataFrame.from_records(records)


def export_csv(frame, output_path):
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export = frame.copy()
    export["captured_at"] = export["captured_at"].apply(
        lambda value: value.isoformat() if hasattr(value, "isoformat") else value
    )
    export["computed_rotation"] = export["computed_rotation"].apply(
        lambda value: "" if value is None else value
    )
    export.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def add_trend_line(ax, plot_data, metric):
    if plot_data[metric].nunique() < 2:
        return
    x = plot_data[metric].to_numpy(dtype=float)
    y = plot_data["score"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#111827", linewidth=1.5)


def quantile_bounds(values, tail_fraction):
    lower = values.quantile(tail_fraction)
    upper = values.quantile(1.0 - tail_fraction)
    if pd.isna(lower) or pd.isna(upper) or lower > upper:
        return None
    return lower, upper


def iqr_bounds(values, multiplier, fallback_tail_fraction):
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr <= 0.0:
        return quantile_bounds(values, fallback_tail_fraction)
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def filter_metric_outliers(plot_data, metric, method, iqr_multiplier, quantile_tail):
    if metric == "quality_score":
        filtered = plot_data[plot_data[metric].between(0.0, 1.0)].copy()
        removed_count = len(plot_data) - len(filtered)
        if filtered.empty:
            return plot_data, 0
        return filtered, removed_count

    if method == "none" or len(plot_data) < 4:
        return plot_data, 0

    values = plot_data[metric]
    if method == "iqr":
        bounds = iqr_bounds(values, iqr_multiplier, quantile_tail)
    else:
        bounds = quantile_bounds(values, quantile_tail)
    if bounds is None:
        return plot_data, 0

    lower, upper = bounds
    filtered = plot_data[plot_data[metric].between(lower, upper)].copy()
    removed_count = len(plot_data) - len(filtered)
    if filtered.empty:
        return plot_data, 0
    return filtered, removed_count


def plot_scores(frame, metrics, output_path, draw_trend, outlier_method, iqr_multiplier, quantile_tail):
    available_metrics = [
        metric for metric in metrics if metric in frame and frame[metric].notna().any()
    ]
    if not available_metrics:
        raise SystemExit("No requested metadata metrics had values for the filtered images.")

    column_count = 2 if len(available_metrics) > 1 else 1
    row_count = math.ceil(len(available_metrics) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7.0 * column_count, 4.8 * row_count),
        squeeze=False,
    )

    for ax, metric in zip(axes.flat, available_metrics):
        raw_plot_data = frame[[metric, "score"]].dropna()
        plot_data, outlier_count = filter_metric_outliers(
            raw_plot_data,
            metric,
            outlier_method,
            iqr_multiplier,
            quantile_tail,
        )
        ax.scatter(
            plot_data[metric],
            plot_data["score"],
            s=13,
            alpha=0.35,
            color="#2563eb",
            edgecolors="none",
        )
        if draw_trend and len(plot_data) >= 2:
            add_trend_line(ax, plot_data, metric)

        correlation = plot_data[metric].corr(plot_data["score"])
        correlation_text = "n/a" if pd.isna(correlation) else f"{correlation:.3f}"
        ax.set_title(f"{METRIC_LABELS[metric]} vs image score")
        ax.set_xlabel(METRIC_LABELS[metric])
        ax.set_ylabel("Image VPI score")
        ax.grid(alpha=0.25)
        ax.text(
            0.02,
            0.98,
            f"n={len(plot_data):,}\nr={correlation_text}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85},
        )
        if outlier_count:
            ax.text(
                0.98,
                0.98,
                f"{outlier_count:,} value(s) hidden",
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=8,
                color="#7f1d1d",
                bbox={"facecolor": "white", "edgecolor": "#fecaca", "alpha": 0.85},
            )

    for ax in axes.flat[len(available_metrics):]:
        ax.axis("off")

    fig.suptitle("Individual image scores by metadata", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return available_metrics


def main():
    from src.database import DatabaseManager

    args = parse_args()
    if args.outlier_iqr_multiplier <= 0.0:
        raise SystemExit("--outlier-iqr-multiplier must be greater than zero.")
    if not 0.0 <= args.outlier_quantile < 0.5:
        raise SystemExit("--outlier-quantile must be at least 0 and less than 0.5.")
    metrics = parse_metrics(args.metrics)
    output_csv = Path(args.output_csv) if str(args.output_csv).strip() else None

    db = DatabaseManager()
    rows = build_image_query(db, args).all()
    if not rows:
        raise SystemExit("No scored images matched the requested filters.")

    frame = rows_to_frame(rows)
    export_csv(frame, output_csv)
    plotted_metrics = plot_scores(
        frame,
        metrics,
        args.output,
        not args.no_trend,
        args.outlier_filter,
        args.outlier_iqr_multiplier,
        args.outlier_quantile,
    )

    print(f"Loaded {len(frame):,} scored image(s).")
    print(f"Plotted metric(s): {', '.join(plotted_metrics)}")
    print(f"Saved plot to {args.output}")
    if output_csv is not None:
        print(f"Saved plotted rows to {output_csv}")


if __name__ == "__main__":
    main()
