import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DatabaseConfig  # noqa: E402


DEFAULT_COLUMNS = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot image-level detection rates by capture time of day for each "
            "pollutant label."
        )
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres SQLAlchemy URL. Defaults to DATABASE_URL from auth/.env.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Optional region city filter, e.g. Edinburgh.",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Optional region country filter.",
    )
    parser.add_argument(
        "--timezone",
        default="Europe/London",
        help="Timezone used to convert capture timestamps before extracting hour.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional detection labels to include. Defaults to every label found.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=DEFAULT_COLUMNS,
        help="Number of subplot columns in the combined plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maps/pollutant_detection_capture_time_of_day.png"),
        help="Output combined plot image path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/pollutant_detection_capture_time_of_day.csv"),
        help="Output hourly label-rate CSV path.",
    )
    parser.add_argument(
        "--per-label-dir",
        type=Path,
        default=None,
        help="Optional folder for one plot per pollutant label.",
    )
    return parser.parse_args()


def get_database_url(database_url):
    if database_url:
        return database_url
    return DatabaseConfig.get_postgres_url()


def normalize_label(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def label_hour(hour):
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 21:
        return "evening/twilight"
    return "night"


def slugify(value):
    return (
        "".join(char if char.isalnum() else "_" for char in value.strip().lower())
        .strip("_")
        or "blank"
    )


def add_time_of_day_bands(ax, show_labels=False):
    bands = [
        (-0.5, 5.5, "#111827", "Night"),
        (5.5, 8.5, "#f59e0b", "Morning/twilight"),
        (8.5, 17.5, "#fef3c7", "Day"),
        (17.5, 20.5, "#f59e0b", "Evening/twilight"),
        (20.5, 23.5, "#111827", "Night"),
    ]
    for start, end, color, label in bands:
        ax.axvspan(start, end, color=color, alpha=0.07, linewidth=0)
        if show_labels:
            ax.text(
                (start + end) / 2,
                0.98,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.5,
                color="#374151",
            )


def build_region_filters(city=None, country=None):
    filters = ["images.source_captured_at IS NOT NULL"]
    params = {}
    if city:
        filters.append("lower(coalesce(regions.city, '')) = lower(:city)")
        params["city"] = city
    if country:
        filters.append("lower(coalesce(regions.country, '')) = lower(:country)")
        params["country"] = country
    return filters, params


def add_label_filter(filters, params, labels):
    if labels:
        cleaned_labels = [normalize_label(label) for label in labels if label.strip()]
        placeholders = []
        for index, label in enumerate(cleaned_labels):
            key = f"label_{index}"
            params[key] = label
            placeholders.append(f":{key}")
        if placeholders:
            filters.append(f"detections.label IN ({', '.join(placeholders)})")
    return filters, params


def load_hourly_label_rates(database_url, timezone, city=None, country=None, labels=None):
    filters, params = build_region_filters(city=city, country=country)
    params["timezone"] = timezone

    where_clause = " AND ".join(filters)
    total_query = text(
        f"""
        SELECT
            date_part('hour', images.source_captured_at AT TIME ZONE :timezone)::int
                AS capture_hour,
            count(images.id)::int AS total_images
        FROM images
        JOIN regions ON regions.id = images.region_id
        WHERE {where_clause}
        GROUP BY capture_hour
        """
    )

    detection_filters = list(filters)
    detection_params = dict(params)
    detection_filters.append("detections.label IS NOT NULL")
    detection_filters, detection_params = add_label_filter(
        detection_filters,
        detection_params,
        labels,
    )
    detection_where_clause = " AND ".join(detection_filters)
    detection_query = text(
        f"""
        SELECT
            detections.label AS detection_label,
            date_part('hour', images.source_captured_at AT TIME ZONE :timezone)::int
                AS capture_hour,
            count(DISTINCT images.id)::int AS images_with_label,
            count(detections.id)::int AS detection_count
        FROM images
        JOIN regions ON regions.id = images.region_id
        JOIN detections ON detections.image_id = images.id
        WHERE {detection_where_clause}
        GROUP BY detections.label, capture_hour
        """
    )

    engine = create_engine(database_url, poolclass=NullPool)
    with engine.connect() as connection:
        total_by_hour = pd.read_sql_query(total_query, connection, params=params)
        label_counts = pd.read_sql_query(
            detection_query,
            connection,
            params=detection_params,
        )

    if total_by_hour.empty:
        raise SystemExit("No images found for the selected filters.")
    if label_counts.empty:
        raise SystemExit("No detections found for the selected filters.")

    total_by_hour["capture_hour"] = total_by_hour["capture_hour"].astype(int)
    total_by_hour["total_images"] = total_by_hour["total_images"].astype(int)
    total_by_hour = (
        pd.DataFrame({"capture_hour": range(24)})
        .merge(total_by_hour, on="capture_hour", how="left")
        .fillna({"total_images": 0})
    )
    total_by_hour["total_images"] = total_by_hour["total_images"].astype(int)

    label_counts["detection_label"] = label_counts["detection_label"].map(
        normalize_label
    )
    label_counts = label_counts[label_counts["detection_label"] != ""].copy()

    if labels:
        label_order = [normalize_label(label) for label in labels if label.strip()]
    else:
        label_order = sorted(label_counts["detection_label"].dropna().unique())
    if not label_order:
        raise SystemExit("No detection labels available to plot.")

    index = pd.MultiIndex.from_product(
        [label_order, range(24)],
        names=["detection_label", "capture_hour"],
    ).to_frame(index=False)
    summary = index.merge(total_by_hour, on="capture_hour", how="left")
    summary = summary.merge(
        label_counts,
        on=["detection_label", "capture_hour"],
        how="left",
    )
    for column in ["total_images", "images_with_label", "detection_count"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0)
        summary[column] = summary[column].astype(int)
    summary["image_detection_rate"] = (
        summary["images_with_label"] / summary["total_images"].replace(0, pd.NA)
    ).fillna(0)
    summary["detections_per_image"] = (
        summary["detection_count"] / summary["total_images"].replace(0, pd.NA)
    ).fillna(0)
    summary["time_of_day"] = summary["capture_hour"].map(label_hour)
    return summary


def plot_label_panel(ax, subset, label, show_band_labels=False):
    x = subset["capture_hour"]
    add_time_of_day_bands(ax, show_labels=show_band_labels)
    ax.bar(
        x,
        subset["images_with_label"],
        width=0.72,
        color="#93c5fd",
        alpha=0.42,
        label="Images with label",
    )
    rate_ax = ax.twinx()
    rate_ax.plot(
        x,
        subset["image_detection_rate"],
        color="#b2182b",
        linewidth=2.0,
        marker="o",
        markersize=2.8,
        label="Image detection rate",
    )

    ax.set_title(label.replace("_", " ").title())
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.set_ylabel("Images with label")
    rate_ax.set_ylabel("Rate")
    rate_ax.set_ylim(0, max(0.01, min(1.0, subset["image_detection_rate"].max() * 1.15)))

    summary = (
        f"images={int(subset['total_images'].sum())} | "
        f"label images={int(subset['images_with_label'].sum())} | "
        f"mean rate={subset['image_detection_rate'].mean():.3f}"
    )
    ax.text(
        0.99,
        0.03,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#374151",
    )
    return rate_ax


def plot_combined(summary, output_path, title_suffix, timezone, columns):
    labels = sorted(summary["detection_label"].unique())
    if not labels:
        raise SystemExit("No labels available to plot.")

    columns = max(1, columns)
    rows = math.ceil(len(labels) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.6 * columns, 3.8 * rows),
        dpi=180,
        sharex=True,
    )
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for index, label in enumerate(labels):
        row = index // columns
        column = index % columns
        ax = axes[row, column]
        subset = summary[summary["detection_label"] == label].copy()
        plot_label_panel(ax, subset, label, show_band_labels=(index < columns))

    for index in range(len(labels), rows * columns):
        row = index // columns
        column = index % columns
        axes[row, column].set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel(f"Local capture hour ({timezone})")

    fig.suptitle(
        f"Image Detection Rates by Pollutant and Capture Time{title_suffix}",
        y=0.995,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_label(summary, output_dir, timezone):
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in sorted(summary["detection_label"].unique()):
        subset = summary[summary["detection_label"] == label].copy()
        fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=180)
        plot_label_panel(ax, subset, label, show_band_labels=True)
        ax.set_xlabel(f"Local capture hour ({timezone})")
        fig.tight_layout()
        fig.savefig(output_dir / f"{slugify(label)}.png")
        plt.close(fig)


def main():
    args = parse_args()
    database_url = get_database_url(args.database_url)
    summary = load_hourly_label_rates(
        database_url,
        args.timezone,
        city=args.city,
        country=args.country,
        labels=args.labels,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    filters = []
    if args.city:
        filters.append(args.city)
    if args.country:
        filters.append(args.country)
    title_suffix = f" ({', '.join(filters)})" if filters else ""

    plot_combined(summary, args.output, title_suffix, args.timezone, args.columns)
    if args.per_label_dir is not None:
        plot_per_label(summary, args.per_label_dir, args.timezone)

    unique_images = int(summary.drop_duplicates("capture_hour")["total_images"].sum())
    labels = sorted(summary["detection_label"].unique())
    print(f"Saved pollutant capture-time plot to {args.output}")
    print(f"Saved hourly pollutant rate CSV to {args.output_csv}")
    if args.per_label_dir is not None:
        print(f"Saved per-label plots to {args.per_label_dir}")
    print(f"Images included: {unique_images}")
    print("Detection labels plotted: " + ", ".join(labels))


if __name__ == "__main__":
    raise SystemExit(main())
