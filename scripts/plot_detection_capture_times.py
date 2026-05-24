import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DatabaseConfig  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot image capture time-of-day split by whether each image has "
            "at least one detection."
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
        "--output",
        type=Path,
        default=Path("maps/detection_capture_time_of_day.png"),
        help="Output plot image path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/detection_capture_time_of_day.csv"),
        help="Output aggregated CSV path.",
    )
    return parser.parse_args()


def get_database_url(database_url):
    if database_url:
        return database_url
    return DatabaseConfig.get_postgres_url()


def load_capture_times(database_url, city=None, country=None):
    filters = ["images.source_captured_at IS NOT NULL"]
    params = {}
    if city:
        filters.append("lower(coalesce(regions.city, '')) = lower(:city)")
        params["city"] = city
    if country:
        filters.append("lower(coalesce(regions.country, '')) = lower(:country)")
        params["country"] = country

    where_clause = " AND ".join(filters)
    query = text(
        f"""
        SELECT
            images.id AS image_id,
            images.source_captured_at,
            CASE WHEN count(detections.id) > 0 THEN true ELSE false END AS has_detection
        FROM images
        JOIN regions ON regions.id = images.region_id
        LEFT JOIN detections ON detections.image_id = images.id
        WHERE {where_clause}
        GROUP BY images.id, images.source_captured_at
        """
    )
    engine = create_engine(database_url, poolclass=NullPool)
    with engine.connect() as connection:
        rows = pd.read_sql_query(query, connection, params=params)

    if rows.empty:
        raise SystemExit("No images found for the selected filters.")
    rows["source_captured_at"] = pd.to_datetime(
        rows["source_captured_at"],
        errors="coerce",
        utc=True,
    )
    rows = rows.dropna(subset=["source_captured_at"])
    if rows.empty:
        raise SystemExit("No images had parseable source_captured_at values.")
    return rows


def aggregate_by_hour(images, timezone):
    images = images.copy()
    images["local_captured_at"] = images["source_captured_at"].dt.tz_convert(timezone)
    images["capture_hour"] = images["local_captured_at"].dt.hour
    grouped = (
        images.groupby(["capture_hour", "has_detection"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={True: "images_with_detections", False: "images_without_detections"})
    )
    summary = pd.DataFrame({"capture_hour": range(24)}).merge(
        grouped.reset_index(),
        on="capture_hour",
        how="left",
    ).fillna(0)
    for column in ["images_with_detections", "images_without_detections"]:
        if column not in summary.columns:
            summary[column] = 0
    summary["total_images"] = (
        summary["images_with_detections"] + summary["images_without_detections"]
    )
    summary["detection_image_rate"] = (
        summary["images_with_detections"] / summary["total_images"].replace(0, pd.NA)
    ).fillna(0)
    summary["time_of_day"] = summary["capture_hour"].map(label_hour)
    return summary


def label_hour(hour):
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 21:
        return "evening/twilight"
    return "night"


def add_time_of_day_bands(ax):
    bands = [
        (-0.5, 5.5, "#111827", "Night"),
        (5.5, 8.5, "#f59e0b", "Morning/twilight"),
        (8.5, 17.5, "#fef3c7", "Day"),
        (17.5, 20.5, "#f59e0b", "Evening/twilight"),
        (20.5, 23.5, "#111827", "Night"),
    ]
    for start, end, color, label in bands:
        ax.axvspan(start, end, color=color, alpha=0.07, linewidth=0)
        ax.text(
            (start + end) / 2,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#374151",
        )


def plot_capture_times(summary, output_path, title_suffix, timezone):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = summary["capture_hour"]

    fig, (count_ax, rate_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    add_time_of_day_bands(count_ax)
    width = 0.42
    count_ax.bar(
        x - width / 2,
        summary["images_without_detections"],
        width=width,
        color="#2166ac",
        alpha=0.78,
        label="Images without detections",
    )
    count_ax.bar(
        x + width / 2,
        summary["images_with_detections"],
        width=width,
        color="#b2182b",
        alpha=0.78,
        label="Images with detections",
    )
    count_ax.set_title(
        f"Image Capture Time of Day by Detection Status{title_suffix}"
    )
    count_ax.set_ylabel("Image count")
    count_ax.legend(frameon=False, loc="upper left")
    count_ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)

    add_time_of_day_bands(rate_ax)
    rate_ax.plot(
        x,
        summary["detection_image_rate"],
        color="#111827",
        linewidth=2.0,
    )
    rate_ax.set_ylabel("Detection image rate")
    rate_ax.set_xlabel(f"Local capture hour ({timezone})")
    rate_ax.set_ylim(0, 1)
    rate_ax.set_xlim(-0.5, 23.5)
    rate_ax.set_xticks(range(24))
    rate_ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    database_url = get_database_url(args.database_url)
    images = load_capture_times(database_url, city=args.city, country=args.country)
    summary = aggregate_by_hour(images, args.timezone)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    filters = []
    if args.city:
        filters.append(args.city)
    if args.country:
        filters.append(args.country)
    title_suffix = f" ({', '.join(filters)})" if filters else ""
    plot_capture_times(summary, args.output, title_suffix, args.timezone)

    print(f"Saved capture time-of-day plot to {args.output}")
    print(f"Saved hourly capture CSV to {args.output_csv}")
    print(f"Images with detections: {int(images['has_detection'].sum())}")
    print(f"Images without detections: {int((~images['has_detection']).sum())}")


if __name__ == "__main__":
    raise SystemExit(main())
