import argparse
import math
import sys
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DatabaseConfig  # noqa: E402

REGULATION_YEARS = {
    "krakow": 2020,
    "gdansk": 2018,
    "wroclaw": 2020,
    "poznan": 2023,
    "lodz": 2016
}

COMPARISON_SPLIT_YEARS = {
    "warsaw": 2020,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot yearly billboard detection rates for each city listed in "
            "an experiment city scan CSV."
        )
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres SQLAlchemy URL. Defaults to DATABASE_URL from auth/.env.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/experiment_cities.csv"),
        help="CSV containing city/year scan windows.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/billboard_detection_rate_by_city_year.csv"),
        help="Output CSV path for the aggregated yearly city rates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maps/billboard_detection_rate_by_city_year.png"),
        help="Output PNG path for the faceted city plot.",
    )
    parser.add_argument(
        "--label",
        default="billboard",
        help="Detection label to count. Defaults to billboard.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of subplot columns in the output figure.",
    )
    return parser.parse_args()


def get_database_url(database_url):
    if database_url:
        return database_url
    return DatabaseConfig.get_postgres_url()


def normalize_text(value):
    if value is None:
        return ""
    text_value = unicodedata.normalize("NFKD", str(value).strip())
    return "".join(char for char in text_value if not unicodedata.combining(char)).casefold()


def get_city_split_year(city, city_ascii):
    city_key = normalize_text(city)
    city_ascii_key = normalize_text(city_ascii)

    regulation_year = (
        REGULATION_YEARS.get(city_key)
        or REGULATION_YEARS.get(city_ascii_key)
    )
    if regulation_year is not None:
        return regulation_year, "regulation"

    comparison_year = (
        COMPARISON_SPLIT_YEARS.get(city_key)
        or COMPARISON_SPLIT_YEARS.get(city_ascii_key)
    )
    if comparison_year is not None:
        return comparison_year, "midpoint"

    return None, None


def load_experiment_city_windows(path: Path):
    if not path.exists():
        raise SystemExit(f"Input CSV not found: {path}")

    cities = pd.read_csv(path)
    required_columns = {
        "city",
        "city_ascii",
        "country",
        "year",
        "start_captured_at",
        "end_captured_at",
    }
    missing = required_columns.difference(cities.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise SystemExit(f"Input CSV is missing required columns: {missing_list}")

    cities = cities.copy()
    cities["city"] = cities["city"].fillna("").astype(str).str.strip()
    cities["city_ascii"] = cities["city_ascii"].fillna("").astype(str).str.strip()
    cities["country"] = cities["country"].fillna("").astype(str).str.strip()
    cities["year"] = pd.to_numeric(cities["year"], errors="coerce")
    cities["start_captured_at"] = pd.to_datetime(
        cities["start_captured_at"],
        errors="coerce",
        utc=True,
    )
    cities["end_captured_at"] = pd.to_datetime(
        cities["end_captured_at"],
        errors="coerce",
        utc=True,
    )

    cities = cities.dropna(subset=["year", "start_captured_at", "end_captured_at"]).copy()
    if cities.empty:
        raise SystemExit("Input CSV did not contain any valid city/year scan windows.")

    cities["year"] = cities["year"].astype(int)
    cities["city_key"] = cities["city"].map(normalize_text)
    cities["city_ascii_key"] = cities["city_ascii"].map(normalize_text)
    cities["country_key"] = cities["country"].map(normalize_text)
    cities["population"] = pd.to_numeric(cities.get("population"), errors="coerce")
    return cities


def load_region_detection_counts(database_url, label):
    engine = create_engine(database_url, poolclass=NullPool)
    query = text(
        """
        SELECT
            regions.id AS region_id,
            regions.city,
            regions.country,
            regions.start_captured_at,
            regions.end_captured_at,
            COUNT(DISTINCT images.id) AS image_count,
            COUNT(detections.id) FILTER (
                WHERE lower(coalesce(detections.label, '')) = lower(:label)
            ) AS target_detection_count,
            COUNT(detections.id) AS total_detection_count
        FROM regions
        LEFT JOIN images ON images.region_id = regions.id
        LEFT JOIN detections ON detections.image_id = images.id
        WHERE
            regions.city IS NOT NULL
            AND regions.country IS NOT NULL
            AND regions.start_captured_at IS NOT NULL
            AND regions.end_captured_at IS NOT NULL
        GROUP BY
            regions.id,
            regions.city,
            regions.country,
            regions.start_captured_at,
            regions.end_captured_at
        """
    )
    with engine.connect() as connection:
        rows = pd.read_sql_query(query, connection, params={"label": label})

    if rows.empty:
        raise SystemExit("No regions with capture windows were found in the database.")

    rows["start_captured_at"] = pd.to_datetime(rows["start_captured_at"], errors="coerce", utc=True)
    rows["end_captured_at"] = pd.to_datetime(rows["end_captured_at"], errors="coerce", utc=True)
    rows["city"] = rows["city"].fillna("").astype(str).str.strip()
    rows["country"] = rows["country"].fillna("").astype(str).str.strip()
    rows["city_key"] = rows["city"].map(normalize_text)
    rows["country_key"] = rows["country"].map(normalize_text)
    rows["image_count"] = pd.to_numeric(rows["image_count"], errors="coerce").fillna(0).astype(int)
    rows["target_detection_count"] = (
        pd.to_numeric(rows["target_detection_count"], errors="coerce").fillna(0).astype(int)
    )
    rows["total_detection_count"] = (
        pd.to_numeric(rows["total_detection_count"], errors="coerce").fillna(0).astype(int)
    )
    return rows


def match_city_windows(city_windows, region_counts):
    region_lookup = {}
    for row in region_counts.itertuples(index=False):
        key = (
            row.city_key,
            row.country_key,
            row.start_captured_at,
            row.end_captured_at,
        )
        region_lookup.setdefault(key, []).append(row)

    summary_rows = []
    for city_row in city_windows.itertuples(index=False):
        candidate_keys = [
            (
                city_row.city_key,
                city_row.country_key,
                city_row.start_captured_at,
                city_row.end_captured_at,
            )
        ]
        if city_row.city_ascii_key and city_row.city_ascii_key != city_row.city_key:
            candidate_keys.append(
                (
                    city_row.city_ascii_key,
                    city_row.country_key,
                    city_row.start_captured_at,
                    city_row.end_captured_at,
                )
            )

        matched = []
        seen_region_ids = set()
        for key in candidate_keys:
            for region in region_lookup.get(key, []):
                if region.region_id in seen_region_ids:
                    continue
                seen_region_ids.add(region.region_id)
                matched.append(region)

        image_count = sum(region.image_count for region in matched)
        target_detection_count = sum(region.target_detection_count for region in matched)
        total_detection_count = sum(region.total_detection_count for region in matched)

        summary_rows.append(
            {
                "city": city_row.city,
                "city_ascii": city_row.city_ascii,
                "country": city_row.country,
                "year": int(city_row.year),
                "population": city_row.population,
                "start_captured_at": city_row.start_captured_at,
                "end_captured_at": city_row.end_captured_at,
                "matching_region_count": len(matched),
                "image_count": int(image_count),
                "billboard_detection_count": int(target_detection_count),
                "total_detection_count": int(total_detection_count),
                "billboards_per_image": (
                    float(target_detection_count) / float(image_count) if image_count > 0 else 0.0
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise SystemExit("No city/year rows could be matched to database regions.")
    return summary


def plot_city_rates(summary, output_path: Path, label: str, cols: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary[summary["matching_region_count"] > 0].copy()
    city_order = (
        summary[["city", "population"]]
        .drop_duplicates("city")
        .sort_values(["population", "city"], ascending=[False, True], na_position="last")
    )["city"].tolist()
    if not city_order:
        raise SystemExit("No city rows available to plot.")

    ncols = max(1, int(cols))
    nrows = math.ceil(len(city_order) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.4 * nrows),
        dpi=180,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    years = sorted(summary["year"].dropna().astype(int).unique().tolist())
    y_max = summary["billboards_per_image"].max()
    y_limit = max(0.05, float(y_max) * 1.15 if pd.notna(y_max) else 0.05)

    for index, city in enumerate(city_order):
        ax = axes[index]
        city_data = summary[summary["city"] == city].copy()
        city_data = city_data.sort_values("year")

        ax.plot(
            city_data["year"],
            city_data["billboards_per_image"],
            color="#c2410c",
            linewidth=2.0,
            marker="o",
            markersize=4.5,
        )
        ax.set_title(city, fontsize=11, fontweight="bold")
        ax.set_xticks(years)
        ax.tick_params(axis="x", labelbottom=True, length=3, labelsize=8)
        ax.set_ylim(0, y_limit)
        ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)

        city_ascii = str(city_data["city_ascii"].dropna().iloc[0]).strip() if "city_ascii" in city_data.columns and not city_data["city_ascii"].dropna().empty else ""
        regulation_year, split_label = get_city_split_year(city, city_ascii)
        if regulation_year is not None:
            ax.axvline(
                regulation_year,
                color="#1d4ed8",
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
            )
            ax.text(
                regulation_year + 0.05,
                y_limit * 0.96,
                split_label,
                rotation=90,
                ha="left",
                va="top",
                fontsize=7,
                color="#1d4ed8",
            )

        total_images = int(city_data["image_count"].sum())
        total_detections = int(city_data["billboard_detection_count"].sum())
        ax.text(
            0.98,
            0.96,
            f"images={total_images} | detections={total_detections}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#4b5563",
        )

    for ax in axes[len(city_order):]:
        ax.axis("off")

    for ax in axes:
        if ax.has_data():
            ax.set_ylabel("")

    fig.supylabel("Billboard detection density", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_regulation_summary(summary):
    rows = []
    plotted = summary[summary["matching_region_count"] > 0].copy()
    if plotted.empty:
        return pd.DataFrame(
            columns=[
                "city",
                "city_ascii",
                "regulation_year",
                "period",
                "year_count",
                "image_count",
                "billboard_detection_count",
                "billboards_per_image",
            ]
        )

    city_groups = plotted.groupby("city", sort=False)
    for city, city_data in city_groups:
        city_ascii = (
            str(city_data["city_ascii"].dropna().iloc[0]).strip()
            if "city_ascii" in city_data.columns and not city_data["city_ascii"].dropna().empty
            else ""
        )
        regulation_year, _ = get_city_split_year(city, city_ascii)
        if regulation_year is None:
            continue

        periods = {
            "before": city_data[city_data["year"] < regulation_year].copy(),
            "after_or_equal": city_data[city_data["year"] >= regulation_year].copy(),
        }
        for period_name, period_data in periods.items():
            image_count = int(period_data["image_count"].sum())
            billboard_detection_count = int(period_data["billboard_detection_count"].sum())
            rows.append(
                {
                    "city": city,
                    "city_ascii": city_ascii,
                    "regulation_year": int(regulation_year),
                    "period": period_name,
                    "year_count": int(len(period_data)),
                    "image_count": image_count,
                    "billboard_detection_count": billboard_detection_count,
                    "billboards_per_image": (
                        float(billboard_detection_count) / float(image_count)
                        if image_count > 0 else 0.0
                    ),
                }
            )

    return pd.DataFrame(rows)


def plot_city_rates(summary, output_path: Path, label: str, cols: int, regulation_summary=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary[summary["matching_region_count"] > 0].copy()
    city_order = (
        summary[["city", "population"]]
        .drop_duplicates("city")
        .sort_values(["population", "city"], ascending=[False, True], na_position="last")
    )["city"].tolist()
    if not city_order:
        raise SystemExit("No city rows available to plot.")

    regulation_lookup = {}
    if regulation_summary is not None and not regulation_summary.empty:
        for row in regulation_summary.itertuples(index=False):
            regulation_lookup[(row.city, row.period)] = row

    ncols = max(1, int(cols))
    nrows = math.ceil(len(city_order) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.4 * nrows),
        dpi=180,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    years = sorted(summary["year"].dropna().astype(int).unique().tolist())
    y_max = summary["billboards_per_image"].max()
    y_limit = max(0.05, float(y_max) * 1.15 if pd.notna(y_max) else 0.05)

    for index, city in enumerate(city_order):
        ax = axes[index]
        city_data = summary[summary["city"] == city].copy()
        city_data = city_data.sort_values("year")

        ax.plot(
            city_data["year"],
            city_data["billboards_per_image"],
            color="#c2410c",
            linewidth=2.0,
            marker="o",
            markersize=4.5,
        )
        ax.set_title(city, fontsize=11, fontweight="bold")
        ax.set_xticks(years)
        ax.tick_params(axis="x", labelbottom=True, length=3, labelsize=8)
        ax.set_ylim(0, y_limit)
        ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)

        city_ascii = str(city_data["city_ascii"].dropna().iloc[0]).strip() if "city_ascii" in city_data.columns and not city_data["city_ascii"].dropna().empty else ""
        regulation_year, split_label = get_city_split_year(city, city_ascii)
        if regulation_year is not None:
            ax.axvline(
                regulation_year,
                color="#1d4ed8",
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
            )
            ax.text(
                regulation_year + 0.05,
                y_limit * 0.96,
                split_label,
                rotation=90,
                ha="left",
                va="top",
                fontsize=7,
                color="#1d4ed8",
            )

            before_row = regulation_lookup.get((city, "before"))
            after_row = regulation_lookup.get((city, "after_or_equal"))
            if before_row is not None:
                ax.text(
                    0.02,
                    0.90,
                    f"Before: {before_row.billboards_per_image:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="#7c2d12",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
                )
            if after_row is not None:
                ax.text(
                    0.98,
                    0.90,
                    f"After: {after_row.billboards_per_image:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    color="#7c2d12",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
                )

        total_images = int(city_data["image_count"].sum())
        total_detections = int(city_data["billboard_detection_count"].sum())
        ax.text(
            0.98,
            0.96,
            f"images={total_images} | detections={total_detections}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#4b5563",
        )

    for ax in axes[len(city_order):]:
        ax.axis("off")

    for ax in axes:
        if ax.has_data():
            ax.set_ylabel("")

    fig.supylabel("Billboard detection density", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    database_url = get_database_url(args.database_url)
    city_windows = load_experiment_city_windows(args.input_csv)
    region_counts = load_region_detection_counts(database_url, args.label)
    summary = match_city_windows(city_windows, region_counts)
    summary = summary.sort_values(["population", "city", "year"], ascending=[False, True, True])
    regulation_summary = build_regulation_summary(summary)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    plot_city_rates(summary, args.output, args.label, args.cols, regulation_summary=regulation_summary)

    matched_rows = int((summary["matching_region_count"] > 0).sum())
    missing_rows = int((summary["matching_region_count"] == 0).sum())
    total_images = int(summary["image_count"].sum())
    total_billboards = int(summary["billboard_detection_count"].sum())

    print(f"Saved yearly city billboard rates to {args.output_csv}")
    print(f"Saved faceted billboard rate plot to {args.output}")
    print(f"Matched city/year rows: {matched_rows}")
    print(f"Unmatched city/year rows: {missing_rows}")
    print(f"Total images across matched rows: {total_images}")
    print(f"Total {args.label} detections across matched rows: {total_billboards}")
    if not regulation_summary.empty:
        print("Regulation summary:")
        for row in regulation_summary.itertuples(index=False):
            city_label = row.city_ascii or row.city
            print(
                f"{city_label} | regulation_year={row.regulation_year} | period={row.period} "
                f"| years={row.year_count} | images={row.image_count} "
                f"| {args.label}_detections={row.billboard_detection_count} "
                f"| rate={row.billboards_per_image:.6f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
