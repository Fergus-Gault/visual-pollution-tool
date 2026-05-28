import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager, Image, OSMFeature, Region  # noqa: E402

DEFAULT_INCOME_ORDER = [
    "LI",
    "LMI",
    "UMI",
    "HI",
]

INCOME_LABEL_MAP = {
    "Low income": "LI",
    "Lower middle income": "LMI",
    "Upper middle income": "UMI",
    "High income": "HI",
}


def load_scorer_class():
    score_path = Path(__file__).resolve().parents[1] / "src" / "pipeline" / "score.py"
    spec = importlib.util.spec_from_file_location("pipeline_score", score_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Scorer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot OSM-only scores against imagery VPI scores, ordered by the "
            "imagery VPI score."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/scores_compare_vpi_osm.csv"),
        help=(
            "Existing CSV with region_id, vpi_score, and osm_score columns. "
            "Ignored when --from-db is used."
        ),
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Recompute scores from the configured Postgres database.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot image path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/osm_vs_imagery_scores_ordered.csv"),
        help="Output ordered score CSV path.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=1,
        help="Minimum image count for a region to be included.",
    )
    parser.add_argument(
        "--min-osm-features",
        type=int,
        default=1,
        help="Minimum OSM feature count for a region to be included.",
    )
    parser.add_argument(
        "--exclude-zero-scores",
        action="store_true",
        help="Exclude regions where either VPI or OSM score is zero.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Order regions from highest imagery VPI score to lowest.",
    )
    parser.add_argument(
        "--chart-style",
        choices=["scatter", "candlestick"],
        default="scatter",
        help="Render the ordered region comparison as a scatter plot or candlestick-style chart.",
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=Path("data/CLASS_2025_10_07.csv"),
        help="CSV containing Code and Income group columns for economic classification plots.",
    )
    parser.add_argument(
        "--economic-output",
        type=Path,
        default=None,
        help="Output path for the economic-classification scatter plot.",
    )
    parser.add_argument(
        "--economic-chart-style",
        choices=["scatter", "candlestick"],
        default="scatter",
        help="Render the economic-classification comparison as a scatter plot or candlestick summary chart.",
    )
    return parser.parse_args()


def load_scores_from_csv(path):
    if not path.exists():
        raise SystemExit(f"Input CSV not found: {path}")
    scores = pd.read_csv(path)
    required = {"region_id", "vpi_score", "osm_score"}
    missing = required - set(scores.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise SystemExit(f"Input CSV missing required columns: {missing_text}")
    return scores


def load_scores_from_db():
    db = DatabaseManager()
    Scorer = load_scorer_class()
    scorer = Scorer(db)
    regions = db.get_all_regions()
    region_ids = [region.id for region in regions]

    vpi_scores = scorer.score_regions(region_ids, apply_image_threshold=False)
    osm_scores = scorer.score_regions_with_osm_only(region_ids)
    image_counts = dict(
        db.session.query(Image.region_id, func.count(Image.id))
        .group_by(Image.region_id)
        .all()
    )
    osm_feature_counts = dict(
        db.session.query(OSMFeature.region_id, func.count(OSMFeature.id))
        .group_by(OSMFeature.region_id)
        .all()
    )
    region_lookup = {region.id: region for region in regions}

    rows = []
    for region_id in region_ids:
        region = region_lookup[region_id]
        rows.append(
            {
                "region_id": region_id,
                "city": region.city or "",
                "country": region.country or "",
                "iso3": region.iso3 or "",
                "image_count": int(image_counts.get(region_id, 0)),
                "osm_feature_count": int(osm_feature_counts.get(region_id, 0)),
                "vpi_score": float(vpi_scores.get(region_id, 0.0)),
                "osm_score": float(osm_scores.get(region_id, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def load_classification_table(path):
    if not path.exists():
        raise SystemExit(f"Classification CSV not found: {path}")

    classifications = pd.read_csv(path)
    required = {"Code", "Income group"}
    missing = required - set(classifications.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise SystemExit(
            f"Classification CSV missing required columns: {missing_text}"
        )

    keep_columns = [
        column
        for column in ["Code", "Economy", "Region", "Income group"]
        if column in classifications.columns
    ]
    classifications = classifications[keep_columns].copy()
    classifications["Code"] = classifications["Code"].astype(str).str.strip().str.upper()
    classifications["Income group"] = (
        classifications["Income group"].astype(str).str.strip()
    )
    classifications["Income group"] = classifications["Income group"].replace(INCOME_LABEL_MAP)
    return classifications


def enrich_scores_with_region_metadata(scores):
    scores = scores.copy()
    if "iso3" in scores.columns and "city" in scores.columns and "country" in scores.columns:
        return scores

    db = DatabaseManager()
    region_rows = pd.DataFrame(
        db.session.query(
            Region.id.label("region_id"),
            Region.city.label("city"),
            Region.country.label("country"),
            Region.iso3.label("iso3"),
        ).all(),
        columns=["region_id", "city", "country", "iso3"],
    )
    if region_rows.empty:
        return scores

    merged = scores.merge(region_rows, on="region_id", how="left", suffixes=("", "_db"))
    for column in ["city", "country", "iso3"]:
        db_column = f"{column}_db"
        if db_column in merged.columns:
            if column not in merged.columns:
                merged[column] = merged[db_column]
            else:
                merged[column] = merged[column].fillna("")
                merged[db_column] = merged[db_column].fillna("")
                merged[column] = merged[column].where(merged[column] != "", merged[db_column])
            merged = merged.drop(columns=[db_column])
        elif column not in merged.columns:
            merged[column] = ""
        else:
            merged[column] = merged[column].fillna("")
    return merged


def prepare_scores(scores, min_images, min_osm_features, exclude_zero_scores, descending):
    scores = scores.copy()
    for column in ["image_count", "osm_feature_count", "city", "country", "iso3"]:
        if column not in scores.columns:
            scores[column] = 0 if column in {"image_count", "osm_feature_count"} else ""
    scores["vpi_score"] = pd.to_numeric(scores["vpi_score"], errors="coerce").fillna(0)
    scores["osm_score"] = pd.to_numeric(scores["osm_score"], errors="coerce").fillna(0)
    scores["image_count"] = pd.to_numeric(scores["image_count"], errors="coerce").fillna(0)
    scores["osm_feature_count"] = pd.to_numeric(
        scores["osm_feature_count"],
        errors="coerce",
    ).fillna(0)
    for column in ["city", "country", "iso3"]:
        scores[column] = scores[column].fillna("").astype(str).str.strip()

    scores = scores[
        (scores["image_count"] >= min_images)
        & (scores["osm_feature_count"] >= min_osm_features)
    ].copy()
    if exclude_zero_scores:
        scores = scores[(scores["vpi_score"] > 0) & (scores["osm_score"] > 0)].copy()

    scores["score_difference"] = scores["osm_score"] - scores["vpi_score"]
    scores["absolute_difference"] = scores["score_difference"].abs()
    scores = scores.sort_values(
        ["vpi_score", "osm_score", "region_id"],
        ascending=[not descending, not descending, True],
    ).reset_index(drop=True)
    scores["ordered_region_index"] = scores.index + 1
    return scores


def merge_with_classifications(scores, classifications):
    if scores.empty:
        return scores

    merged = scores.copy()
    merged["iso3"] = merged["iso3"].fillna("").astype(str).str.strip().str.upper()
    merged = merged.merge(
        classifications,
        left_on="iso3",
        right_on="Code",
        how="left",
    )
    merged = merged[
        merged["Income group"].notna() & (merged["Income group"] != "")
    ].copy()
    return merged


def order_income_groups(values):
    ordered = [group for group in DEFAULT_INCOME_ORDER if group in values]
    extras = sorted(value for value in values if value not in DEFAULT_INCOME_ORDER)
    return ordered + extras


def with_style_suffix(path, style):
    return path.with_name(f"{path.stem}_{style}{path.suffix}")


def draw_candles(
    ax,
    x,
    lower,
    upper,
    width,
    up_color,
    down_color,
    alpha=0.72,
    linewidth=1.4,
    cap_width_ratio=0.72,
):
    cap_half_width = (width * cap_width_ratio) / 2.0
    for x_value, low_value, high_value in zip(x, lower, upper):
        color = up_color if high_value >= low_value else down_color
        bottom = min(low_value, high_value)
        height = abs(high_value - low_value)
        ax.vlines(
            x_value,
            bottom,
            bottom + height,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.hlines(
            [bottom, bottom + height],
            x_value - cap_half_width,
            x_value + cap_half_width,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        rect = plt.Rectangle(
            (x_value - width / 2.0, bottom),
            width,
            max(height, 1e-9),
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect)


def plot_scores_scatter(scores, output_path):
    if scores.empty:
        raise SystemExit("No regions left after filtering.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = scores["ordered_region_index"]

    fig, ax = plt.subplots(figsize=(14, 7), dpi=180)
    ax.scatter(
        x,
        scores["vpi_score"],
        s=18,
        color="#2166ac",
        alpha=0.78,
        label="Imagery VPI score",
    )
    ax.scatter(
        x,
        scores["osm_score"],
        s=18,
        color="#b2182b",
        alpha=0.78,
        label="OSM score",
    )
    coefficients = np.polyfit(x, scores["osm_score"], deg=1)
    best_fit = np.poly1d(coefficients)(x)
    ax.plot(
        x,
        best_fit,
        color="#7f1d1d",
        linewidth=2.0,
        alpha=0.9,
        label="OSM best fit",
    )
    ax.vlines(
        x,
        scores["vpi_score"],
        scores["osm_score"],
        color="#6b7280",
        alpha=0.18,
        linewidth=0.7,
    )

    ax.set_title("OSM Score vs Imagery VPI Score by Region")
    ax.set_xlabel("Regions ordered by imagery VPI score")
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, loc="upper left")
    ax.margins(x=0.01)

    summary = (
        f"n={len(scores)} | "
        f"mean OSM-VPI={scores['score_difference'].mean():.4f} | "
        f"mean abs diff={scores['absolute_difference'].mean():.4f}"
    )
    ax.text(
        0.995,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_scores_candlestick(scores, output_path):
    if scores.empty:
        raise SystemExit("No regions left after filtering.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = scores["ordered_region_index"]
    lower = np.minimum(scores["vpi_score"], scores["osm_score"])
    upper = np.maximum(scores["vpi_score"], scores["osm_score"])
    osm_higher = scores["osm_score"] >= scores["vpi_score"]

    fig, ax = plt.subplots(figsize=(14, 7), dpi=180)
    draw_candles(
        ax,
        x,
        scores["vpi_score"],
        scores["osm_score"],
        width=0.82,
        up_color="#b2182b",
        down_color="#2166ac",
        alpha=0.72,
        linewidth=1.6,
        cap_width_ratio=0.85,
    )
    candle_means = (scores["vpi_score"] + scores["osm_score"]) / 2.0
    ax.plot(
        x,
        candle_means,
        color="#4b5563",
        linewidth=1.5,
        alpha=0.9,
        label="Candle mean",
    )
    ax.scatter(
        x[osm_higher],
        upper[osm_higher],
        s=12,
        color="#7f1d1d",
        alpha=0.85,
        edgecolors="none",
        label="OSM > VPI",
    )
    ax.scatter(
        x[~osm_higher],
        upper[~osm_higher],
        s=12,
        color="#0f3f75",
        alpha=0.85,
        edgecolors="none",
        label="VPI > OSM",
    )

    ax.set_title("OSM Score vs Imagery VPI Score by Region")
    ax.set_xlabel("Regions ordered by imagery VPI score")
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, loc="upper left")
    ax.margins(x=0.01)

    summary = (
        f"n={len(scores)} | "
        f"mean OSM-VPI={scores['score_difference'].mean():.4f} | "
        f"mean abs diff={scores['absolute_difference'].mean():.4f}"
    )
    ax.text(
        0.995,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_scores(scores, output_path, chart_style):
    if chart_style == "candlestick":
        plot_scores_candlestick(scores, output_path)
        return
    plot_scores_scatter(scores, output_path)


def plot_scores_by_economic_scatter(scores, output_path):
    if scores.empty:
        raise SystemExit("No classified regions left after filtering.")

    income_order = order_income_groups(scores["Income group"].unique())
    if not income_order:
        raise SystemExit("No economic classifications available for the plotted regions.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_lookup = {label: index for index, label in enumerate(income_order)}
    rng = np.random.default_rng(42)

    imagery_points = scores.copy()
    imagery_points["x"] = imagery_points["Income group"].map(x_lookup).astype(float)
    imagery_points["x"] = imagery_points["x"] + rng.uniform(-0.18, -0.03, size=len(imagery_points))

    osm_points = scores.copy()
    osm_points["x"] = osm_points["Income group"].map(x_lookup).astype(float)
    osm_points["x"] = osm_points["x"] + rng.uniform(0.03, 0.18, size=len(osm_points))

    sizes = np.clip(np.sqrt(scores["image_count"].astype(float).clip(lower=1)), 10, 40)
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=180)
    ax.scatter(
        imagery_points["x"],
        imagery_points["vpi_score"],
        s=sizes,
        color="#2166ac",
        alpha=0.72,
        edgecolors="none",
        label="Imagery VPI score",
    )
    ax.scatter(
        osm_points["x"],
        osm_points["osm_score"],
        s=sizes,
        color="#b2182b",
        alpha=0.72,
        edgecolors="none",
        label="OSM score",
    )

    imagery_medians = (
        scores.groupby("Income group", observed=False)["vpi_score"]
        .median()
        .reindex(income_order)
    )
    osm_medians = (
        scores.groupby("Income group", observed=False)["osm_score"]
        .median()
        .reindex(income_order)
    )

    valid_imagery = imagery_medians.dropna()
    if not valid_imagery.empty:
        ax.plot(
            [x_lookup[group] - 0.1 for group in valid_imagery.index],
            valid_imagery.values,
            color="#0f3f75",
            linewidth=1.8,
            marker="o",
            markersize=4,
        )

    valid_osm = osm_medians.dropna()
    if not valid_osm.empty:
        ax.plot(
            [x_lookup[group] + 0.1 for group in valid_osm.index],
            valid_osm.values,
            color="#7f1d1d",
            linewidth=1.8,
            marker="o",
            markersize=4,
        )

    ax.set_title("OSM Score vs Imagery VPI Score by Economic Classification")
    ax.set_xlabel("Economic classification")
    ax.set_ylabel("Score")
    ax.set_xticks(range(len(income_order)))
    ax.set_xticklabels(income_order, rotation=25, ha="right")
    ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")

    summary = (
        f"regions={scores['region_id'].nunique()} | "
        f"median VPI={scores['vpi_score'].median():.4f} | "
        f"median OSM={scores['osm_score'].median():.4f} | "
        f"median OSM-VPI={scores['score_difference'].median():.4f}"
    )
    ax.text(
        0.995,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_scores_by_economic_candlestick(scores, output_path):
    if scores.empty:
        raise SystemExit("No classified regions left after filtering.")

    income_order = order_income_groups(scores["Income group"].unique())
    if not income_order:
        raise SystemExit("No economic classifications available for the plotted regions.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_lookup = {label: index for index, label in enumerate(income_order)}
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=180)

    def summary_table(column_name):
        grouped = scores.groupby("Income group", observed=False)[column_name]
        return pd.DataFrame(
            {
                "min": grouped.min(),
                "q1": grouped.quantile(0.25),
                "mean": grouped.mean(),
                "median": grouped.median(),
                "q3": grouped.quantile(0.75),
                "max": grouped.max(),
            }
        ).reindex(income_order)

    def draw_summary(summary, offset, body_color, label):
        mean_points_x = []
        mean_points_y = []
        first = True
        for income_group in income_order:
            row = summary.loc[income_group]
            if row.isna().any():
                continue
            x_value = x_lookup[income_group] + offset
            mean_points_x.append(x_value)
            mean_points_y.append(row["mean"])
            ax.vlines(
                x_value,
                row["min"],
                row["max"],
                color=body_color,
                linewidth=1.5,
                alpha=0.8,
            )
            ax.hlines(
                [row["min"], row["max"]],
                x_value - 0.1,
                x_value + 0.1,
                color=body_color,
                linewidth=1.5,
                alpha=0.8,
            )
            rect = plt.Rectangle(
                (x_value - 0.14, row["q1"]),
                0.28,
                max(row["q3"] - row["q1"], 1e-9),
                facecolor=body_color,
                edgecolor=body_color,
                alpha=0.55,
                label=label if first else None,
            )
            first = False
            ax.add_patch(rect)
            ax.hlines(
                row["median"],
                x_value - 0.11,
                x_value + 0.11,
                color="#111827",
                linewidth=1.3,
            )
        if mean_points_x:
            ax.plot(
                mean_points_x,
                mean_points_y,
                color=body_color,
                linewidth=1.5,
                alpha=0.95,
                linestyle="--",
            )

    draw_summary(summary_table("vpi_score"), -0.14, "#2166ac", "Imagery VPI score")
    draw_summary(summary_table("osm_score"), 0.14, "#b2182b", "OSM score")

    ax.set_title("OSM Score vs Imagery VPI Score by Economic Classification")
    ax.set_xlabel("Economic classification")
    ax.set_ylabel("Score")
    ax.set_xticks(range(len(income_order)))
    ax.set_xticklabels(income_order, rotation=25, ha="right")
    ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")

    summary = (
        f"regions={scores['region_id'].nunique()} | "
        f"median VPI={scores['vpi_score'].median():.4f} | "
        f"median OSM={scores['osm_score'].median():.4f} | "
        f"median OSM-VPI={scores['score_difference'].median():.4f}"
    )
    ax.text(
        0.995,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_scores_by_economic_classification(scores, output_path, chart_style):
    if chart_style == "candlestick":
        plot_scores_by_economic_candlestick(scores, output_path)
        return
    plot_scores_by_economic_scatter(scores, output_path)


def save_ordered_csv(scores, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_csv, index=False)


def main():
    args = parse_args()
    scores = load_scores_from_db() if args.from_db else load_scores_from_csv(args.input_csv)
    scores = enrich_scores_with_region_metadata(scores)
    scores = prepare_scores(
        scores,
        args.min_images,
        args.min_osm_features,
        args.exclude_zero_scores,
        args.descending,
    )
    classifications = load_classification_table(args.classification_csv)
    classified_scores = merge_with_classifications(scores, classifications)
    output_path = args.output or with_style_suffix(
        Path("maps/osm_vs_imagery_scores.png"),
        args.chart_style,
    )
    economic_output_path = args.economic_output or with_style_suffix(
        Path("maps/osm_vs_imagery_scores_economic.png"),
        args.economic_chart_style,
    )
    plot_scores(scores, output_path, args.chart_style)
    plot_scores_by_economic_classification(
        classified_scores,
        economic_output_path,
        args.economic_chart_style,
    )
    save_ordered_csv(scores, args.output_csv)

    print(f"Saved plot to {output_path}")
    print(f"Saved economic classification plot to {economic_output_path}")
    print(f"Saved ordered scores to {args.output_csv}")
    print(f"Regions plotted: {len(scores)}")
    print(
        "Scores are ordered by imagery VPI score on the x-axis; vertical grey "
        "segments show OSM/VPI divergence for each region."
    )
    print(f"Ordered chart style: {args.chart_style}")
    print(f"Economic chart style: {args.economic_chart_style}")


if __name__ == "__main__":
    raise SystemExit(main())
