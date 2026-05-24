import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualise score divergence and data coverage for regions ordered "
            "by imagery VPI score."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/scores_compare_vpi_osm.csv"),
        help="CSV containing vpi_score, osm_score, image_count, and osm_feature_count.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maps/score_data_coverage.png"),
        help="Output combined plot image path.",
    )
    parser.add_argument(
        "--osm-ordered-output",
        type=Path,
        default=Path("maps/score_data_coverage_ordered_by_osm.png"),
        help="Output combined plot ordered by OSM score.",
    )
    parser.add_argument(
        "--imagery-output",
        type=Path,
        default=Path("maps/imagery_data_coverage.png"),
        help="Output imagery coverage plot image path.",
    )
    parser.add_argument(
        "--osm-output",
        type=Path,
        default=Path("maps/osm_data_coverage.png"),
        help="Output OSM coverage plot image path.",
    )
    parser.add_argument(
        "--ordered-csv",
        type=Path,
        default=Path("data/score_data_coverage_ordered.csv"),
        help="Output ordered CSV with derived columns.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=75,
        help="Rolling median window used to smooth data-volume lines.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=1,
        help="Minimum image count to include.",
    )
    parser.add_argument(
        "--min-osm-features",
        type=int,
        default=1,
        help="Minimum OSM feature count to include.",
    )
    return parser.parse_args()


def load_scores(path):
    if not path.exists():
        raise SystemExit(f"Input CSV not found: {path}")
    scores = pd.read_csv(path)
    required = {"region_id", "vpi_score", "osm_score", "image_count", "osm_feature_count"}
    missing = required - set(scores.columns)
    if missing:
        raise SystemExit(f"Input CSV missing columns: {', '.join(sorted(missing))}")
    return scores


def prepare_scores(scores, min_images, min_osm_features):
    scores = scores.copy()
    numeric_columns = ["vpi_score", "osm_score", "image_count", "osm_feature_count"]
    for column in numeric_columns:
        scores[column] = pd.to_numeric(scores[column], errors="coerce").fillna(0)

    scores = scores[
        (scores["image_count"] >= min_images)
        & (scores["osm_feature_count"] >= min_osm_features)
    ].copy()
    scores["score_difference"] = scores["osm_score"] - scores["vpi_score"]
    scores["absolute_difference"] = scores["score_difference"].abs()
    return scores


def order_scores(scores, primary_score, secondary_score, rolling_window):
    ordered = scores.sort_values(
        [primary_score, secondary_score, "region_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    ordered["ordered_region_index"] = ordered.index + 1
    window = max(1, min(int(rolling_window), len(scores)))
    ordered["image_count_rolling_median"] = (
        ordered["image_count"].rolling(window, center=True, min_periods=1).median()
    )
    ordered["osm_feature_count_rolling_median"] = (
        ordered["osm_feature_count"].rolling(window, center=True, min_periods=1).median()
    )
    return ordered


def correlation_text(scores):
    image_corr = scores["vpi_score"].corr(scores["image_count"], method="spearman")
    osm_corr = scores["vpi_score"].corr(scores["osm_feature_count"], method="spearman")
    return (
        f"Spearman rho: VPI vs image count = {image_corr:.3f}; "
        f"VPI vs OSM features = {osm_corr:.3f}"
    )


def plot_combined_coverage(scores, output_path, ordered_by_label):
    if scores.empty:
        raise SystemExit("No rows left after filtering.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = scores["ordered_region_index"]

    fig, (score_ax, count_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.35]},
    )

    score_ax.scatter(
        x,
        scores["vpi_score"],
        s=14,
        color="#2166ac",
        alpha=0.75,
        label="Imagery VPI score",
    )
    score_ax.scatter(
        x,
        scores["osm_score"],
        s=14,
        color="#b2182b",
        alpha=0.75,
        label="OSM score",
    )
    score_ax.vlines(
        x,
        scores["vpi_score"],
        scores["osm_score"],
        color="#6b7280",
        alpha=0.14,
        linewidth=0.65,
    )
    score_ax.set_title(f"Scores and Data Coverage by Region Ordered by {ordered_by_label}")
    score_ax.set_ylabel("Score")
    score_ax.legend(frameon=False, loc="upper left")
    score_ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)

    count_ax.plot(
        x,
        scores["image_count_rolling_median"],
        color="#0f766e",
        linewidth=2.0,
        label="Image count rolling median",
    )
    count_ax.plot(
        x,
        scores["osm_feature_count_rolling_median"],
        color="#7c3aed",
        linewidth=2.0,
        label="OSM feature count rolling median",
    )
    count_ax.scatter(
        x,
        scores["image_count"],
        s=6,
        color="#0f766e",
        alpha=0.12,
    )
    count_ax.scatter(
        x,
        scores["osm_feature_count"],
        s=6,
        color="#7c3aed",
        alpha=0.12,
    )
    count_ax.set_xlabel(f"Regions ordered by {ordered_by_label}")
    count_ax.set_ylabel("Data count, log scale")
    count_ax.set_yscale("log")
    count_ax.legend(frameon=False, loc="upper right")
    count_ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)
    count_ax.text(
        0.005,
        0.04,
        correlation_text(scores),
        transform=count_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_single_coverage(scores, output_path, count_column, rolling_column, title, color):
    if scores.empty:
        raise SystemExit("No rows left after filtering.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = scores["ordered_region_index"]

    fig, (score_ax, count_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.55]},
    )

    score_ax.scatter(
        x,
        scores["vpi_score"],
        s=14,
        color="#2166ac",
        alpha=0.75,
        label="Imagery VPI score",
    )
    score_ax.scatter(
        x,
        scores["osm_score"],
        s=14,
        color="#b2182b",
        alpha=0.75,
        label="OSM score",
    )
    score_ax.set_title(title)
    score_ax.set_ylabel("Score")
    score_ax.legend(frameon=False, loc="upper left")
    score_ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)

    count_ax.scatter(
        x,
        scores[count_column],
        s=8,
        color=color,
        alpha=0.22,
        label=count_column.replace("_", " ").title(),
    )
    count_ax.plot(
        x,
        scores[rolling_column],
        color=color,
        linewidth=2.2,
        label="Rolling median",
    )
    count_ax.set_xlabel("Regions ordered by imagery VPI score")
    count_ax.set_ylabel("Data count, log scale")
    count_ax.set_yscale("log")
    count_ax.legend(frameon=False, loc="upper right")
    count_ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    scores = load_scores(args.input_csv)
    scores = prepare_scores(
        scores,
        args.min_images,
        args.min_osm_features,
    )
    vpi_ordered_scores = order_scores(
        scores,
        "vpi_score",
        "osm_score",
        args.rolling_window,
    )
    osm_ordered_scores = order_scores(
        scores,
        "osm_score",
        "vpi_score",
        args.rolling_window,
    )
    plot_combined_coverage(vpi_ordered_scores, args.output, "Imagery VPI Score")
    plot_combined_coverage(osm_ordered_scores, args.osm_ordered_output, "OSM Score")
    plot_single_coverage(
        vpi_ordered_scores,
        args.imagery_output,
        "image_count",
        "image_count_rolling_median",
        "Imagery Coverage by Region Ordered by VPI Score",
        "#0f766e",
    )
    plot_single_coverage(
        vpi_ordered_scores,
        args.osm_output,
        "osm_feature_count",
        "osm_feature_count_rolling_median",
        "OSM Coverage by Region Ordered by VPI Score",
        "#7c3aed",
    )
    args.ordered_csv.parent.mkdir(parents=True, exist_ok=True)
    vpi_ordered_scores.to_csv(args.ordered_csv, index=False)

    print(f"Saved coverage plot to {args.output}")
    print(f"Saved OSM-ordered coverage plot to {args.osm_ordered_output}")
    print(f"Saved imagery coverage plot to {args.imagery_output}")
    print(f"Saved OSM coverage plot to {args.osm_output}")
    print(f"Saved ordered coverage data to {args.ordered_csv}")
    print(f"Regions plotted: {len(vpi_ordered_scores)}")
    print(correlation_text(vpi_ordered_scores))


if __name__ == "__main__":
    raise SystemExit(main())
