import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager, Image, OSMFeature  # noqa: E402


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
        default=Path("maps/osm_vs_imagery_scores.png"),
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
                "image_count": int(image_counts.get(region_id, 0)),
                "osm_feature_count": int(osm_feature_counts.get(region_id, 0)),
                "vpi_score": float(vpi_scores.get(region_id, 0.0)),
                "osm_score": float(osm_scores.get(region_id, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def prepare_scores(scores, min_images, min_osm_features, exclude_zero_scores, descending):
    scores = scores.copy()
    for column in ["image_count", "osm_feature_count"]:
        if column not in scores.columns:
            scores[column] = 0
    scores["vpi_score"] = pd.to_numeric(scores["vpi_score"], errors="coerce").fillna(0)
    scores["osm_score"] = pd.to_numeric(scores["osm_score"], errors="coerce").fillna(0)
    scores["image_count"] = pd.to_numeric(scores["image_count"], errors="coerce").fillna(0)
    scores["osm_feature_count"] = pd.to_numeric(
        scores["osm_feature_count"],
        errors="coerce",
    ).fillna(0)

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


def plot_scores(scores, output_path):
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


def save_ordered_csv(scores, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_csv, index=False)


def main():
    args = parse_args()
    scores = load_scores_from_db() if args.from_db else load_scores_from_csv(args.input_csv)
    scores = prepare_scores(
        scores,
        args.min_images,
        args.min_osm_features,
        args.exclude_zero_scores,
        args.descending,
    )
    plot_scores(scores, args.output)
    save_ordered_csv(scores, args.output_csv)

    print(f"Saved plot to {args.output}")
    print(f"Saved ordered scores to {args.output_csv}")
    print(f"Regions plotted: {len(scores)}")
    print(
        "Scores are ordered by imagery VPI score on the x-axis; vertical grey "
        "segments show OSM/VPI divergence for each region."
    )


if __name__ == "__main__":
    raise SystemExit(main())
