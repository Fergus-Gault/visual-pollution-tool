import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import DatabaseManager, Detection, Image, OSMFeature, Region  # noqa: E402


DEFAULT_POLLUTANT_MAP = {
    "barrier": {"osm": ["barrier"], "imagery": ["barrier"]},
    "billboard": {"osm": ["advertising", "billboard"], "imagery": ["billboard"]},
    "bin": {"osm": ["bin"], "imagery": ["bin"]},
    "mobile_advertisement": {
        "osm": ["mobile_advertisement"],
        "imagery": ["mobile_advertisement"],
    },
    "road_sign": {"osm": ["traffic_sign"], "imagery": ["road_sign"]},
    "shop_sign": {"osm": ["shop_sign"], "imagery": ["shop_sign"]},
    "utility_pole": {"osm": ["power"], "imagery": ["utility_pole"]},
}

RATE_MODE_HELP = {
    "composition": (
        "category count divided by the source total in the same region "
        "(OSM category / all OSM features, imagery label / all detections)"
    ),
    "per-image": (
        "category count divided by image count for both sources; useful when "
        "you want comparable rates per sampled image"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create per-pollutant scatter plots comparing OSM and imagery rates "
            "by region, ordered by imagery rate."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maps/pollutant_rates_osm_vs_imagery"),
        help="Folder for generated scatter plot PNGs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/pollutant_rates_osm_vs_imagery.csv"),
        help="CSV path for the per-region rate table.",
    )
    parser.add_argument(
        "--rate-mode",
        choices=sorted(RATE_MODE_HELP),
        default="composition",
        help=(
            "How rates are calculated. composition: "
            f"{RATE_MODE_HELP['composition']}; per-image: "
            f"{RATE_MODE_HELP['per-image']}."
        ),
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=1,
        help="Minimum image count required for a region to be included.",
    )
    parser.add_argument(
        "--min-osm-features",
        type=int,
        default=1,
        help="Minimum OSM feature count required for a region to be included.",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=1,
        help="Minimum detection count required for a region to be included.",
    )
    parser.add_argument(
        "--pollutants",
        nargs="+",
        default=None,
        help=(
            "Optional pollutant names to plot. Defaults to all mapped pollutants: "
            + ", ".join(sorted(DEFAULT_POLLUTANT_MAP))
        ),
    )
    parser.add_argument(
        "--include-zero-zero",
        action="store_true",
        help="Keep points where both OSM and imagery rates are zero.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Order regions from highest imagery rate to lowest.",
    )
    parser.add_argument(
        "--label-top-differences",
        type=int,
        default=0,
        help="Label this many regions with the largest absolute rate differences.",
    )
    return parser.parse_args()


def slugify(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "blank"


def normalize_category(value):
    return (value or "").strip() or "<blank>"


def validate_args(args):
    if args.min_images < 0:
        raise SystemExit("--min-images must be zero or greater.")
    if args.min_osm_features < 0:
        raise SystemExit("--min-osm-features must be zero or greater.")
    if args.min_detections < 0:
        raise SystemExit("--min-detections must be zero or greater.")
    if args.label_top_differences < 0:
        raise SystemExit("--label-top-differences must be zero or greater.")

    if args.pollutants is None:
        return sorted(DEFAULT_POLLUTANT_MAP)

    requested = [pollutant.strip() for pollutant in args.pollutants if pollutant.strip()]
    unknown = sorted(set(requested) - set(DEFAULT_POLLUTANT_MAP))
    if unknown:
        raise SystemExit(
            "Unknown pollutant(s): "
            + ", ".join(unknown)
            + ". Available pollutants: "
            + ", ".join(sorted(DEFAULT_POLLUTANT_MAP))
        )
    return requested


def load_region_totals(db):
    regions = pd.DataFrame(
        db.session.query(
            Region.id.label("region_id"),
            Region.name.label("region_name"),
            Region.city.label("city"),
            Region.country.label("country"),
            Region.iso3.label("iso3"),
        ).all(),
        columns=["region_id", "region_name", "city", "country", "iso3"],
    )
    if regions.empty:
        return regions

    image_counts = pd.DataFrame(
        db.session.query(Image.region_id, func.count(Image.id)).group_by(Image.region_id).all(),
        columns=["region_id", "image_count"],
    )
    detection_counts = pd.DataFrame(
        db.session.query(Image.region_id, func.count(Detection.id))
        .join(Detection, Detection.image_id == Image.id)
        .group_by(Image.region_id)
        .all(),
        columns=["region_id", "total_detections"],
    )
    osm_counts = pd.DataFrame(
        db.session.query(OSMFeature.region_id, func.count(OSMFeature.id))
        .group_by(OSMFeature.region_id)
        .all(),
        columns=["region_id", "total_osm_features"],
    )

    totals = regions.merge(image_counts, on="region_id", how="left")
    totals = totals.merge(detection_counts, on="region_id", how="left")
    totals = totals.merge(osm_counts, on="region_id", how="left")
    count_columns = ["image_count", "total_detections", "total_osm_features"]
    totals[count_columns] = totals[count_columns].fillna(0).astype(int)
    for column in ["city", "country", "iso3"]:
        totals[column] = totals[column].fillna("")
    return totals


def load_detection_counts(db):
    rows = db.session.query(
        Image.region_id,
        Detection.label,
        func.count(Detection.id),
    ).join(Detection, Detection.image_id == Image.id).group_by(
        Image.region_id,
        Detection.label,
    ).all()

    counts = {}
    for region_id, raw_label, count in rows:
        label = normalize_category(raw_label)
        counts[(region_id, label)] = int(count)
    return counts


def load_osm_counts(db):
    rows = db.session.query(
        OSMFeature.region_id,
        OSMFeature.osm_type,
        func.count(OSMFeature.id),
    ).group_by(
        OSMFeature.region_id,
        OSMFeature.osm_type,
    ).all()

    counts = {}
    for region_id, raw_osm_type, count in rows:
        osm_type = normalize_category(raw_osm_type)
        counts[(region_id, osm_type)] = int(count)
    return counts


def sum_counts(counts, region_id, categories):
    return sum(counts.get((region_id, category), 0) for category in categories)


def calculate_rate(count, denominator):
    if denominator <= 0:
        return 0.0
    rate = float(count) / float(denominator)
    return rate if math.isfinite(rate) else 0.0


def build_rate_table(
    totals,
    detection_counts,
    osm_counts,
    pollutants,
    rate_mode,
    include_zero_zero,
):
    rows = []
    for region in totals.itertuples(index=False):
        for pollutant in pollutants:
            mapping = DEFAULT_POLLUTANT_MAP[pollutant]
            osm_count = sum_counts(osm_counts, region.region_id, mapping["osm"])
            imagery_count = sum_counts(
                detection_counts,
                region.region_id,
                mapping["imagery"],
            )

            if rate_mode == "composition":
                osm_denominator = region.total_osm_features
                imagery_denominator = region.total_detections
            else:
                osm_denominator = region.image_count
                imagery_denominator = region.image_count

            osm_rate = calculate_rate(osm_count, osm_denominator)
            imagery_rate = calculate_rate(imagery_count, imagery_denominator)
            if not include_zero_zero and osm_rate == 0 and imagery_rate == 0:
                continue

            rows.append(
                {
                    "pollutant": pollutant,
                    "region_id": region.region_id,
                    "region_name": region.region_name,
                    "city": region.city,
                    "country": region.country,
                    "iso3": region.iso3,
                    "image_count": region.image_count,
                    "total_detections": region.total_detections,
                    "total_osm_features": region.total_osm_features,
                    "imagery_categories": ";".join(mapping["imagery"]),
                    "osm_categories": ";".join(mapping["osm"]),
                    "imagery_count": imagery_count,
                    "osm_count": osm_count,
                    "imagery_rate": imagery_rate,
                    "osm_rate": osm_rate,
                    "rate_difference": osm_rate - imagery_rate,
                    "absolute_rate_difference": abs(osm_rate - imagery_rate),
                }
            )

    if not rows:
        return pd.DataFrame()

    rates = pd.DataFrame(rows)
    return rates


def order_rates_by_imagery(rates, descending):
    if rates.empty:
        return rates

    ordered_groups = []
    for pollutant in sorted(rates["pollutant"].unique()):
        subset = rates[rates["pollutant"] == pollutant].copy()
        subset = subset.sort_values(
            ["imagery_rate", "country", "city", "region_name", "region_id"],
            ascending=[not descending, True, True, True, True],
        ).reset_index(drop=True)
        subset["ordered_region_index"] = subset.index + 1
        ordered_groups.append(subset)

    return pd.concat(ordered_groups, ignore_index=True)


def filter_totals(totals, min_images, min_osm_features, min_detections):
    return totals[
        (totals["image_count"] >= min_images)
        & (totals["total_osm_features"] >= min_osm_features)
        & (totals["total_detections"] >= min_detections)
    ].copy()


def region_label(row):
    parts = [part for part in [row.get("city"), row.get("country")] if part]
    return ", ".join(parts) if parts else row["region_id"][:8]


def plot_pollutant(subset, pollutant, rate_mode, output_path, label_top_differences):
    subset = subset.sort_values("ordered_region_index").copy()
    x = subset["ordered_region_index"]

    fig, ax = plt.subplots(figsize=(14, 6.8), dpi=180)

    sizes = np.clip(np.sqrt(subset["image_count"].astype(float).clip(lower=1)), 9, 38)
    ax.scatter(
        x,
        subset["imagery_rate"],
        s=sizes,
        color="#2166ac",
        alpha=0.72,
        edgecolors="none",
        label="Imagery rate",
    )
    ax.scatter(
        x,
        subset["osm_rate"],
        s=sizes,
        color="#b2182b",
        alpha=0.72,
        edgecolors="none",
        label="OSM rate",
    )
    ax.vlines(
        x,
        subset["imagery_rate"],
        subset["osm_rate"],
        color="#6b7280",
        alpha=0.16,
        linewidth=0.7,
    )

    if label_top_differences:
        labels = subset.nlargest(label_top_differences, "absolute_rate_difference")
        for _, row in labels.iterrows():
            label_y = max(row["imagery_rate"], row["osm_rate"])
            ax.annotate(
                region_label(row),
                (row["ordered_region_index"], label_y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                color="#262626",
            )

    pretty_pollutant = pollutant.replace("_", " ").title()
    ax.set_title(f"{pretty_pollutant}: OSM vs Imagery Rates by Region")
    ax.set_xlabel("Regions ordered by imagery rate")
    ax.set_ylabel("Rate")
    ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.margins(x=0.01)

    summary = (
        f"regions={subset['region_id'].nunique()} | "
        f"median imagery={subset['imagery_rate'].median():.4f} | "
        f"median OSM={subset['osm_rate'].median():.4f} | "
        f"median OSM-imagery={subset['rate_difference'].median():.4f} | "
        f"mode={rate_mode}"
    )
    ax.text(
        0.99,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#374151",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def export_plots(rates, rate_mode, output_dir, label_top_differences):
    if rates.empty:
        raise SystemExit("No rates available after filtering.")

    output_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    for pollutant in sorted(rates["pollutant"].unique()):
        subset = rates[rates["pollutant"] == pollutant].copy()
        if subset.empty:
            continue
        output_path = output_dir / f"{slugify(pollutant)}.png"
        plot_pollutant(
            subset,
            pollutant,
            rate_mode,
            output_path,
            label_top_differences,
        )
        plotted += 1
    return plotted


def main():
    args = parse_args()
    pollutants = validate_args(args)
    db = DatabaseManager()

    totals = load_region_totals(db)
    totals = filter_totals(
        totals,
        args.min_images,
        args.min_osm_features,
        args.min_detections,
    )
    if totals.empty:
        raise SystemExit("No regions matched the requested minimum counts.")

    detection_counts = load_detection_counts(db)
    osm_counts = load_osm_counts(db)
    rates = build_rate_table(
        totals,
        detection_counts,
        osm_counts,
        pollutants,
        args.rate_mode,
        args.include_zero_zero,
    )
    rates = order_rates_by_imagery(rates, args.descending)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(args.output_csv, index=False)
    plot_count = export_plots(
        rates,
        args.rate_mode,
        args.output_dir,
        args.label_top_differences,
    )

    print(f"Saved {plot_count} pollutant scatter plot(s) to {args.output_dir}")
    print(f"Saved per-region rate table to {args.output_csv}")
    print(f"Regions included before pollutant zero filtering: {totals['region_id'].nunique()}")
    print(f"Rate mode: {args.rate_mode} - {RATE_MODE_HELP[args.rate_mode]}")


if __name__ == "__main__":
    raise SystemExit(main())
