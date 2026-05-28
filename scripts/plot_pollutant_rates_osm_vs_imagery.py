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

INCOME_COLOR_MAP = {
    "LI": "#b2182b",
    "LMI": "#ef8a62",
    "UMI": "#67a9cf",
    "HI": "#2166ac",
}


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
        default=None,
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
        "--chart-style",
        choices=["scatter"],
        default="scatter",
        help="Render the per-pollutant ordered plots. Ordered plots currently use scatter only.",
    )
    parser.add_argument(
        "--label-top-differences",
        type=int,
        default=0,
        help="Label this many regions with the largest absolute rate differences.",
    )
    parser.add_argument(
        "--exclude-outliers",
        action="store_true",
        help="Exclude plot-only outlier regions using an IQR rule on imagery and OSM rates.",
    )
    parser.add_argument(
        "--outlier-iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier used when --exclude-outliers is enabled.",
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=Path("data/CLASS_2025_10_07.csv"),
        help="CSV containing Code and Income group columns for economic classification plots.",
    )
    parser.add_argument(
        "--economic-output-dir",
        type=Path,
        default=None,
        help="Folder for per-pollutant economic-classification scatter plot PNGs.",
    )
    parser.add_argument(
        "--economic-chart-style",
        choices=["scatter", "candlestick"],
        default="scatter",
        help="Render the economic-classification plots as scatter plots or candlestick summary charts.",
    )
    parser.add_argument(
        "--economic-scatter-mode",
        choices=["xy", "split"],
        default="xy",
        help="When --economic-chart-style=scatter, use an x-y comparison scatter or the previous split-by-source economic scatter.",
    )
    parser.add_argument(
        "--bias-output-dir",
        type=Path,
        default=None,
        help="Folder for per-pollutant country-level bias plot PNGs.",
    )
    parser.add_argument(
        "--correlation-stats-csv",
        type=Path,
        default=Path("data/pollutant_rates_osm_vs_imagery_correlation.csv"),
        help="CSV path for per-pollutant country-level correlation and bias stats.",
    )
    return parser.parse_args()


def slugify(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "blank"


def with_style_suffix(path, style):
    return path.with_name(f"{path.name}_{style}")


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


def get_split_axis_limits(values):
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    finite_values = finite_values[finite_values >= 0]
    if finite_values.size < 8:
        return None
    overall_max = float(np.max(finite_values))
    if overall_max <= 0:
        return None

    scale = max(overall_max, 1e-12)
    sorted_values = np.sort(finite_values)
    candidate_pairs = []

    q1, q3 = np.quantile(sorted_values, [0.25, 0.75])
    iqr = q3 - q1
    if iqr > 0:
        outlier_threshold = q3 + (1.5 * iqr)
        non_outliers = sorted_values[sorted_values <= outlier_threshold]
        outliers = sorted_values[sorted_values > outlier_threshold]
        if non_outliers.size >= 4 and outliers.size > 0:
            candidate_pairs.append(
                (float(np.max(non_outliers)), float(np.min(outliers))))

    for lower_q, upper_q in [(0.90, 0.98), (0.92, 0.99), (0.95, 0.995)]:
        lower_value = float(np.quantile(sorted_values, lower_q))
        upper_value = float(np.quantile(sorted_values, upper_q))
        if upper_value > lower_value:
            candidate_pairs.append((lower_value, upper_value))

    # Catch the "single extreme spike over a dense near-zero cloud" case.
    core_cap = float(np.quantile(sorted_values, 0.97))
    spike_floor = float(np.quantile(sorted_values, 0.995))
    if spike_floor > core_cap:
        candidate_pairs.append((core_cap, spike_floor))
    if sorted_values.size >= 2 and float(sorted_values[-1]) > float(sorted_values[-2]):
        candidate_pairs.append(
            (float(np.quantile(sorted_values, 0.98)), float(sorted_values[-1])))

    for lower_max, upper_min in candidate_pairs:
        gap = upper_min - lower_max
        if gap <= 0:
            continue

        gap_ratio = gap / scale
        separation_ratio = upper_min / max(lower_max, scale * 0.005, 1e-12)
        near_zero_cluster = lower_max <= (
            scale * 0.08) and upper_min >= (scale * 0.18)
        clearly_separated = gap_ratio >= 0.06 and separation_ratio >= 1.45
        single_spike_pattern = (
            upper_min >= scale * 0.45
            and lower_max <= scale * 0.25
            and separation_ratio >= 2.0
        )
        if not (near_zero_cluster or clearly_separated or single_spike_pattern):
            continue

        lower_pad = max(gap * 0.10, scale * 0.008)
        upper_pad = max(gap * 0.08, scale * 0.008)
        lower_limit = lower_max + lower_pad
        if lower_limit <= 0:
            lower_limit = max(scale * 0.04, upper_min * 0.35)
        upper_limit = upper_min - upper_pad
        if upper_limit <= lower_limit:
            upper_limit = lower_limit + max(scale * 0.02, gap * 0.20)
        if upper_limit >= overall_max:
            continue

        return {
            "lower_ylim": (0.0, lower_limit),
            "upper_ylim": (upper_limit, overall_max * 1.04),
        }

    return None


def create_y_axes(values, figsize):
    split_limits = get_split_axis_limits(values)
    if split_limits is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=180)
        return fig, [ax], None

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=figsize,
        dpi=180,
        gridspec_kw={"height_ratios": [1.2, 3.2], "hspace": 0.05},
    )
    ax_top.set_ylim(*split_limits["upper_ylim"])
    ax_bottom.set_ylim(*split_limits["lower_ylim"])
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bottom.xaxis.tick_bottom()

    diagonal_size = 0.012
    kwargs = dict(color="#111111", clip_on=False, linewidth=1.0)
    ax_top.plot((-diagonal_size, +diagonal_size), (-diagonal_size, +
                diagonal_size), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - diagonal_size, 1 + diagonal_size),
                (-diagonal_size, +diagonal_size), transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot((-diagonal_size, +diagonal_size), (1 - diagonal_size,
                   1 + diagonal_size), transform=ax_bottom.transAxes, **kwargs)
    ax_bottom.plot((1 - diagonal_size, 1 + diagonal_size), (1 - diagonal_size,
                   1 + diagonal_size), transform=ax_bottom.transAxes, **kwargs)
    return fig, [ax_top, ax_bottom], split_limits


def configure_split_axis(ax_top, ax_bottom, split_limits):
    ax_top.set_ylim(*split_limits["upper_ylim"])
    ax_bottom.set_ylim(*split_limits["lower_ylim"])
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bottom.xaxis.tick_bottom()

    diagonal_size = 0.012
    marker_kwargs = dict(
        marker=[(-1, -1), (1, 1)],
        markersize=9,
        linestyle="none",
        color="#111111",
        mec="#111111",
        mew=1.0,
        clip_on=False,
    )
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **marker_kwargs)
    ax_bottom.plot(
        [0, 1], [1, 1], transform=ax_bottom.transAxes, **marker_kwargs)


def create_source_axes_pair(imagery_values, osm_values, figsize):
    imagery_split = get_split_axis_limits(imagery_values)
    osm_split = get_split_axis_limits(osm_values)
    use_split_layout = imagery_split is not None or osm_split is not None

    fig = plt.figure(figsize=figsize, dpi=180, constrained_layout=True)
    if use_split_layout:
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1.2, 3.2],
            hspace=0.05,
            wspace=0.18,
        )
    else:
        gs = fig.add_gridspec(1, 2, wspace=0.18)

    axes_by_source = {}
    split_by_source = {"imagery": imagery_split, "osm": osm_split}
    for col_index, source in enumerate(["imagery", "osm"]):
        split_limits = split_by_source[source]
        if use_split_layout and split_limits is not None:
            ax_top = fig.add_subplot(gs[0, col_index])
            ax_bottom = fig.add_subplot(gs[1, col_index], sharex=ax_top)
            configure_split_axis(ax_top, ax_bottom, split_limits)
            axes_by_source[source] = [ax_top, ax_bottom]
        elif use_split_layout:
            ax_full = fig.add_subplot(gs[:, col_index])
            axes_by_source[source] = [ax_full]
        else:
            ax_full = fig.add_subplot(gs[0, col_index])
            axes_by_source[source] = [ax_full]

    return fig, axes_by_source, split_by_source


def apply_axis_formatting(axes, ylabel):
    for ax in axes:
        ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)
    if len(axes) == 1:
        axes[0].set_ylabel(ylabel)
    else:
        axes[0].set_ylabel("")
        axes[1].set_ylabel(ylabel)


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

    requested = [pollutant.strip()
                 for pollutant in args.pollutants if pollutant.strip()]
    unknown = sorted(set(requested) - set(DEFAULT_POLLUTANT_MAP))
    if unknown:
        raise SystemExit(
            "Unknown pollutant(s): "
            + ", ".join(unknown)
            + ". Available pollutants: "
            + ", ".join(sorted(DEFAULT_POLLUTANT_MAP))
        )
    return requested


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
    classifications["Code"] = classifications["Code"].astype(
        str).str.strip().str.upper()
    classifications["Income group"] = (
        classifications["Income group"].astype(str).str.strip()
    )
    classifications["Income group"] = classifications["Income group"].replace(
        INCOME_LABEL_MAP)
    return classifications


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
        db.session.query(Image.region_id, func.count(
            Image.id)).group_by(Image.region_id).all(),
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
            osm_count = sum_counts(
                osm_counts, region.region_id, mapping["osm"])
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


def merge_with_classifications(rates, classifications):
    if rates.empty:
        return rates

    merged = rates.copy()
    merged["iso3"] = merged["iso3"].fillna(
        "").astype(str).str.strip().str.upper()
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


def aggregate_rates_for_groups(rates, group_columns, rate_mode):
    if rates.empty:
        return rates.copy()

    grouped = (
        rates.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            image_count=("image_count", "sum"),
            total_detections=("total_detections", "sum"),
            total_osm_features=("total_osm_features", "sum"),
            imagery_count=("imagery_count", "sum"),
            osm_count=("osm_count", "sum"),
            region_count=("region_id", "nunique"),
        )
    )
    if rate_mode == "composition":
        grouped["imagery_rate"] = grouped.apply(
            lambda row: calculate_rate(row["imagery_count"], row["total_detections"]),
            axis=1,
        )
        grouped["osm_rate"] = grouped.apply(
            lambda row: calculate_rate(row["osm_count"], row["total_osm_features"]),
            axis=1,
        )
    else:
        grouped["imagery_rate"] = grouped.apply(
            lambda row: calculate_rate(row["imagery_count"], row["image_count"]),
            axis=1,
        )
        grouped["osm_rate"] = grouped.apply(
            lambda row: calculate_rate(row["osm_count"], row["image_count"]),
            axis=1,
        )
    grouped["rate_difference"] = grouped["osm_rate"] - grouped["imagery_rate"]
    grouped["absolute_rate_difference"] = grouped["rate_difference"].abs()
    return grouped


def aggregate_country_rates(rates, rate_mode):
    grouped = aggregate_rates_for_groups(rates, ["country", "iso3"], rate_mode)
    grouped["country"] = grouped["country"].fillna("").astype(str).str.strip()
    return grouped[grouped["country"] != ""].copy()


def order_income_groups(values):
    ordered = [group for group in DEFAULT_INCOME_ORDER if group in values]
    extras = sorted(
        value for value in values if value not in DEFAULT_INCOME_ORDER)
    return ordered + extras


def non_outlier_mask(values, iqr_multiplier):
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(True, index=values.index)

    q1 = float(finite.quantile(0.25))
    q3 = float(finite.quantile(0.75))
    iqr = q3 - q1
    if not math.isfinite(iqr) or iqr <= 0.0:
        return pd.Series(True, index=values.index)

    lower_bound = q1 - (float(iqr_multiplier) * iqr)
    upper_bound = q3 + (float(iqr_multiplier) * iqr)
    return (numeric >= lower_bound) & (numeric <= upper_bound)


def filter_outliers_for_plot(subset, iqr_multiplier):
    if subset.empty:
        return subset, 0

    mask = (
        non_outlier_mask(subset["imagery_rate"], iqr_multiplier)
        & non_outlier_mask(subset["osm_rate"], iqr_multiplier)
    )
    filtered = subset.loc[mask].copy()
    removed_count = int(len(subset) - len(filtered))

    # Keep the original data if filtering would wipe out the plot entirely.
    if filtered.empty:
        return subset.copy(), 0
    return filtered, removed_count


def reindex_ordered_subset(subset, descending):
    if subset.empty:
        return subset
    subset = subset.sort_values(
        ["imagery_rate", "country", "city", "region_name", "region_id"],
        ascending=[not descending, True, True, True, True],
    ).reset_index(drop=True)
    subset["ordered_region_index"] = subset.index + 1
    return subset


def region_label(row):
    parts = [part for part in [row.get("city"), row.get("country")] if part]
    return ", ".join(parts) if parts else row["region_id"][:8]


def apply_source_panel_formatting(axes_by_source, x_label, x_ticks=None, x_ticklabels=None):
    for source, axes in axes_by_source.items():
        title = "Imagery rate" if source == "imagery" else "OSM rate"
        axes[0].set_title(title)
        apply_axis_formatting(axes, "Rate")
        axes[-1].set_xlabel(x_label)
        if x_ticks is not None:
            for ax in axes:
                ax.set_xticks(x_ticks)
                if x_ticklabels is not None:
                    ax.set_xticklabels(x_ticklabels, rotation=25, ha="right")
        for ax in axes:
            ax.margins(x=0.01)


def plot_xy_rate_scatter(
    subset,
    output_path,
    label_column,
    title,
    point_size_column,
    size_scale,
    point_color,
):
    if subset.empty:
        raise SystemExit("No data available to plot.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 7.0), dpi=180)

    point_sizes = np.clip(
        np.sqrt(subset[point_size_column].astype(float).clip(lower=1.0)) * size_scale,
        28,
        135,
    )
    ax.scatter(
        subset["imagery_rate"],
        subset["osm_rate"],
        s=point_sizes,
        color=point_color,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.6,
    )

    axis_max = float(
        max(
            subset["imagery_rate"].max(),
            subset["osm_rate"].max(),
            0.01,
        )
    )
    axis_limit = axis_max * 1.05
    ax.plot(
        [0.0, axis_limit],
        [0.0, axis_limit],
        color="#6b7280",
        linewidth=1.0,
        linestyle="--",
        alpha=0.75,
    )

    for row in subset.itertuples(index=False):
        label = getattr(row, label_column, "")
        if not label:
            continue
        ax.annotate(
            str(label),
            (row.imagery_rate, row.osm_rate),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7.5,
            color="#262626",
        )

    ax.set_title(title, y=0.995)
    ax.set_xlabel("Imagery rate")
    ax.set_ylabel("OSM rate")
    ax.set_xlim(0.0, axis_limit)
    ax.set_ylim(0.0, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def compute_correlation_stats(subset):
    if subset.empty:
        return None

    imagery = pd.to_numeric(subset["imagery_rate"], errors="coerce")
    osm = pd.to_numeric(subset["osm_rate"], errors="coerce")
    paired = pd.DataFrame({"imagery_rate": imagery, "osm_rate": osm}).dropna()
    if paired.empty:
        return None

    pearson = float(paired["imagery_rate"].corr(paired["osm_rate"], method="pearson"))
    spearman = float(paired["imagery_rate"].corr(paired["osm_rate"], method="spearman"))

    slope = float("nan")
    intercept = float("nan")
    if len(paired) >= 2 and paired["imagery_rate"].nunique() >= 2:
        slope, intercept = np.polyfit(
            paired["imagery_rate"].to_numpy(),
            paired["osm_rate"].to_numpy(),
            deg=1,
        )

    difference = paired["osm_rate"] - paired["imagery_rate"]
    return {
        "point_count": int(len(paired)),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "slope": float(slope),
        "intercept": float(intercept),
        "mean_difference": float(difference.mean()),
        "median_difference": float(difference.median()),
        "mean_absolute_difference": float(difference.abs().mean()),
    }


def plot_country_correlation_scatter(subset, pollutant, output_path, label_top_differences=0):
    if subset.empty:
        return None

    stats = compute_correlation_stats(subset)
    if stats is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 7.0), dpi=180)
    point_sizes = np.clip(
        np.sqrt(subset["image_count"].astype(float).clip(lower=1.0)) * 1.4,
        28,
        135,
    )
    ax.scatter(
        subset["imagery_rate"],
        subset["osm_rate"],
        s=point_sizes,
        color="#7c3aed",
        alpha=0.78,
        edgecolors="white",
        linewidths=0.6,
    )

    axis_max = float(max(subset["imagery_rate"].max(), subset["osm_rate"].max(), 0.01))
    axis_limit = axis_max * 1.05
    ax.plot(
        [0.0, axis_limit],
        [0.0, axis_limit],
        color="#6b7280",
        linewidth=1.0,
        linestyle="--",
        alpha=0.75,
    )

    if math.isfinite(stats["slope"]) and math.isfinite(stats["intercept"]):
        x_values = np.array([0.0, axis_limit], dtype=float)
        y_values = (stats["slope"] * x_values) + stats["intercept"]
        ax.plot(
            x_values,
            y_values,
            color="#4c1d95",
            linewidth=1.5,
            alpha=0.85,
        )

    if label_top_differences:
        labels = subset.nlargest(label_top_differences, "absolute_rate_difference")
        for _, row in labels.iterrows():
            ax.annotate(
                region_label(row),
                (row["imagery_rate"], row["osm_rate"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7.5,
                color="#262626",
            )

    pretty_pollutant = pollutant.replace("_", " ").title()
    ax.set_title(pretty_pollutant, y=0.995)
    ax.set_xlabel("Imagery rate")
    ax.set_ylabel("OSM rate")
    ax.set_xlim(0.0, axis_limit)
    ax.set_ylim(0.0, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.text(
        0.995,
        0.02,
        (
            f"regions={stats['point_count']} | "
            f"pearson={stats['pearson_r']:.3f} | "
            f"spearman={stats['spearman_r']:.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#374151",
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return stats


def plot_country_bias(subset, pollutant, output_path):
    if subset.empty:
        return False

    plotted = subset.copy()
    plotted["mean_rate"] = (plotted["imagery_rate"] + plotted["osm_rate"]) / 2.0
    plotted["rate_difference"] = plotted["osm_rate"] - plotted["imagery_rate"]
    plotted["Income group"] = plotted["Income group"].fillna("").astype(str).str.strip()
    plotted["point_color"] = plotted["Income group"].map(INCOME_COLOR_MAP).fillna("#6b7280")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 7.0), dpi=180)
    point_sizes = np.clip(
        np.sqrt(plotted["image_count"].astype(float).clip(lower=1.0)) * 1.4,
        28,
        135,
    )
    ax.scatter(
        plotted["mean_rate"],
        plotted["rate_difference"],
        s=point_sizes,
        c=plotted["point_color"],
        alpha=0.78,
        edgecolors="white",
        linewidths=0.6,
    )
    ax.axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--", alpha=0.75)

    pretty_pollutant = pollutant.replace("_", " ").title()
    ax.set_title(pretty_pollutant, y=0.995)
    ax.set_xlabel("Mean rate")
    ax.set_ylabel("OSM - imagery")
    ax.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)

    legend_handles = []
    for income_group in DEFAULT_INCOME_ORDER:
        if income_group not in set(plotted["Income group"]):
            continue
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=INCOME_COLOR_MAP.get(income_group, "#6b7280"),
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=7,
                label=income_group,
            )
        )
    if "" in set(plotted["Income group"]):
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="#6b7280",
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=7,
                label="Unknown",
            )
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            title="Income group",
        )

    ax.text(
        0.995,
        0.02,
        (
            f"regions={len(plotted)} | "
            f"mean diff={plotted['rate_difference'].mean():.3f} | "
            f"median diff={plotted['rate_difference'].median():.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    fig.savefig(output_path)
    plt.close(fig)
    return True


def plot_pollutant_scatter(subset, pollutant, rate_mode, output_path, label_top_differences, exclude_outliers=False, outlier_iqr_multiplier=1.5):
    if exclude_outliers:
        subset, _ = filter_outliers_for_plot(subset, outlier_iqr_multiplier)
    plot_country_correlation_scatter(
        subset,
        pollutant,
        output_path,
        label_top_differences=label_top_differences,
    )


def plot_pollutant_candlestick(subset, pollutant, rate_mode, output_path, label_top_differences):
    subset = subset.sort_values("ordered_region_index").copy()
    x = subset["ordered_region_index"]
    fig, axes_by_source, split_by_source = create_source_axes_pair(
        subset["imagery_rate"].to_numpy(),
        subset["osm_rate"].to_numpy(),
        figsize=(15.5, 6.8),
    )
    for ax in axes_by_source["imagery"]:
        draw_candles(
            ax,
            x,
            np.zeros(len(subset)),
            subset["imagery_rate"],
            width=0.82,
            up_color="#2166ac",
            down_color="#2166ac",
            alpha=0.72,
            linewidth=1.6,
            cap_width_ratio=0.85,
        )
    for ax in axes_by_source["osm"]:
        draw_candles(
            ax,
            x,
            np.zeros(len(subset)),
            subset["osm_rate"],
            width=0.82,
            up_color="#b2182b",
            down_color="#b2182b",
            alpha=0.72,
            linewidth=1.6,
            cap_width_ratio=0.85,
        )
    for ax in axes_by_source["imagery"]:
        ax.plot(
            x,
            subset["imagery_rate"],
            color="#0f3f75",
            linewidth=1.5,
            alpha=0.9,
            label="Mean",
        )
    for ax in axes_by_source["osm"]:
        ax.plot(
            x,
            subset["osm_rate"],
            color="#7f1d1d",
            linewidth=1.5,
            alpha=0.9,
            label="Mean",
        )

    if label_top_differences:
        labels = subset.nlargest(
            label_top_differences, "absolute_rate_difference")
        for _, row in labels.iterrows():
            for source, value_column in [("imagery", "imagery_rate"), ("osm", "osm_rate")]:
                label_y = row[value_column]
                split_limits = split_by_source[source]
                target_ax = (
                    axes_by_source[source][0]
                    if split_limits and label_y >= split_limits["upper_ylim"][0]
                    else axes_by_source[source][-1]
                )
                target_ax.annotate(
                    region_label(row),
                    (row["ordered_region_index"], label_y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    color="#262626",
                )

    pretty_pollutant = pollutant.replace("_", " ").title()
    fig.suptitle(pretty_pollutant, y=0.995)
    apply_source_panel_formatting(
        axes_by_source, "Regions ordered by imagery rate")
    for axes in axes_by_source.values():
        axes[-1].legend(frameon=False, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_pollutant(
    subset,
    pollutant,
    rate_mode,
    output_path,
    label_top_differences,
    chart_style,
    exclude_outliers=False,
    outlier_iqr_multiplier=1.5,
):
    plot_pollutant_scatter(
        subset,
        pollutant,
        rate_mode,
        output_path,
        label_top_differences,
        exclude_outliers=exclude_outliers,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
    )


def plot_pollutant_by_economic_xy_scatter(subset, pollutant, rate_mode, output_path):
    grouped = aggregate_rates_for_groups(subset, ["Income group"], rate_mode)
    income_order = order_income_groups(grouped["Income group"].tolist())
    if not income_order:
        return False
    grouped["Income group"] = pd.Categorical(
        grouped["Income group"], categories=income_order, ordered=True
    )
    grouped = grouped.sort_values("Income group").copy()
    pretty_pollutant = pollutant.replace("_", " ").title()
    plot_xy_rate_scatter(
        grouped,
        output_path,
        label_column="Income group",
        title=pretty_pollutant,
        point_size_column="image_count",
        size_scale=2.2,
        point_color="#0f766e",
    )
    return True


def plot_pollutant_by_economic_split_scatter(subset, pollutant, rate_mode, output_path):
    income_order = order_income_groups(subset["Income group"].unique())
    if not income_order:
        return False

    x_lookup = {label: index for index, label in enumerate(income_order)}
    rng = np.random.default_rng(42)

    imagery_points = subset.copy()
    imagery_points["x"] = imagery_points["Income group"].map(
        x_lookup).astype(float)
    imagery_points["x"] = imagery_points["x"] + \
        rng.uniform(-0.18, -0.03, size=len(imagery_points))

    osm_points = subset.copy()
    osm_points["x"] = subset["Income group"].map(x_lookup).astype(float)
    osm_points["x"] = osm_points["x"] + \
        rng.uniform(0.03, 0.18, size=len(osm_points))

    fig, axes_by_source, _ = create_source_axes_pair(
        subset["imagery_rate"].to_numpy(),
        subset["osm_rate"].to_numpy(),
        figsize=(12.5, 6.8),
    )
    sizes = np.clip(np.sqrt(subset["image_count"].astype(
        float).clip(lower=1)) * 1.35, 15, 58)

    for ax in axes_by_source["imagery"]:
        ax.scatter(
            imagery_points["x"],
            imagery_points["imagery_rate"],
            s=sizes,
            color="#2166ac",
            alpha=0.7,
            edgecolors="none",
        )
    for ax in axes_by_source["osm"]:
        ax.scatter(
            osm_points["x"],
            osm_points["osm_rate"],
            s=sizes,
            color="#b2182b",
            alpha=0.7,
            edgecolors="none",
        )

    imagery_medians = (
        subset.groupby("Income group", observed=False)["imagery_rate"]
        .median()
        .reindex(income_order)
    )
    osm_medians = (
        subset.groupby("Income group", observed=False)["osm_rate"]
        .median()
        .reindex(income_order)
    )

    valid_imagery = imagery_medians.dropna()
    if not valid_imagery.empty:
        for ax in axes_by_source["imagery"]:
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
        for ax in axes_by_source["osm"]:
            ax.plot(
                [x_lookup[group] + 0.1 for group in valid_osm.index],
                valid_osm.values,
                color="#7f1d1d",
                linewidth=1.8,
                marker="o",
                markersize=4,
            )

    pretty_pollutant = pollutant.replace("_", " ").title()
    fig.suptitle(pretty_pollutant, y=0.995)
    apply_source_panel_formatting(
        axes_by_source,
        "Economic classification",
        x_ticks=range(len(income_order)),
        x_ticklabels=income_order,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return True


def plot_pollutant_by_economic_candlestick(subset, pollutant, rate_mode, output_path):
    income_order = order_income_groups(subset["Income group"].unique())
    if not income_order:
        return False

    x_lookup = {label: index for index, label in enumerate(income_order)}
    fig, axes_by_source, _ = create_source_axes_pair(
        subset["imagery_rate"].to_numpy(),
        subset["osm_rate"].to_numpy(),
        figsize=(12.5, 6.8),
    )

    def summary_table(column_name):
        grouped = subset.groupby("Income group", observed=False)[column_name]
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

    def draw_summary(summary, axes, body_color, label):
        mean_points_x = []
        mean_points_y = []
        first = True
        for income_group in income_order:
            row = summary.loc[income_group]
            if row.isna().any():
                continue
            x_value = x_lookup[income_group]
            mean_points_x.append(x_value)
            mean_points_y.append(row["mean"])
            for axis_index, ax in enumerate(axes):
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
                    label=label if first and axis_index == len(
                        axes) - 1 else None,
                )
                ax.add_patch(rect)
                ax.hlines(
                    row["median"],
                    x_value - 0.14,
                    x_value + 0.14,
                    color="#111827",
                    linewidth=1.3,
                )
            first = False
        if mean_points_x:
            for ax in axes:
                ax.plot(
                    mean_points_x,
                    mean_points_y,
                    color=body_color,
                    linewidth=1.5,
                    alpha=0.95,
                    linestyle="--",
                )

    draw_summary(summary_table("imagery_rate"),
                 axes_by_source["imagery"], "#2166ac", "Imagery rate")
    draw_summary(summary_table("osm_rate"),
                 axes_by_source["osm"], "#b2182b", "OSM rate")

    pretty_pollutant = pollutant.replace("_", " ").title()
    fig.suptitle(pretty_pollutant, y=0.995)
    apply_source_panel_formatting(
        axes_by_source,
        "Economic classification",
        x_ticks=range(len(income_order)),
        x_ticklabels=income_order,
    )
    for axes in axes_by_source.values():
        axes[-1].legend(frameon=False, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return True


def plot_pollutant_by_economic_classification(
    subset,
    pollutant,
    rate_mode,
    output_path,
    chart_style,
    scatter_mode,
):
    if chart_style == "candlestick":
        return plot_pollutant_by_economic_candlestick(
            subset,
            pollutant,
            rate_mode,
            output_path,
        )
    if scatter_mode == "split":
        return plot_pollutant_by_economic_split_scatter(
            subset,
            pollutant,
            rate_mode,
            output_path,
        )
    return plot_pollutant_by_economic_xy_scatter(
        subset,
        pollutant,
        rate_mode,
        output_path,
    )


def export_plots(rates, rate_mode, output_dir, label_top_differences, chart_style, exclude_outliers, outlier_iqr_multiplier, descending):
    if rates.empty:
        raise SystemExit("No rates available after filtering.")

    output_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    removed_total = 0
    for pollutant in sorted(rates["pollutant"].unique()):
        subset = rates[rates["pollutant"] == pollutant].copy()
        if subset.empty:
            continue
        if exclude_outliers and chart_style == "scatter":
            country_subset = subset.copy()
            _, removed_count = filter_outliers_for_plot(
                country_subset, outlier_iqr_multiplier)
            removed_total += removed_count
        elif exclude_outliers:
            subset, removed_count = filter_outliers_for_plot(
                subset, outlier_iqr_multiplier)
            removed_total += removed_count
            subset = reindex_ordered_subset(subset, descending)
        output_path = output_dir / f"{slugify(pollutant)}.png"
        plot_pollutant(
            subset,
            pollutant,
            rate_mode,
            output_path,
            label_top_differences,
            chart_style,
            exclude_outliers=exclude_outliers,
            outlier_iqr_multiplier=outlier_iqr_multiplier,
        )
        plotted += 1
    return plotted, removed_total


def export_economic_plots(rates, rate_mode, output_dir, classifications, chart_style, scatter_mode, exclude_outliers, outlier_iqr_multiplier):
    classified = merge_with_classifications(rates, classifications)
    if classified.empty:
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    removed_total = 0
    for pollutant in sorted(classified["pollutant"].unique()):
        subset = classified[classified["pollutant"] == pollutant].copy()
        if subset.empty:
            continue
        if exclude_outliers:
            subset, removed_count = filter_outliers_for_plot(
                subset, outlier_iqr_multiplier)
            removed_total += removed_count
        output_path = output_dir / f"{slugify(pollutant)}.png"
        if plot_pollutant_by_economic_classification(
            subset,
            pollutant,
            rate_mode,
            output_path,
            chart_style,
            scatter_mode,
        ):
            plotted += 1
    return plotted, removed_total


def export_country_bias_plots(rates, rate_mode, output_dir, classifications, exclude_outliers, outlier_iqr_multiplier):
    output_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    removed_total = 0
    for pollutant in sorted(rates["pollutant"].unique()):
        subset = rates[rates["pollutant"] == pollutant].copy()
        if subset.empty:
            continue
        country_subset = merge_with_classifications(subset, classifications)
        if exclude_outliers:
            country_subset, removed_count = filter_outliers_for_plot(
                country_subset, outlier_iqr_multiplier)
            removed_total += removed_count
        output_path = output_dir / f"{slugify(pollutant)}.png"
        if plot_country_bias(country_subset, pollutant, output_path):
            plotted += 1
    return plotted, removed_total


def export_correlation_stats(rates, rate_mode, output_path, exclude_outliers, outlier_iqr_multiplier):
    rows = []
    for pollutant in sorted(rates["pollutant"].unique()):
        subset = rates[rates["pollutant"] == pollutant].copy()
        if subset.empty:
            continue
        country_subset = subset.copy()
        if exclude_outliers:
            country_subset, _ = filter_outliers_for_plot(
                country_subset, outlier_iqr_multiplier)
        stats = compute_correlation_stats(country_subset)
        if stats is None:
            continue
        rows.append(
            {
                "pollutant": pollutant,
                "region_count": stats["point_count"],
                "pearson_r": stats["pearson_r"],
                "spearman_r": stats["spearman_r"],
                "slope": stats["slope"],
                "intercept": stats["intercept"],
                "mean_osm_minus_imagery": stats["mean_difference"],
                "median_osm_minus_imagery": stats["median_difference"],
                "mean_absolute_difference": stats["mean_absolute_difference"],
            }
        )

    stats_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_path, index=False)
    return len(stats_df)


def main():
    args = parse_args()
    pollutants = validate_args(args)
    if args.outlier_iqr_multiplier <= 0:
        raise SystemExit("--outlier-iqr-multiplier must be greater than 0.")
    classifications = load_classification_table(args.classification_csv)
    db = DatabaseManager()
    output_dir = args.output_dir or with_style_suffix(
        Path("maps/pollutant_rates_osm_vs_imagery"),
        args.chart_style,
    )
    economic_output_dir = args.economic_output_dir or with_style_suffix(
        Path("maps/pollutant_rates_osm_vs_imagery_economic"),
        args.economic_chart_style,
    )
    bias_output_dir = args.bias_output_dir or Path("maps/pollutant_rates_osm_vs_imagery_bias")

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
    plot_count, ordered_removed_count = export_plots(
        rates,
        args.rate_mode,
        output_dir,
        args.label_top_differences,
        args.chart_style,
        args.exclude_outliers,
        args.outlier_iqr_multiplier,
        args.descending,
    )
    economic_plot_count, economic_removed_count = export_economic_plots(
        rates,
        args.rate_mode,
        economic_output_dir,
        classifications,
        args.economic_chart_style,
        args.economic_scatter_mode,
        args.exclude_outliers,
        args.outlier_iqr_multiplier,
    )
    bias_plot_count, bias_removed_count = export_country_bias_plots(
        rates,
        args.rate_mode,
        bias_output_dir,
        classifications,
        args.exclude_outliers,
        args.outlier_iqr_multiplier,
    )
    correlation_stats_count = export_correlation_stats(
        rates,
        args.rate_mode,
        args.correlation_stats_csv,
        args.exclude_outliers,
        args.outlier_iqr_multiplier,
    )

    print(f"Saved {plot_count} pollutant scatter plot(s) to {output_dir}")
    print(
        f"Saved {economic_plot_count} pollutant economic scatter plot(s) to "
        f"{economic_output_dir}"
    )
    print(f"Saved {bias_plot_count} pollutant bias plot(s) to {bias_output_dir}")
    print(
        f"Saved {correlation_stats_count} pollutant correlation row(s) to "
        f"{args.correlation_stats_csv}"
    )
    print(f"Saved per-region rate table to {args.output_csv}")
    print(
        f"Regions included before pollutant zero filtering: {totals['region_id'].nunique()}")
    print(f"Rate mode: {args.rate_mode} - {RATE_MODE_HELP[args.rate_mode]}")
    print(f"Ordered chart style: {args.chart_style}")
    print(f"Economic chart style: {args.economic_chart_style}")
    print(f"Economic scatter mode: {args.economic_scatter_mode}")
    if args.exclude_outliers:
        print(
            f"Plot-only outlier filtering enabled with IQR multiplier "
            f"{args.outlier_iqr_multiplier:.3f}"
        )
        print(f"Ordered plots removed outlier rows: {ordered_removed_count}")
        print(f"Economic plots removed outlier rows: {economic_removed_count}")
        print(f"Bias plots removed outlier rows: {bias_removed_count}")


if __name__ == "__main__":
    raise SystemExit(main())
