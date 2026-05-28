import argparse
import importlib.util
import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOP_COLOR = "#c43c39"
BOTTOM_COLOR = "#2b8a67"
BOTH_COLOR = "#7d6370"
BASE_WORLD_COLOR = "#efefeb"
BASE_EDGE_COLOR = "#b7b7b0"
LINE_COLOR = "#9a9a9a"
TEXT_COLOR = "#252525"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot country-level VPI quantile gaps and selected city-level "
            "top/bottom quantile bars."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/vpi_city_quantiles_by_country.csv"),
        help="Input CSV from scripts/vpi_city_quantiles_by_country.py.",
    )
    parser.add_argument(
        "--gap-output",
        type=Path,
        default=Path("maps/vpi_country_quantile_gap.png"),
        help="Output path for the country top-vs-bottom quantile gap plot.",
    )
    parser.add_argument(
        "--bars-output",
        type=Path,
        default=Path("maps/vpi_city_quantile_bars.png"),
        help="Output path for selected city quantile bar plot.",
    )
    parser.add_argument(
        "--top-countries",
        type=int,
        default=25,
        help="Number of countries to include in the country gap plot.",
    )
    parser.add_argument(
        "--bar-countries",
        type=int,
        default=8,
        help="Number of countries to include in the city bar plot.",
    )
    parser.add_argument(
        "--cities-per-group",
        type=int,
        default=4,
        help="Maximum top and bottom cities to show per country in the bar plot.",
    )
    parser.add_argument(
        "--min-tested-cities",
        type=int,
        default=3,
        help="Minimum tested cities required for a country to appear in plots.",
    )
    parser.add_argument(
        "--country-order",
        choices=["gap", "tested_cities", "top_mean"],
        default="gap",
        help="How to choose and order countries in the plots.",
    )
    parser.add_argument(
        "--country-map-output",
        type=Path,
        default=Path("maps/vpi_country_quantile_membership_map.png"),
        help=(
            "Output path for an optional world map showing countries that appear "
            "in the selected top and bottom quantile groups."
        ),
    )
    parser.add_argument(
        "--skip-country-map",
        action="store_true",
        help="Skip generating the world country membership map.",
    )
    return parser.parse_args()


def load_quantiles(path):
    if not path.exists():
        raise SystemExit(f"Input CSV not found: {path}")

    data = pd.read_csv(path)
    required = {
        "quantile_group",
        "country",
        "tested_city_count",
        "city",
        "vpi_score",
        "total_images",
        "region_count",
    }
    missing = required - set(data.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise SystemExit(f"Input CSV missing required columns: {missing_text}")

    data = data.copy()
    data["quantile_group"] = data["quantile_group"].astype(str).str.lower()
    data = data[data["quantile_group"].isin(["top", "bottom"])]
    for column in ["tested_city_count", "vpi_score", "total_images", "region_count"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)

    data["country"] = data["country"].fillna("").astype(str)
    data["city"] = data["city"].fillna("").astype(str)
    return data[(data["country"] != "") & (data["city"] != "")]


def build_country_summary(data, min_tested_cities):
    grouped = (
        data.groupby(["country", "quantile_group"], as_index=False)
        .agg(
            mean_vpi=("vpi_score", "mean"),
            selected_city_count=("city", "nunique"),
            tested_city_count=("tested_city_count", "max"),
        )
    )
    pivot = grouped.pivot(index="country", columns="quantile_group", values="mean_vpi")
    counts = grouped.groupby("country", as_index=True).agg(
        tested_city_count=("tested_city_count", "max"),
        selected_city_count=("selected_city_count", "sum"),
    )
    summary = counts.join(pivot, how="left").reset_index()
    summary = summary.dropna(subset=["top", "bottom"])
    summary = summary[summary["tested_city_count"] >= min_tested_cities].copy()
    summary["gap"] = summary["top"] - summary["bottom"]
    summary["midpoint"] = (summary["top"] + summary["bottom"]) / 2
    return summary


def build_country_membership_summary(data, min_tested_cities):
    grouped = (
        data.groupby(["country", "quantile_group"], as_index=False)
        .agg(
            tested_city_count=("tested_city_count", "max"),
            selected_city_count=("city", "nunique"),
        )
    )
    if grouped.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "tested_city_count",
                "top_count",
                "bottom_count",
                "membership",
            ]
        )

    pivot = grouped.pivot(
        index="country",
        columns="quantile_group",
        values="selected_city_count",
    ).fillna(0)
    counts = grouped.groupby("country", as_index=True).agg(
        tested_city_count=("tested_city_count", "max"),
    )
    summary = counts.join(pivot, how="left").fillna(0).reset_index()
    for column in ["top", "bottom"]:
        if column not in summary.columns:
            summary[column] = 0
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0).astype(int)

    summary = summary[summary["tested_city_count"] >= min_tested_cities].copy()
    summary = summary[(summary["top"] > 0) | (summary["bottom"] > 0)].copy()
    summary["membership"] = summary.apply(country_membership_label, axis=1)
    summary = summary.rename(columns={"top": "top_count", "bottom": "bottom_count"})
    return summary


def order_summary(summary, order_by, limit):
    if order_by == "tested_cities":
        sort_columns = ["tested_city_count", "gap", "top", "country"]
    elif order_by == "top_mean":
        sort_columns = ["top", "gap", "tested_city_count", "country"]
    else:
        sort_columns = ["gap", "tested_city_count", "top", "country"]

    ascending = [False, False, False, True]
    return summary.sort_values(sort_columns, ascending=ascending).head(limit)


def wrap_label(value, width=24):
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def country_membership_label(row):
    if int(row.get("top", 0)) > 0 and int(row.get("bottom", 0)) > 0:
        return "both"
    if int(row.get("top", 0)) > 0:
        return "top"
    return "bottom"


def normalize_country_name(value):
    return " ".join(str(value).strip().casefold().split())


def country_aliases():
    return {
        normalize_country_name("United States"): "United States of America",
        normalize_country_name("Burma"): "Myanmar",
        normalize_country_name("Congo (Kinshasa)"): "Dem. Rep. Congo",
        normalize_country_name("Korea, South"): "South Korea",
        normalize_country_name("Côte d’Ivoire"): "Côte d'Ivoire",
        normalize_country_name("CÃ´te dâ€™Ivoire"): "Côte d'Ivoire",
    }


def resolve_naturalearth_world_path():
    spec = importlib.util.find_spec("pyogrio")
    if spec is None or spec.origin is None:
        return None

    candidate = (
        Path(spec.origin).resolve().parent
        / "tests"
        / "fixtures"
        / "naturalearth_lowres"
        / "naturalearth_lowres.shp"
    )
    if candidate.exists():
        return candidate
    return None


def load_world_boundaries():
    world_path = resolve_naturalearth_world_path()
    if world_path is None:
        raise SystemExit(
            "Could not find a local world boundary dataset for the country map."
        )

    world = gpd.read_file(world_path)
    world = world[world.geometry.notna()].copy()
    world = world[~world.geometry.is_empty].copy()
    world["country_name_key"] = world["name"].map(normalize_country_name)
    return world


def blend_hex_colors(first, second):
    first = first.lstrip("#")
    second = second.lstrip("#")
    return "#{:02x}{:02x}{:02x}".format(
        *[
            int(round((int(first[index:index + 2], 16) + int(second[index:index + 2], 16)) / 2))
            for index in range(0, 6, 2)
        ]
    )


def plot_country_membership_map(data, output_path, min_tested_cities):
    membership = build_country_membership_summary(data, min_tested_cities)
    if membership.empty:
        raise SystemExit("No country rows available for the country membership map.")

    world = load_world_boundaries()
    alias_lookup = country_aliases()
    membership = membership.copy()
    membership["country_name_key"] = membership["country"].map(normalize_country_name)
    membership["world_name_key"] = membership["country_name_key"].map(
        lambda key: normalize_country_name(alias_lookup.get(key, key))
    )

    joined = world.merge(
        membership,
        left_on="country_name_key",
        right_on="world_name_key",
        how="left",
    )

    category_colors = {
        "top": TOP_COLOR,
        "bottom": BOTTOM_COLOR,
        "both": BOTH_COLOR,
    }
    joined["plot_color"] = joined["membership"].map(category_colors).fillna(BASE_WORLD_COLOR)

    missing = membership.loc[~membership["world_name_key"].isin(set(world["country_name_key"]))]
    if not missing.empty:
        missing_names = ", ".join(sorted(missing["country"].unique()))
        print(f"Could not match these countries on the world map: {missing_names}")

    fig, ax = plt.subplots(figsize=(16, 9))
    joined.plot(
        ax=ax,
        color=joined["plot_color"],
        edgecolor=BASE_EDGE_COLOR,
        linewidth=0.45,
    )

    ax.set_title("Countries Represented in Selected Top and Bottom VPI Quantile Groups")
    ax.set_axis_off()

    legend_handles = [
        Patch(facecolor=TOP_COLOR, edgecolor=BASE_EDGE_COLOR, label="Top group only"),
        Patch(facecolor=BOTTOM_COLOR, edgecolor=BASE_EDGE_COLOR, label="Bottom group only"),
        Patch(
            facecolor=BOTH_COLOR,
            edgecolor=BASE_EDGE_COLOR,
            label="Appears in both",
        ),
        Patch(facecolor=BASE_WORLD_COLOR, edgecolor=BASE_EDGE_COLOR, label="Not selected"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_country_gap(summary, output_path):
    if summary.empty:
        raise SystemExit("No country rows available for the gap plot.")

    plot_data = summary.sort_values("gap", ascending=True).reset_index(drop=True)
    height = max(6, 0.34 * len(plot_data) + 1.8)
    fig, ax = plt.subplots(figsize=(11, height))

    y = range(len(plot_data))
    ax.hlines(y, plot_data["bottom"], plot_data["top"], color=LINE_COLOR, linewidth=2)
    ax.scatter(plot_data["bottom"], y, color=BOTTOM_COLOR, s=42, label="Bottom quantile mean", zorder=3)
    ax.scatter(plot_data["top"], y, color=TOP_COLOR, s=42, label="Top quantile mean", zorder=3)

    for idx, row in plot_data.iterrows():
        ax.text(
            max(row["top"], row["bottom"]) + 0.006,
            idx,
            f"{int(row['tested_city_count'])} cities",
            va="center",
            fontsize=8,
            color="#555555",
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels([wrap_label(country) for country in plot_data["country"]], fontsize=9)
    ax.set_xlabel("Mean city VPI score")
    ax.set_title("Within-Country VPI Gap Between Top and Bottom City Quantiles")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def select_city_rows(data, selected_countries, cities_per_group):
    rows = []
    country_order = {country: idx for idx, country in enumerate(selected_countries)}
    for country in selected_countries:
        country_rows = data[data["country"] == country]
        top_rows = (
            country_rows[country_rows["quantile_group"] == "top"]
            .sort_values(["vpi_score", "total_images", "city"], ascending=[False, False, True])
            .head(cities_per_group)
        )
        bottom_rows = (
            country_rows[country_rows["quantile_group"] == "bottom"]
            .sort_values(["vpi_score", "total_images", "city"], ascending=[True, False, True])
            .head(cities_per_group)
        )
        rows.append(top_rows)
        rows.append(bottom_rows)

    selected = pd.concat(rows, ignore_index=True)
    selected["country_order"] = selected["country"].map(country_order)
    selected["group_order"] = selected["quantile_group"].map({"top": 0, "bottom": 1})
    selected = selected.sort_values(
        ["country_order", "group_order", "vpi_score", "city"],
        ascending=[True, True, False, True],
    )
    return selected


def plot_city_bars(data, summary, output_path, bar_countries, cities_per_group):
    if summary.empty:
        raise SystemExit("No country rows available for the city bar plot.")

    selected_countries = summary["country"].head(bar_countries).tolist()
    plot_data = select_city_rows(data, selected_countries, cities_per_group)
    if plot_data.empty:
        raise SystemExit("No city rows available for the city bar plot.")

    labels = [
        f"{row.city} ({row.country})"
        for row in plot_data.itertuples(index=False)
    ]
    labels = [wrap_label(label, width=34) for label in labels]
    colors = [TOP_COLOR if group == "top" else BOTTOM_COLOR for group in plot_data["quantile_group"]]

    height = max(7, 0.42 * len(plot_data) + 1.8)
    fig, ax = plt.subplots(figsize=(12, height))
    y = range(len(plot_data))
    ax.barh(y, plot_data["vpi_score"], color=colors, height=0.72)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("City VPI score")
    ax.set_title("Selected Top and Bottom Quantile Cities by Country")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=TOP_COLOR, label="Top quantile"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=BOTTOM_COLOR, label="Bottom quantile"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for idx, row in enumerate(plot_data.itertuples(index=False)):
        ax.text(
            row.vpi_score + 0.004,
            idx,
            f"{row.vpi_score:.3f}",
            va="center",
            fontsize=8,
            color=TEXT_COLOR,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    if args.top_countries < 1:
        raise SystemExit("--top-countries must be at least 1.")
    if args.bar_countries < 1:
        raise SystemExit("--bar-countries must be at least 1.")
    if args.cities_per_group < 1:
        raise SystemExit("--cities-per-group must be at least 1.")
    if args.min_tested_cities < 1:
        raise SystemExit("--min-tested-cities must be at least 1.")

    data = load_quantiles(args.input_csv)
    summary = build_country_summary(data, args.min_tested_cities)
    selected_summary = order_summary(summary, args.country_order, args.top_countries)

    plot_country_gap(selected_summary, args.gap_output)
    plot_city_bars(
        data,
        selected_summary,
        args.bars_output,
        args.bar_countries,
        args.cities_per_group,
    )
    if not args.skip_country_map:
        plot_country_membership_map(
            data,
            args.country_map_output,
            args.min_tested_cities,
        )

    print(f"Countries in gap plot: {len(selected_summary)}")
    print(f"Saved country gap plot to {args.gap_output}")
    print(f"Saved city bar plot to {args.bars_output}")
    if not args.skip_country_map:
        print(f"Saved country membership map to {args.country_map_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
