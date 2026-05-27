import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TOP_COLOR = "#c43c39"
BOTTOM_COLOR = "#2b8a67"
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

    print(f"Countries in gap plot: {len(selected_summary)}")
    print(f"Saved country gap plot to {args.gap_output}")
    print(f"Saved city bar plot to {args.bars_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
