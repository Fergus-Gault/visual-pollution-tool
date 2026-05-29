import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from PIL import Image as PILImage
from plotly.subplots import make_subplots
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database.models import Image, Region
from src.database.database import DatabaseManager

BINS = [0, 5, 10, 25, 50, 100, 250, 1000]
CATEGORY_LABELS = [
    "0",
    "1-5",
    "5-10",
    "10-25",
    "25-50",
    "50-100",
    "100-250",
    "250+",
]
CATEGORY_COLORS = [
    "#f3f4f6",
    "#e2e8f0",
    "#c7d2fe",
    "#93c5fd",
    "#60a5fa",
    "#2563eb",
    "#1d4ed8",
    "#172554",
]
FIGURE_WIDTH = 1800
FIGURE_HEIGHT = 660
BASE_DPI = 96


def parse_args():
    parser = argparse.ArgumentParser(
        prog="MapDataset",
        description="Create dataset coverage maps as HTML and a static image.",
    )
    parser.add_argument(
        "--html-output",
        default="./maps/city_dataset_vs_300_images_map.html",
        help="Path for the interactive HTML map.",
    )
    parser.add_argument(
        "--image-output",
        default="./maps/city_dataset_vs_300_images_map.png",
        help="Path for the static map image.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Effective DPI for the static image export.",
    )
    return parser.parse_args()


def build_step_colorscale(colors):
    if len(colors) == 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]

    colorscale = []
    last_index = len(colors) - 1
    for index, color in enumerate(colors):
        start = index / last_index
        end = (index + 1) / last_index if index < last_index else 1.0
        colorscale.append([start, color])
        colorscale.append([end, color])
    return colorscale


def categorize_country_counts(counts):
    counts = counts.copy()
    counts["city_count"] = counts["city_count"].fillna(0).astype(int)
    counts["category"] = pd.cut(
        counts["city_count"],
        bins=BINS,
        labels=CATEGORY_LABELS[1:],
        include_lowest=False,
        ordered=True,
    )
    counts["category"] = counts["category"].astype(object)
    counts.loc[counts["city_count"] == 0, "category"] = CATEGORY_LABELS[0]
    category_code_lookup = {
        label: index for index, label in enumerate(CATEGORY_LABELS)
    }
    counts["category_code"] = counts["category"].map(
        category_code_lookup).astype(int)
    return counts


def load_dataset_counts(db):
    counted = pd.DataFrame(
        db.session.query(
            Region.iso3.label("iso3"),
            func.count(func.distinct(Region.city)).label("city_count"),
        )
        .filter(
            Region.iso3.isnot(None),
            Region.population.isnot(None),
        )
        .group_by(Region.iso3)
        .all(),
        columns=["iso3", "city_count"],
    )

    all_iso3 = pd.DataFrame(
        db.session.query(Region.iso3)
        .filter(Region.iso3.isnot(None))
        .distinct()
        .all(),
        columns=["iso3"],
    )
    return categorize_country_counts(
        all_iso3.merge(counted, on="iso3", how="left")
    )


def load_cities_300_image_counts(db):
    city_image_counts = (
        db.session.query(
            Region.iso3.label("iso3"),
            Region.city.label("city"),
            func.count(Image.id).label("image_count"),
        )
        .join(Image, Image.region_id == Region.id)
        .filter(
            Region.iso3.isnot(None),
            Region.city.isnot(None),
        )
        .group_by(Region.iso3, Region.city)
        .having(func.count(Image.id) >= 300)
        .subquery()
    )

    counted = pd.DataFrame(
        db.session.query(
            city_image_counts.c.iso3,
            func.count().label("city_count"),
        )
        .group_by(city_image_counts.c.iso3)
        .all(),
        columns=["iso3", "city_count"],
    )

    all_iso3 = pd.DataFrame(
        db.session.query(Region.iso3)
        .filter(Region.iso3.isnot(None))
        .distinct()
        .all(),
        columns=["iso3"],
    )
    return categorize_country_counts(
        all_iso3.merge(counted, on="iso3", how="left")
    )


def build_trace(dataframe, coloraxis_name):
    return go.Choropleth(
        locations=dataframe["iso3"],
        z=dataframe["category_code"],
        locationmode="ISO-3",
        customdata=dataframe[["city_count", "category"]],
        hovertemplate=(
            "ISO3=%{location}<br>"
            "City count=%{customdata[0]}<br>"
            "Band=%{customdata[1]}<extra></extra>"
        ),
        coloraxis=coloraxis_name,
        marker_line_color="#ffffff",
        marker_line_width=0.3,
    )


def add_manual_legend(fig):
    start_x = 0.18
    step_x = 0.095
    swatch_width = 0.018
    text_gap = 0.008
    title_y = -0.02
    item_y = -0.065

    fig.add_annotation(
        x=0.08,
        y=title_y,
        xref="paper",
        yref="paper",
        text="Cities per country",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=20, color="#111827", family="Arial Black"),
    )

    for index, (label, color) in enumerate(zip(CATEGORY_LABELS, CATEGORY_COLORS)):
        x0 = start_x + (index * step_x)
        x1 = x0 + swatch_width
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=x0,
            x1=x1,
            y0=item_y - 0.015,
            y1=item_y + 0.015,
            line=dict(color="#6b7280", width=0.9),
            fillcolor=color,
        )
        fig.add_annotation(
            x=x1 + text_gap,
            y=item_y,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(size=17, color="#374151", family="Arial Black"),
        )


def main():
    args = parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be greater than 0.")

    db = DatabaseManager()
    dataset_counts = load_dataset_counts(db)
    cities_300_counts = load_cities_300_image_counts(db)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
        horizontal_spacing=0.03,
    )

    fig.add_trace(build_trace(dataset_counts, "coloraxis"), row=1, col=1)
    fig.add_trace(build_trace(cities_300_counts, "coloraxis"), row=1, col=2)

    fig.update_layout(
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        margin=dict(l=0, r=40, t=36, b=95),
        coloraxis=dict(
            cmin=-0.5,
            cmax=len(CATEGORY_LABELS) - 0.5,
            colorscale=build_step_colorscale(CATEGORY_COLORS),
            showscale=False,
        ),
    )
    fig.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#4b5563",
        projection_type="equirectangular",
        bgcolor="white",
    )
    add_manual_legend(fig)

    fig.show()

    html_output_path = Path(args.html_output)
    image_output_path = Path(args.image_output)
    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    image_output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(html_output_path)
    fig.write_image(
        image_output_path,
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        scale=args.dpi / BASE_DPI,
    )
    with PILImage.open(image_output_path) as image:
        image.save(image_output_path, dpi=(args.dpi, args.dpi))


if __name__ == "__main__":
    main()
