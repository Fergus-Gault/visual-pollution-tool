import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database.database import DatabaseManager
from src.database.models import Image, Region


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
    counts["category_code"] = counts["category"].map(category_code_lookup).astype(int)
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
    legend_x0 = 1.01
    legend_x1 = 1.03
    text_x = 1.037
    start_y = 0.84
    step_y = 0.075

    fig.add_annotation(
        x=legend_x0,
        y=start_y + 0.06,
        xref="paper",
        yref="paper",
        text="Cities per country",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=20, color="#111827", family="Arial Black"),
    )

    for index, (label, color) in enumerate(zip(CATEGORY_LABELS, CATEGORY_COLORS)):
        y_center = start_y - (index * step_y)
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=legend_x0,
            x1=legend_x1,
            y0=y_center - 0.018,
            y1=y_center + 0.018,
            line=dict(color="#6b7280", width=0.9),
            fillcolor=color,
        )
        fig.add_annotation(
            x=text_x,
            y=y_center,
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
        width=1800,
        height=660,
        margin=dict(l=0, r=280, t=36, b=0),
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
    fig.write_html("./maps/city_dataset_vs_300_images_map.html")


if __name__ == "__main__":
    main()
