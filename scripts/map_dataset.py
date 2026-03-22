import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
from sqlalchemy import func

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database.models import Region
from src.database.database import DatabaseManager

# Load dataset
db = DatabaseManager()
iso3_counts = pd.DataFrame(
    db.session.query(
        Region.iso3.label("iso3"),
        func.count(func.distinct(Region.city)).label("city_count")
    )
    .filter(
        Region.iso3.isnot(None),
        Region.population.isnot(None),
    )
    .group_by(Region.iso3)
    .all(),
    columns=["iso3", "city_count"]
)

# Define bins
bins = [0, 5, 10, 25, 50, 100, 250, 1000]
labels = [
    "1–5",
    "5–10",
    "10–25",
    "25–50",
    "50–100",
    "100–250",
    "250+"
]

iso3_counts["category"] = pd.cut(
    iso3_counts["city_count"],
    bins=bins,
    labels=labels,
    include_lowest=False,
    ordered=True
)
iso3_counts["category"] = iso3_counts["category"].astype(object)
iso3_counts.loc[iso3_counts["city_count"] == 0, "category"] = "0"
category_labels = ["0", *labels]

# Create choropleth
fig = px.choropleth(
    iso3_counts,
    locations="iso3",
    locationmode="ISO-3",
    color="category",
    hover_name="iso3",
    hover_data={"city_count": True},
    category_orders={"category": category_labels},
    title="Number of Cities per Country (Population ≥100k)",
)

fig.update_layout(
    width=1200,
    height=700,
    geo=dict(showframe=False, showcoastlines=True),
    margin=dict(l=0, r=0, t=50, b=0),
    legend=dict(
        title="Cities per country",
        title_font_size=30,
        font=dict(size=24),
        itemsizing="constant",
        itemwidth=100
    )
)
fig.show()

# Optional save
fig.write_html("./maps/cities_over_100k_binned_map.html")
