import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
from sqlalchemy import func, union_all

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database.models import Region, Image, OSMFeature
from src.database.database import DatabaseManager

db = DatabaseManager()

city_image_counts = (
    db.session.query(
        Region.iso3.label("iso3"),
        Region.city.label("city"),
        func.count(Image.id).label("item_count")
    )
    .join(Image, Image.region_id == Region.id)
    .filter(
        Region.iso3.isnot(None),
        Region.city.isnot(None)
    )
    .group_by(Region.iso3, Region.city)
)

city_osm_feature_counts = (
    db.session.query(
        Region.iso3.label("iso3"),
        Region.city.label("city"),
        func.count(OSMFeature.id).label("item_count")
    )
    .join(OSMFeature, OSMFeature.region_id == Region.id)
    .filter(
        Region.iso3.isnot(None),
        Region.city.isnot(None)
    )
    .group_by(Region.iso3, Region.city)
)

unified_city_counts = union_all(
    city_image_counts.statement,
    city_osm_feature_counts.statement
).subquery("unified_city_counts")

city_combined_counts_subquery = (
    db.session.query(
        func.sum(unified_city_counts.c.item_count).label("combined_count"),
        unified_city_counts.c.iso3,
        unified_city_counts.c.city
    )
    .select_from(unified_city_counts)
    .group_by(unified_city_counts.c.iso3, unified_city_counts.c.city)
    .having(func.sum(unified_city_counts.c.item_count) >= 300)
    .subquery()
)

iso3_city_counts = pd.DataFrame(
    db.session.query(
        city_combined_counts_subquery.c.iso3,
        func.count().label("city_count")
    )
    .group_by(city_combined_counts_subquery.c.iso3)
    .all(),
    columns=["iso3", "city_count"]
)

all_iso3 = pd.DataFrame(
    db.session.query(Region.iso3)
    .filter(Region.iso3.isnot(None))
    .distinct()
    .all(),
    columns=["iso3"]
)

iso3_counts = all_iso3.merge(iso3_city_counts, on="iso3", how="left")
iso3_counts["city_count"] = iso3_counts["city_count"].fillna(0).astype(int)

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

fig = px.choropleth(
    iso3_counts,
    locations="iso3",
    locationmode="ISO-3",
    color="category",
    hover_name="iso3",
    hover_data={"city_count": True},
    category_orders={"category": category_labels},
    title="Cities per Country (At Least 300 Combined Images and OSM Features)",
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

fig.write_html("./maps/cities_with_300_combined_images_osm_binned_map.html")
