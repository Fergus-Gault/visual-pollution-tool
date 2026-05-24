import argparse
import sys
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from branca.colormap import LinearColormap

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DatabaseConfig, ScoreConfig  # noqa: E402


DEFAULT_SIMD_DIR = Path(
    r"C:\Users\fergu\Downloads\simd2020_withgeog\simd2020_withgeog"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a SIMD deprivation and visual pollution overlay for one "
            "Scottish council area."
        )
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=DEFAULT_SIMD_DIR / "sc_dz_11.shp",
        help="Path to the SIMD data-zone shapefile.",
    )
    parser.add_argument(
        "--simd-csv",
        type=Path,
        default=DEFAULT_SIMD_DIR / "simd2020_withinds.csv",
        help="Path to simd2020_withinds.csv.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help=(
            "Postgres SQLAlchemy URL. Defaults to DATABASE_URL from auth/.env, "
            "matching the rest of the project."
        ),
    )
    parser.add_argument(
        "--council-area",
        default="City of Edinburgh",
        help="SIMD Council_area value to extract.",
    )
    parser.add_argument(
        "--pollution-city",
        default="Edinburgh",
        help="City value on pollution regions to use for image/detection points.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maps/Edinburgh/simd_pollution_overlays"),
        help="Directory for generated Folium HTML maps.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/edinburgh_simd_pollution_by_datazone.csv"),
        help="Output per-data-zone comparison CSV.",
    )
    return parser.parse_args()


def require_path(path, description):
    if not path.exists():
        raise SystemExit(f"{description} not found: {path}")


def load_simd_datazones(shapefile, simd_csv, council_area):
    zones = gpd.read_file(shapefile)
    indicators = pd.read_csv(simd_csv)
    merged = zones.merge(
        indicators,
        left_on="DataZone",
        right_on="Data_Zone",
        how="left",
        validate="one_to_one",
    )
    city_zones = merged[merged["Council_area"].eq(council_area)].copy()
    if city_zones.empty:
        raise SystemExit(f"No SIMD data zones found for Council_area={council_area!r}")
    city_zones["SIMD_deprivation_intensity"] = (
        101 - city_zones["SIMD_2020v2_Percentile"]
    )
    return city_zones.to_crs("EPSG:4326")


def get_database_url(database_url):
    if database_url:
        return database_url
    return DatabaseConfig.get_postgres_url()


def load_pollution_points(database_url, city):
    query = text(
        """
        SELECT
            images.id AS image_id,
            images.lat,
            images.lng,
            detections.label,
            detections.confidence
        FROM images
        JOIN regions ON regions.id = images.region_id
        LEFT JOIN detections ON detections.image_id = images.id
        WHERE lower(coalesce(regions.city, '')) = lower(?)
          AND images.lat IS NOT NULL
          AND images.lng IS NOT NULL
    """.replace("?", ":city")
    )
    engine = create_engine(database_url, poolclass=NullPool)
    with engine.connect() as connection:
        rows = pd.read_sql_query(query, connection, params={"city": city})

    if rows.empty:
        return gpd.GeoDataFrame(
            rows,
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )

    rows["confidence"] = rows["confidence"].fillna(1.0).clip(lower=0.0, upper=1.0)
    rows["severity"] = rows["label"].map(ScoreConfig.SEVERITY_SCORES).fillna(0.0)
    rows["weighted_detection"] = rows["severity"] * rows["confidence"]
    geometry = gpd.points_from_xy(rows["lng"], rows["lat"], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry=geometry, crs="EPSG:4326")


def aggregate_pollution_by_datazone(city_zones, pollution_points):
    base_columns = [
        "DataZone",
        "Name",
        "Intermediate_Zone",
        "Council_area",
        "Total_population",
        "SIMD2020v2_Rank",
        "SIMD_2020v2_Percentile",
        "SIMD2020v2_Decile",
        "SIMD2020v2_Quintile",
        "SIMD_deprivation_intensity",
    ]
    summary = city_zones[base_columns + ["geometry"]].copy()
    label_columns = [f"{label}_per_image" for label in ScoreConfig.SEVERITY_SCORES]

    if pollution_points.empty:
        summary["image_count"] = 0
        summary["detection_count"] = 0
        summary["detections_per_image"] = 0.0
        summary["vpi_score_per_region"] = 0.0
        for column in label_columns:
            summary[column] = 0.0
        return summary

    joined = gpd.sjoin(
        pollution_points,
        city_zones[["DataZone", "geometry"]],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        summary["image_count"] = 0
        summary["detection_count"] = 0
        summary["detections_per_image"] = 0.0
        summary["vpi_score_per_region"] = 0.0
        for column in label_columns:
            summary[column] = 0.0
        return summary

    grouped = joined.groupby("DataZone").agg(
        image_count=("image_id", "nunique"),
        detection_count=("label", lambda labels: labels.notna().sum()),
        vpi_weighted_sum=("weighted_detection", "sum"),
    )
    denominator = grouped["image_count"].replace(0, pd.NA)
    grouped["detections_per_image"] = (
        grouped["detection_count"] / denominator
    ).fillna(0.0)
    grouped["vpi_score_per_region"] = (
        grouped["vpi_weighted_sum"] / denominator
    ).fillna(0.0)

    label_counts = (
        joined[joined["label"].notna()]
        .pivot_table(
            index="DataZone",
            columns="label",
            values="image_id",
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns=lambda label: f"{label}_count")
    )
    grouped = grouped.join(label_counts, how="left")
    for label in ScoreConfig.SEVERITY_SCORES:
        count_column = f"{label}_count"
        rate_column = f"{label}_per_image"
        if count_column not in grouped.columns:
            grouped[count_column] = 0
        grouped[rate_column] = (
            grouped[count_column] / denominator
        ).fillna(0.0)

    summary = summary.merge(grouped, on="DataZone", how="left")
    for column in summary.columns:
        if column == "geometry":
            continue
        if (
            column in ["image_count", "detection_count", "vpi_weighted_sum"]
            or column.endswith("_count")
            or column.endswith("_per_image")
            or column == "vpi_score_per_region"
        ):
            summary[column] = summary[column].fillna(0)
    for column in label_columns:
        if column not in summary.columns:
            summary[column] = 0.0
    return summary


def blue_red_colormap(values, caption):
    values = values.dropna()
    cap_note = ""
    if values.empty:
        vmin = 0.0
        vmax = 1.0
    else:
        vmin = float(values.min())
        vmax = float(values.quantile(0.95))
        max_value = float(values.max())
        if vmax < max_value:
            cap_note = " (red capped at 95th percentile)"
        if vmin == vmax:
            vmax = max_value
            cap_note = ""
        if vmin == vmax:
            vmax = vmin + 1e-9
    colormap = LinearColormap(
        colors=["#2166ac", "#f7f7f7", "#b2182b"],
        vmin=vmin,
        vmax=vmax,
    )
    colormap.caption = f"{caption}{cap_note}"
    return colormap


def add_deprivation_layer(map_obj, summary):
    colormap = blue_red_colormap(
        summary["SIMD_deprivation_intensity"],
        "SIMD deprivation intensity: blue = low, red = high",
    )
    colormap.add_to(map_obj)

    def style(feature):
        value = feature["properties"].get("SIMD_deprivation_intensity")
        return {
            "fillColor": "#999999" if value is None else colormap(value),
            "color": "#333333",
            "weight": 0.35,
            "fillOpacity": 0.68,
        }

    folium.GeoJson(
        summary,
        name="SIMD deprivation",
        style_function=style,
        tooltip=build_tooltip(),
    ).add_to(map_obj)


def build_tooltip(extra_field=None, extra_alias=None):
    fields = [
        "DataZone",
        "Name",
        "Intermediate_Zone",
        "SIMD2020v2_Rank",
        "SIMD_2020v2_Percentile",
        "SIMD_deprivation_intensity",
        "image_count",
        "detection_count",
    ]
    aliases = [
        "Data zone",
        "Name",
        "Intermediate zone",
        "SIMD rank",
        "SIMD percentile",
        "Deprivation intensity",
        "Images",
        "Detections",
    ]
    if extra_field is not None:
        fields.append(extra_field)
        aliases.append(extra_alias or extra_field)
    return folium.GeoJsonTooltip(
        fields=fields,
        aliases=aliases,
        localize=True,
        sticky=False,
    )


def add_metric_layer(map_obj, summary, metric_column, metric_name):
    values = summary[metric_column].fillna(0)
    positive = summary[values > 0].copy()
    if positive.empty:
        return

    colormap = blue_red_colormap(
        values,
        f"{metric_name}: blue = low, red = high",
    )
    colormap.add_to(map_obj)
    centroids = positive.to_crs("EPSG:27700").centroid.to_crs("EPSG:4326")
    for (_, row), point in zip(positive.iterrows(), centroids):
        metric_value = float(row[metric_column])
        popup = (
            f"<b>{row['DataZone']}</b><br>"
            f"{row['Name']}<br>"
            f"SIMD rank: {row['SIMD2020v2_Rank']}<br>"
            f"SIMD percentile: {row['SIMD_2020v2_Percentile']}<br>"
            f"Images: {int(row['image_count'])}<br>"
            f"Detections: {int(row['detection_count'])}<br>"
            f"{metric_name}: {metric_value:.4f}"
        )
        folium.CircleMarker(
            location=[point.y, point.x],
            radius=7,
            color="#111111",
            weight=0.8,
            fill=True,
            fill_color=colormap(metric_value),
            fill_opacity=0.86,
            popup=folium.Popup(popup, max_width=320),
        ).add_to(map_obj)


def create_metric_map(summary, metric_column, metric_name, output_path, council_area):
    minx, miny, maxx, maxy = summary.total_bounds
    centre = [(miny + maxy) / 2, (minx + maxx) / 2]
    map_obj = folium.Map(location=centre, zoom_start=11, tiles="CartoDB positron")
    add_deprivation_layer(map_obj, summary)
    add_metric_layer(map_obj, summary, metric_column, metric_name)
    map_obj.fit_bounds([[miny, minx], [maxy, maxx]])
    folium.LayerControl(collapsed=False).add_to(map_obj)
    map_obj.get_root().html.add_child(
        folium.Element(
            f"<h3 align='center' style='font-size:16px'><b>{council_area}: {metric_name}</b></h3>"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(output_path)


def slugify(value):
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def save_outputs(summary, output_csv, output_dir, council_area):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_columns = [column for column in summary.columns if column != "geometry"]
    summary[csv_columns].sort_values("SIMD2020v2_Rank").to_csv(
        output_csv,
        index=False,
    )

    maps = [
        (
            "detections_per_image",
            "All detections per image",
            output_dir / "detections_per_image.html",
        ),
        (
            "vpi_score_per_region",
            "VPI score per region",
            output_dir / "vpi_score_per_region.html",
        ),
    ]
    for label in ScoreConfig.SEVERITY_SCORES:
        column = f"{label}_per_image"
        maps.append(
            (
                column,
                f"{label.replace('_', ' ').title()} detections per image",
                output_dir / f"{slugify(label)}_per_image.html",
            )
        )

    for metric_column, metric_name, output_path in maps:
        create_metric_map(
            summary,
            metric_column,
            metric_name,
            output_path,
            council_area,
        )

    return [output_path for _, _, output_path in maps]


def print_metric_explanation():
    print("Colour scale: blue = low value, red = high value on every generated map.")
    print(
        "SIMD deprivation intensity = 101 - SIMD_2020v2_Percentile, "
        "so higher/red means more deprived."
    )
    print("Detections per image = detection_count / image_count.")
    print("Per-pollutant rate = detections of that label / image_count.")
    print(
        "VPI score per region = sum(label severity score * detection confidence) "
        "/ image_count."
    )


def main():
    args = parse_args()
    require_path(args.shapefile, "SIMD shapefile")
    require_path(args.simd_csv, "SIMD indicator CSV")

    city_zones = load_simd_datazones(
        args.shapefile,
        args.simd_csv,
        args.council_area,
    )
    database_url = get_database_url(args.database_url)
    pollution_points = load_pollution_points(database_url, args.pollution_city)
    pollution_points = pollution_points[
        pollution_points.geometry.within(city_zones.union_all())
    ].copy()
    summary = aggregate_pollution_by_datazone(city_zones, pollution_points)
    output_maps = save_outputs(
        summary,
        args.output_csv,
        args.output_dir,
        args.council_area,
    )

    print_metric_explanation()
    print(f"Saved comparison CSV to {args.output_csv}")
    print(f"Saved {len(output_maps)} overlay maps to {args.output_dir}")
    print(f"SIMD data zones: {len(summary)}")
    print(f"Pollution points inside area: {pollution_points['image_id'].nunique()}")


if __name__ == "__main__":
    raise SystemExit(main())
