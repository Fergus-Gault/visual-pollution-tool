import argparse
import sys
from pathlib import Path

import folium
import pandas as pd
from branca.colormap import LinearColormap
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.api.models import BoundingBox  # noqa: E402
from src.config import DatabaseConfig, ScoreConfig  # noqa: E402
from src.utils import RegionManager  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-bin an existing country scan into a finer grid using image "
            "lat/lng coordinates, then calculate VPI scores per new cell."
        )
    )
    parser.add_argument(
        "--country",
        default="United Kingdom",
        help="Country value on the source region rows.",
    )
    parser.add_argument(
        "--source-city-prefix",
        default=None,
        help=(
            "Source region city prefix. Defaults to '<country> grid ', matching "
            "collect_country.py country scans."
        ),
    )
    parser.add_argument(
        "--subregions",
        type=int,
        default=10000,
        help="Target number of new grid cells.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=1,
        help="Minimum images required for a new grid cell to receive a VPI score.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres SQLAlchemy URL. Defaults to DATABASE_URL from auth/.env.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/united_kingdom_regrid_10000_vpi.csv"),
        help="Output CSV for regridded VPI scores.",
    )
    parser.add_argument(
        "--output-map",
        type=Path,
        default=Path("maps/united_kingdom_regrid_10000_vpi.html"),
        help="Output Folium map for regridded VPI scores.",
    )
    return parser.parse_args()


def get_database_url(database_url):
    if database_url:
        return database_url
    return DatabaseConfig.get_postgres_url()


def normalise(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def get_source_region_filter(country, source_city_prefix):
    prefix = source_city_prefix
    if prefix is None:
        prefix = f"{country} grid "
    return country, prefix


def load_source_bounds(connection, country, source_city_prefix):
    country, prefix = get_source_region_filter(country, source_city_prefix)
    query = text(
        """
        SELECT
            min(min_lng) AS min_lng,
            min(min_lat) AS min_lat,
            max(max_lng) AS max_lng,
            max(max_lat) AS max_lat,
            count(*) AS region_count
        FROM regions
        WHERE lower(coalesce(country, '')) = lower(:country)
          AND lower(coalesce(city, '')) LIKE lower(:prefix_like)
        """
    )
    row = connection.execute(
        query,
        {"country": country, "prefix_like": f"{prefix}%"},
    ).mappings().one()
    if row["region_count"] == 0:
        raise SystemExit(
            f"No source regions found for country={country!r} and city prefix={prefix!r}."
        )
    return {
        "min_lng": float(row["min_lng"]),
        "min_lat": float(row["min_lat"]),
        "max_lng": float(row["max_lng"]),
        "max_lat": float(row["max_lat"]),
        "region_count": int(row["region_count"]),
        "prefix": prefix,
    }


def load_image_detection_rows(connection, country, source_city_prefix):
    country, prefix = get_source_region_filter(country, source_city_prefix)
    query = text(
        """
        SELECT
            images.id AS image_id,
            images.lng,
            images.lat,
            detections.label
        FROM images
        JOIN regions ON regions.id = images.region_id
        LEFT JOIN detections ON detections.image_id = images.id
        WHERE lower(coalesce(regions.country, '')) = lower(:country)
          AND lower(coalesce(regions.city, '')) LIKE lower(:prefix_like)
          AND images.lng IS NOT NULL
          AND images.lat IS NOT NULL
        """
    )
    rows = pd.read_sql_query(
        query,
        connection,
        params={"country": country, "prefix_like": f"{prefix}%"},
    )
    if rows.empty:
        raise SystemExit("No geolocated images found for the selected source regions.")
    return rows


def build_connected_grid(bounds, target_subregions):
    bbox = BoundingBox(
        bounds["min_lng"],
        bounds["min_lat"],
        bounds["max_lng"],
        bounds["max_lat"],
    )
    cells = RegionManager.get_connected_grid_subregions(bbox, target_subregions)
    grid_rows = len({round(cell.min_lat, 12) for cell in cells})
    grid_cols = len({round(cell.min_lng, 12) for cell in cells})
    return cells, grid_rows, grid_cols


def assign_grid_cells(rows, bounds, target_subregions):
    rows = rows.copy()
    cells, grid_rows, grid_cols = build_connected_grid(bounds, target_subregions)
    lng_step = (bounds["max_lng"] - bounds["min_lng"]) / grid_cols
    lat_step = (bounds["max_lat"] - bounds["min_lat"]) / grid_rows

    col = ((rows["lng"] - bounds["min_lng"]) / lng_step).astype(int)
    row = ((rows["lat"] - bounds["min_lat"]) / lat_step).astype(int)
    rows["grid_col"] = col.clip(lower=0, upper=grid_cols - 1)
    rows["grid_row"] = row.clip(lower=0, upper=grid_rows - 1)
    rows["grid_id"] = (
        rows["grid_row"].astype(str).str.zfill(3)
        + "_"
        + rows["grid_col"].astype(str).str.zfill(3)
    )
    rows["min_lng"] = bounds["min_lng"] + rows["grid_col"] * lng_step
    rows["max_lng"] = rows["min_lng"] + lng_step
    rows["min_lat"] = bounds["min_lat"] + rows["grid_row"] * lat_step
    rows["max_lat"] = rows["min_lat"] + lat_step
    return rows, cells, grid_rows, grid_cols


def build_grid_cells(grid_cells, grid_cols):
    cell_rows = []
    for idx, bbox in enumerate(grid_cells):
        grid_row = idx // grid_cols
        grid_col = idx % grid_cols
        cell_rows.append(
            {
                "grid_id": f"{grid_row:03d}_{grid_col:03d}",
                "image_count": 0,
                "min_lng": bbox.min_lng,
                "min_lat": bbox.min_lat,
                "max_lng": bbox.max_lng,
                "max_lat": bbox.max_lat,
            }
        )
    return pd.DataFrame(cell_rows).set_index("grid_id")


def compute_vpi_scores(rows, cells, grid_cols, min_images):
    severity_scores = ScoreConfig.SEVERITY_SCORES
    severity_count = len(severity_scores)
    base = build_grid_cells(cells, grid_cols)
    populated = rows.groupby("grid_id").agg(
        image_count=("image_id", "nunique"),
        min_lng=("min_lng", "first"),
        min_lat=("min_lat", "first"),
        max_lng=("max_lng", "first"),
        max_lat=("max_lat", "first"),
    )
    base.update(populated)
    base["image_count"] = base["image_count"].fillna(0).astype(int)

    detected = rows[rows["label"].isin(severity_scores.keys())].copy()
    if detected.empty:
        base["detection_count"] = 0
        base["unique_detection_labels"] = 0
        base["detections_per_image"] = 0.0
        base["vpi_score"] = 0.0
        return base.reset_index()

    label_counts = (
        detected.groupby(["grid_id", "label"])
        .size()
        .rename("label_count")
        .reset_index()
    )
    totals = label_counts.groupby("grid_id").agg(
        detection_count=("label_count", "sum"),
        unique_detection_labels=("label", "nunique"),
    )
    label_counts["weighted_count"] = label_counts.apply(
        lambda row: severity_scores[row["label"]] * row["label_count"],
        axis=1,
    )
    weighted = label_counts.groupby("grid_id")["weighted_count"].sum()

    scored = base.join(totals, how="left").join(weighted, how="left")
    scored[["detection_count", "unique_detection_labels", "weighted_count"]] = scored[
        ["detection_count", "unique_detection_labels", "weighted_count"]
    ].fillna(0)
    scored["detections_per_image"] = (
        scored["detection_count"] / scored["image_count"].replace(0, pd.NA)
    ).fillna(0.0)

    ccr = scored["unique_detection_labels"] / severity_count
    sws = scored["weighted_count"] / scored["detection_count"].replace(0, pd.NA)
    scored["vpi_score"] = (ccr * sws).fillna(0.0)
    scored.loc[scored["image_count"] < min_images, "vpi_score"] = 0.0
    return scored.reset_index()


def save_map(scored, output_map):
    output_map.parent.mkdir(parents=True, exist_ok=True)
    min_lng = scored["min_lng"].min()
    min_lat = scored["min_lat"].min()
    max_lng = scored["max_lng"].max()
    max_lat = scored["max_lat"].max()
    centre = [(min_lat + max_lat) / 2, (min_lng + max_lng) / 2]
    map_obj = folium.Map(location=centre, zoom_start=6, tiles="CartoDB positron")

    positive_scores = scored[scored["vpi_score"] > 0]["vpi_score"]
    if positive_scores.empty:
        colormap = None
    else:
        vmin = float(positive_scores.min())
        vmax = float(positive_scores.quantile(0.95))
        if vmin == vmax:
            vmax = float(positive_scores.max())
        if vmin == vmax:
            vmax = vmin + 1e-9
        colormap = LinearColormap(
            colors=["#1a9850", "#fee08b", "#d73027"],
            vmin=vmin,
            vmax=vmax,
        )
        colormap.caption = "Regridded VPI score (green = low, red = high)"
        colormap.add_to(map_obj)

    for row in scored.itertuples(index=False):
        if row.image_count <= 0:
            continue
        fill_color = "#d1d5db"
        fill_opacity = 0.18
        if colormap is not None and row.vpi_score > 0:
            fill_color = colormap(row.vpi_score)
            fill_opacity = 0.62
        popup = (
            f"<b>{row.grid_id}</b><br>"
            f"VPI score: {row.vpi_score:.6f}<br>"
            f"Images: {int(row.image_count)}<br>"
            f"Detections: {int(row.detection_count)}<br>"
            f"Detections / image: {row.detections_per_image:.4f}"
        )
        folium.Rectangle(
            bounds=[[row.min_lat, row.min_lng], [row.max_lat, row.max_lng]],
            color="#374151",
            weight=0.35,
            fill=True,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            popup=folium.Popup(popup, max_width=280),
        ).add_to(map_obj)

    total_cells = len(scored)
    drawn_cells = int((scored["image_count"] > 0).sum())
    positive_cells = int((scored["vpi_score"] > 0).sum())
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: white;
        border: 1px solid #999;
        padding: 10px 12px;
        font-size: 13px;
        line-height: 1.45;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    ">
        <b>Regridded UK VPI</b><br>
        Total grid cells: {total_cells:,}<br>
        Drawn cells with images: {drawn_cells:,}<br>
        Positive VPI cells: {positive_cells:,}<br>
        Empty cells hidden: {total_cells - drawn_cells:,}
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))

    map_obj.fit_bounds([[min_lat, min_lng], [max_lat, max_lng]])
    map_obj.save(output_map)


def main():
    args = parse_args()
    engine = create_engine(get_database_url(args.database_url), poolclass=NullPool)
    with engine.connect() as connection:
        bounds = load_source_bounds(connection, args.country, args.source_city_prefix)
        rows = load_image_detection_rows(connection, args.country, args.source_city_prefix)

    assigned, cells, grid_rows, grid_cols = assign_grid_cells(
        rows,
        bounds,
        args.subregions,
    )
    scored = compute_vpi_scores(
        assigned,
        cells,
        grid_cols,
        args.min_images,
    )
    scored.insert(1, "country", args.country)
    scored.insert(2, "source_region_prefix", bounds["prefix"])
    scored.insert(3, "target_grid_rows", grid_rows)
    scored.insert(4, "target_grid_cols", grid_cols)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.sort_values("vpi_score", ascending=False).to_csv(args.output_csv, index=False)
    save_map(scored, args.output_map)

    populated_cells = int((scored["image_count"] > 0).sum())
    scored_cells = int((scored["vpi_score"] > 0).sum())
    print(f"Source regions: {bounds['region_count']}")
    print(f"Target grid dimensions: {grid_rows} rows x {grid_cols} cols = {grid_rows * grid_cols}")
    print(f"Cells with images: {populated_cells}")
    print(f"Cells with positive VPI score: {scored_cells}")
    print(f"Saved regridded VPI CSV to {args.output_csv}")
    print(f"Saved regridded VPI map to {args.output_map}")


if __name__ == "__main__":
    raise SystemExit(main())
