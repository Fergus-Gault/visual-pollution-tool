import argparse
import csv
from pathlib import Path
import sys

from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.database import DatabaseManager


def normalize(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def quantile_range_label(start_fraction, end_fraction):
    start_percent = int(round(start_fraction * 100.0))
    end_percent = int(round(end_fraction * 100.0))
    if start_percent == 0:
        return f"Bottom {end_percent}%"
    return f"{start_percent}-{end_percent}%"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="SampleImagesByScoreBand",
        description="Randomly sample images from each VPI score band using images.score.",
    )
    parser.add_argument(
        "--source",
        default="mapillary",
        help="Image source to sample from. Use an empty string to include all sources.",
    )
    parser.add_argument(
        "--samples-per-band",
        type=int,
        default=3,
        help="How many random image examples to return from each score band.",
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=10,
        help="How many positive-score quantile bands to use.",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Optional country filter applied through the parent region.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Optional city filter applied through the parent region.",
    )
    parser.add_argument(
        "--exclude-zero-score",
        action="store_true",
        help="Do not include a separate zero-score sample band.",
    )
    parser.add_argument(
        "--output",
        default="data/image_score_band_examples.csv",
        help="CSV path for the sampled examples.",
    )
    return parser.parse_args()


def fetch_zero_score_examples(db, args):
    if args.exclude_zero_score:
        return []

    query = text(
        """
        WITH zero_images AS (
            SELECT
                images.id AS image_id,
                images.region_id AS region_id,
                regions.city AS city,
                regions.country AS country,
                images.source AS source,
                images.id_from_source AS source_image_id,
                images.source_captured_at AS captured_at,
                images.url AS url,
                images.score AS score,
                ROW_NUMBER() OVER (ORDER BY random()) AS sample_rank
            FROM images
            LEFT JOIN regions ON regions.id = images.region_id
            WHERE COALESCE(images.score, 0.0) <= 0.0
              AND (:country_key = '' OR lower(COALESCE(regions.country, '')) = :country_key)
              AND (:city_key = '' OR lower(COALESCE(regions.city, '')) = :city_key)
              AND (:source_key = '' OR lower(COALESCE(images.source, '')) = :source_key)
        )
        SELECT
            image_id,
            region_id,
            city,
            country,
            source,
            source_image_id,
            captured_at,
            url,
            score
        FROM zero_images
        WHERE sample_rank <= :samples_per_band
        ORDER BY sample_rank
        """
    )
    rows = db.session.execute(
        query,
        {
            "country_key": normalize(args.country),
            "city_key": normalize(args.city),
            "source_key": normalize(args.source),
            "samples_per_band": args.samples_per_band,
        },
    ).mappings()
    return [dict(row, band_index=0, band_label="Zero score") for row in rows]


def fetch_positive_band_examples(db, args):
    query = text(
        """
        WITH positive_images AS (
            SELECT
                images.id AS image_id,
                images.region_id AS region_id,
                regions.city AS city,
                regions.country AS country,
                images.source AS source,
                images.id_from_source AS source_image_id,
                images.source_captured_at AS captured_at,
                images.url AS url,
                images.score AS score,
                NTILE(:band_count) OVER (ORDER BY images.score, images.id) AS band_index
            FROM images
            LEFT JOIN regions ON regions.id = images.region_id
            WHERE images.score > 0.0
              AND (:country_key = '' OR lower(COALESCE(regions.country, '')) = :country_key)
              AND (:city_key = '' OR lower(COALESCE(regions.city, '')) = :city_key)
              AND (:source_key = '' OR lower(COALESCE(images.source, '')) = :source_key)
        ),
        sampled AS (
            SELECT
                image_id,
                region_id,
                city,
                country,
                source,
                source_image_id,
                captured_at,
                url,
                score,
                band_index,
                ROW_NUMBER() OVER (PARTITION BY band_index ORDER BY random()) AS sample_rank
            FROM positive_images
        )
        SELECT
            image_id,
            region_id,
            city,
            country,
            source,
            source_image_id,
            captured_at,
            url,
            score,
            band_index
        FROM sampled
        WHERE sample_rank <= :samples_per_band
        ORDER BY band_index, sample_rank
        """
    )
    rows = db.session.execute(
        query,
        {
            "band_count": args.bands,
            "country_key": normalize(args.country),
            "city_key": normalize(args.city),
            "source_key": normalize(args.source),
            "samples_per_band": args.samples_per_band,
        },
    ).mappings()

    labelled_rows = []
    for row in rows:
        band_index = int(row["band_index"])
        start_fraction = float(band_index - 1) / float(args.bands)
        end_fraction = float(band_index) / float(args.bands)
        labelled_rows.append(
            dict(
                row,
                band_index=band_index,
                band_label=quantile_range_label(start_fraction, end_fraction),
            )
        )
    return labelled_rows


def main():
    args = parse_args()
    if args.samples_per_band < 1:
        raise SystemExit("--samples-per-band must be at least 1.")
    if args.bands < 2:
        raise SystemExit("--bands must be at least 2.")

    db = DatabaseManager()
    zero_rows = fetch_zero_score_examples(db, args)
    positive_rows = fetch_positive_band_examples(db, args)
    rows = zero_rows + positive_rows

    if not rows:
        raise SystemExit("No scored images matched the requested filters.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "band_index",
                "band_label",
                "image_id",
                "region_id",
                "city",
                "country",
                "source",
                "source_image_id",
                "captured_at",
                "url",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} sampled image example(s) to {output_path}")
    for row in rows:
        print(
            f"{row['band_label']}: image_id={row['image_id']} "
            f"city={row['city'] or 'Unknown'} country={row['country'] or 'Unknown'} "
            f"score={float(row['score'] or 0.0):.6f} url={row['url']}"
        )


if __name__ == "__main__":
    main()
