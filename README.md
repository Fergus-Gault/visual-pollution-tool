# Visual Pollution Tool

## Overview

This project provides an end-to-end pipeline for collecting, annotating, and analysing visual pollution using large-scale street-view imagery. It is designed to support dataset creation, model training, inference, mapping, and region-level scoring for research and experimentation.

At a high level, the workflow is:

1. Collect imagery and metadata into the database.
2. Run inference and generate maps.
3. Upload samples to Label Studio for annotation.
4. Export labelled data for downstream training.
5. Train and validate a YOLO model.
6. Score regions using model detections and optionally OSM features.

## Key Features

- Large-scale street-view image collection from Mapillary and KartaView
- OpenStreetMap data integration for city boundaries and region features
- Interactive visualisation of images and detections using Folium
- Label Studio integration for annotation
- Dataset export for training and bulk data download as tar shards
- YOLO model training, validation, and inference
- Region scoring for downstream analysis

## Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/Fergus-Gault/visual-pollution-tool.git
cd visual-pollution-tool
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `auth/.env`

The project reads configuration from `auth/.env` by default. Create that file and add the values you need:

```env
DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<database>
MAPILLARY_ACCESS_TOKEN=<your_mapillary_token>
KARTAVIEW_ACCESS_TOKEN=<optional_kartaview_token>
LABEL_STUDIO_API_KEY=<your_label_studio_api_key>
STADIA_MAPS_API=<optional_stadia_maps_key>
EXTRA_TOKEN_1=<optional_extra_mapillary_token>
EXTRA_TOKEN_2=<optional_extra_mapillary_token>
EXTRA_TOKEN_3=<optional_extra_mapillary_token>
```

Notes:

- `DATABASE_URL` is required. The application expects a PostgreSQL database.
- `MAPILLARY_ACCESS_TOKEN` is required for Mapillary collection.
- `KARTAVIEW_ACCESS_TOKEN` is optional. It is only needed if you want authenticated KartaView requests.
- `LABEL_STUDIO_API_KEY` is needed for `label.py` and `create_dataset.py`.
- `STADIA_MAPS_API` is optional. Without it, maps still use the Stadia tile URL, but adding a key is recommended.
- `EXTRA_TOKEN_1` to `EXTRA_TOKEN_3` are optional helper tokens for higher-throughput multi-process collection.

### 5. Start supporting services when needed

Some workflows depend on external services:

- PostgreSQL must be running before collection, inference, scoring, or dataset export.
- Label Studio must be running at `http://localhost:8080` before `label.py` or `create_dataset.py`.

Start Label Studio with:

```bash
label-studio start
```

## Root Directory Scripts

The repository root contains the main entry-point scripts. You usually run these directly with `python <script>.py`.

### Collection and ingestion

#### `collect.py`

Main collection entry point for one city or for a text/CSV file of places.

```bash
python collect.py <file_or_city> [options]
```

Examples:

```bash
python collect.py edinburgh --country uk
python collect.py cities.txt
```

Useful options:

- `--country` / `-c`: add a country when collecting a single city
- `--collect-only`: collect imagery and metadata but skip inference
- `--override`: rescan a city even if it already exists in the database
- `--region-method`: choose region construction, defaults to `shape`
- `--dense`: increase scan density for a single city
- `--fetch-osm`: also fetch OSM features
- `--debug`: enable debug mode

Use this when you want to populate the database for a specific city or a small list of cities.

#### `collect_worldcities.py`

Batch collection entry point for many cities from a CSV, filtered by population.

```bash
python collect_worldcities.py --file data/worldcities.csv --min-population 100000
```

The default `data/worldcities.csv` dataset comes from [SimpleMaps World Cities](https://simplemaps.com/data/world-cities).

Use this when you want a large-scale multi-city collection run rather than targeted city collection.

### Annotation and dataset creation

#### `label.py`

Uploads a random sample of database images into Label Studio, including model predictions when they exist.

```bash
python label.py
```

Use this after you have collected imagery and want to begin manual annotation.

#### `create_dataset.py`

Fetches annotated tasks from Label Studio, downloads the underlying images, converts annotations, and creates a labelled dataset export under `./datasets/`.

```bash
python create_dataset.py
```

Use this after labelling is complete and you want to export the annotated subset for downstream training work.

#### `download_data.py`

Exports the database-backed image collection into tar shards plus an `index.ndjson` manifest.

```bash
python download_data.py [--download-path <path>] [--shard-size <count>]
```

Use this when you want a portable bulk export of the collected dataset rather than only the labelled subset.

### Model execution

#### `run_inference.py`

Runs inference on all regions or a filtered city/country selection, then writes region detection maps.

```bash
python run_inference.py
python run_inference.py --city Edinburgh --country UK
```

Use this after collection or after training a new model.

#### `train.py`

Trains the YOLO model from a dataset configuration file.

```bash
python train.py --path ./data/datasets/v2/data.yaml
```

Common optional arguments include `--epochs`, `--imgsz`, `--device`, `--base-model`, `--batch`, `--workers`, `--name`, and `--wandb-project`.

Use this once you have a training config such as a `data.yaml` file prepared for the YOLO training run.

#### `validate.py`

Runs validation for a trained model on a chosen split.

```bash
python validate.py --path ./data/datasets/v2/data.yaml --model-path data/model/best.pt
```

Use this to evaluate a trained checkpoint on `test`, `val`, or another supported split.

### Analysis

#### `score_regions.py`

Computes region-level scores from detections, with an optional OSM-aware method.

```bash
python score_regions.py --method vpi
python score_regions.py --method vpi_osm --city Edinburgh --country UK
```

Use this after inference when you want comparable regional scores for analysis.

## Typical Workflows

### Collect data for a city and run inference

```bash
python collect.py edinburgh --country uk --fetch-osm
python run_inference.py --city Edinburgh --country UK
```

### Label data and export annotations

```bash
label-studio start
python label.py
python create_dataset.py
```

### Train and validate a model

```bash
python train.py --path ./data/datasets/v2/data.yaml
python validate.py --path ./data/datasets/v2/data.yaml --model-path data/model/best.pt
```

### Export the full collected dataset

```bash
python download_data.py --download-path ./exports --shard-size 10000
```

### Score regions after inference

```bash
python score_regions.py --method vpi
```

## Outputs at a Glance

Depending on the script, outputs are typically written to:

- PostgreSQL for collected metadata, regions, detections, and OSM features
- `./datasets/` for exported datasets created by `create_dataset.py` and `download_data.py`
- `./data/datasets/` for any YOLO-style training configs and prepared training assets you maintain separately
- `./data/model/` for trained model checkpoints
- `./data/` for generated scores and other exported artifacts

## Project Status

This project has been completed as part of a dissertation. Small updates may be made for maintenance or to fix issues, but no major new features are planned. The code is provided as-is for research and experimentation purposes.
