
# Visual Pollution Tool

## Overview

This project provides an end-to-end pipeline for collecting, annotating, and analysing visual pollution using large-scale street-view imagery. It is designed to support dataset creation, model training, and inference for research and experimentation.

The tool supports automated data collection, visualisation, annotation via Label Studio, and training and evaluation of object detection models.

## Key Features

- Large-scale street-view image collection from Mapillary and KartaView
- OpenStreetMap data integration for city boundaries
- Interactive visualisation of images and detections using Folium
- Integrated Label Studio setup for annotation
- Dataset export and augmentation for training
- YOLO model training and evaluation
- Inference on collected imagery

## Installation

Clone the repository:

```bash
git clone https://github.com/Fergus-Gault/visual-pollution-tool.git
cd visual-pollution-tool
```

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

### Mapillary API token

Obtain an API token from the [Mapillary developer dashboard](https://www.mapillary.com/dashboard/developers).
Create a file at `./auth/.env` containing:

```bash
MAPILLARY_ACCESS_TOKEN=<your_token>
```

Alternatively, place the token in another location and update the path in `src/config/config.py`.

## Usage

### Data collection

Run:

```bash
python collect.py <file_or_city> [options]
```

### Arguments

**file_or_city**

- Either a city name (for example `edinburgh`).
- Or a path to a text file containing one city name per line.

**Options**

- `--debug` - Enter debug mode.
- `--collect-only` - Only collect imagery and metadata. Skip inference even if a model is provided.
- `--override` - Force a rescan of a city even if it is already in the database. **Warning: This will delete the current data for that city in the database.**

### Data labelling

Run:

```bash
label-studio start
```

In another terminal run:

```bash
python label.py
```

This will import 20 random images per region (configurable) to Label Studio, and attach any predictions from previously run inference.
To continue labelling just restart Label Studio.

### Training pipeline

The project provides two files for a training pipeline.

- `create_dataset.py` - This pulls annotated images from Label Studio (which must be running), and exports the bounding boxes and image paths to an .ndjson file, it also downloads the images.
- `train.py` - This takes the generated .ndjson file as an input, and then trains a YOLO model. Images are augmented using the `albumentations` library, these augmentations can be modified in `src/config/config.py`

## Typical Workflow

1. Provide a city or list of cities.
2. Collect street-view imagery and metadata.
3. Visualise coverage and detections using Folium.
4. Import data to Label Studio for annotation.
5. Export and augment labelled dataset.
6. Train YOLO models.
7. Run inference and analyse results.

## Project Status

This project is for my disseratation, and therefore is under active development.
