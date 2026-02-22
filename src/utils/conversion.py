import json
from src.config import TrainConfig


def _bbox_px_to_ls_percent(x, y, w, h, img_width, img_height):
    x_pct = (x / img_width) * 100.0
    y_pct = (y / img_height) * 100.0
    w_pct = (w / img_width) * 100.0
    h_pct = (h / img_height) * 100.0

    return x_pct, y_pct, w_pct, h_pct


def _make_rectangle_prediction(label, x_pct, y_pct, w_pct, h_pct, img_width, img_height):
    return {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
            "x": float(x_pct),
            "y": float(y_pct),
            "width": float(w_pct),
            "height": float(h_pct),
            "rectanglelabels": [label],
        },
        "original_width": int(img_width),
        "original_height": int(img_height),
        "image_rotation": 0,
    }


def get_prediction(img, pred):
    bbox_coords = json.loads(pred.bbox) if isinstance(
        pred.bbox, str) else pred.bbox
    x1, y1, x2, y2 = bbox_coords
    w, h = x2 - x1, y2 - y1

    x_pct, y_pct, w_pct, h_pct = _bbox_px_to_ls_percent(
        x1, y1, w, h, img.width, img.height)
    return _make_rectangle_prediction(pred.label, x_pct, y_pct, w_pct, h_pct, img.width, img.height)


def convert_ls_to_yolo(tasks):
    out = {}

    for task in tasks:
        image_id = task.get("image_id")
        if not image_id:
            continue

        results = task.get("results") or []
        yolo_boxes = []
        for item in results:
            if item.get("type") != "rectanglelabels":
                continue

            val = item.get("value") or {}
            labels = val.get("rectanglelabels") or []
            if not labels:
                continue

            label_name = labels[0]
            if label_name not in TrainConfig.LABELS:
                continue
            cls = int(TrainConfig.LABELS[label_name])

            x0 = float(val.get("x", 0.0)) / 100.0
            y0 = float(val.get("y", 0.0)) / 100.0
            w = float(val.get("width", 0.0)) / 100.0
            h = float(val.get("height", 0.0)) / 100.0

            x0 = min(max(x0, 0.0), 1.0)
            y0 = min(max(y0, 0.0), 1.0)
            x1 = min(max(x0 + w, 0.0), 1.0)
            y1 = min(max(y0 + h, 0.0), 1.0)

            w = max(0.0, x1 - x0)
            h = max(0.0, y1 - y0)
            if w == 0.0 or h == 0.0:
                continue

            xc = x0 + w / 2.0
            yc = y0 + h / 2.0

            yolo_boxes.append([cls, xc, yc, w, h])

        out[image_id] = yolo_boxes

    return out
