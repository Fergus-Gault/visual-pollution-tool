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
    # TODO: Restructure and split up
    out = {}

    for task in tasks:
        data = task.get("data", {})
        image_id = data.get("image_id")
        if image_id is None:
            continue

        annotations = task.get("annotations") or []
        if not annotations:
            out[image_id] = []
            continue

        annotations_sorted = sorted(
            annotations,
            key=lambda a: a.get("created_at") or "",
            reverse=True,
        )
        ann = annotations_sorted[0]

        yolo_lines = []

        for item in ann.get("result", []):
            if item.get("type") != "rectanglelabels":
                continue

            val = item.get("value", {})
            labels = val.get("rectanglelabels") or []
            if not labels:
                continue

            label_name = labels[0]
            if label_name not in TrainConfig.LABELS:
                continue
            cls = TrainConfig.LABELS[label_name]

            x0 = float(val.get("x", 0.0)) / 100.0
            y0 = float(val.get("y", 0.0)) / 100.0
            w = float(val.get("width", 0.0)) / 100.0
            h = float(val.get("height", 0.0)) / 100.0

            xc = x0 + w / 2.0
            yc = y0 + h / 2.0
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        out[image_id] = yolo_lines

    return out
