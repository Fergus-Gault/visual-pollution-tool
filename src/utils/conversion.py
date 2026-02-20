import json


def _bbox_px_to_ls_percent(x, y, w, h, img_width, img_height):
    x_pct = (x / img_width) * 100.0
    y_pct = (y / img_height) * 100.0
    w_pct = (w / img_width) * 100.0
    h_pct = (h / img_height) * 100.0

    return x_pct, y_pct, w_pct, h_pct


def _make_rectangle_prediction(label, x_pct, y_pct, w_pct, h_pct):
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
        }
    }


def get_prediction(img, det):
    bbox_coords = json.loads(det.bbox) if isinstance(
        det.bbox, str) else det.bbox
    x1, y1, x2, y2 = bbox_coords
    x, y = x1, y1
    w, h = x2-x1, y2-y1

    x_pct, y_pct, w_pct, h_pct = _bbox_px_to_ls_percent(
        x, y, w, h, img.width, img.height)
    return _make_rectangle_prediction(det.label, x_pct, y_pct, w_pct, h_pct)
