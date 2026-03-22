from src.config import ScoreConfig
from src.database import DatabaseManager, Image, Detection, OSMFeature
from collections import defaultdict
import math
from sqlalchemy import func


class Scorer:
    def __init__(self, db: DatabaseManager):
        self.severity_scores = ScoreConfig.SEVERITY_SCORES
        self.osm_severity_scores = ScoreConfig.OSM_SEVERITY_SCORES
        self.osm_weight = ScoreConfig.OSM_WEIGHT
        self.db = db

    def score_region(self, region_id, method="naive"):
        if method == "naive":
            return self._score_region_naive(region_id)
        if method == "bulk":
            scores = self.score_regions(
                [region_id], apply_image_threshold=True)
            return scores.get(region_id, 0.0)
        if method in ("bulk_osm", "bulk_with_osm"):
            scores = self.score_regions_with_osm(
                [region_id], apply_image_threshold=True)
            return scores.get(region_id, 0.0)
        return 0.0

    def score_regions(self, region_ids=None, apply_image_threshold=True):
        if not self.severity_scores:
            if region_ids is None:
                return {}
            return {region_id: 0.0 for region_id in region_ids}

        region_id_filter = self._normalize_region_ids(region_ids)
        image_count_by_region = self._fetch_image_count_by_region(
            region_id_filter)
        total_by_region = self._fetch_total_detections_by_region(
            region_id_filter)
        label_counts = self._fetch_label_counts_by_region(region_id_filter)
        target_region_ids = self._target_region_ids(
            region_id_filter, image_count_by_region, total_by_region, label_counts)

        return self._build_scores(
            target_region_ids,
            image_count_by_region,
            total_by_region,
            label_counts,
            apply_image_threshold,
        )

    def score_regions_with_osm(self, region_ids=None, apply_image_threshold=True):
        detection_scores = self.score_regions(
            region_ids=region_ids, apply_image_threshold=False)
        if not self.osm_severity_scores:
            return detection_scores

        region_id_filter = self._normalize_region_ids(region_ids)
        image_count_by_region = self._fetch_image_count_by_region(
            region_id_filter)
        osm_total_by_region = self._fetch_total_osm_features_by_region(
            region_id_filter)
        osm_type_counts = self._fetch_osm_type_counts_by_region(
            region_id_filter)
        target_region_ids = set(detection_scores.keys()) | set(
            image_count_by_region.keys()) | set(osm_total_by_region.keys()) | set(osm_type_counts.keys())

        if not target_region_ids:
            return detection_scores

        weight = min(max(self.osm_weight, 0.0), 1.0)
        final_scores = {}
        for region_id in target_region_ids:
            if apply_image_threshold and (image_count_by_region.get(region_id, 0) < ScoreConfig.IMAGES_PER_REGION_THRESHOLD) or (osm_total_by_region.get(region_id, 0) < ScoreConfig.FEATURES_PER_REGION_THRESHOLD):
                final_scores[region_id] = 0.0
                continue
            detection_score = detection_scores.get(region_id, 0.0)
            osm_score = self._compute_osm_score_for_region(
                region_id, osm_total_by_region, osm_type_counts)
            if osm_total_by_region.get(region_id, 0) == 0:
                final_scores[region_id] = detection_score
                continue
            score = ((1.0 - weight) * detection_score) + (weight * osm_score)
            final_scores[region_id] = score if math.isfinite(score) else 0.0

        return final_scores

    def _normalize_region_ids(self, region_ids):
        if region_ids is None:
            return None
        return set(region_ids)

    def _fetch_image_count_by_region(self, region_id_filter):
        query = (
            self.db.session.query(Image.region_id, func.count(Image.id))
            .group_by(Image.region_id)
        )
        if region_id_filter is not None:
            query = query.filter(Image.region_id.in_(region_id_filter))
        return {region_id: count for region_id, count in query.all()}

    def _fetch_total_detections_by_region(self, region_id_filter):
        query = (
            self.db.session.query(Image.region_id, func.count(Detection.id))
            .join(Detection, Detection.image_id == Image.id)
            .group_by(Image.region_id)
        )
        if region_id_filter is not None:
            query = query.filter(Image.region_id.in_(region_id_filter))
        return {region_id: total for region_id, total in query.all()}

    def _fetch_label_counts_by_region(self, region_id_filter):
        query = (
            self.db.session.query(
                Image.region_id, Detection.label, func.count(Detection.id))
            .join(Detection, Detection.image_id == Image.id)
            .filter(Detection.label.in_(list(self.severity_scores.keys())))
            .group_by(Image.region_id, Detection.label)
        )
        if region_id_filter is not None:
            query = query.filter(Image.region_id.in_(region_id_filter))

        label_counts = defaultdict(dict)
        for region_id, label, count in query.all():
            label_counts[region_id][label] = count
        return label_counts

    def _fetch_total_osm_features_by_region(self, region_id_filter):
        query = (
            self.db.session.query(OSMFeature.region_id,
                                  func.count(OSMFeature.id))
            .group_by(OSMFeature.region_id)
        )
        if region_id_filter is not None:
            query = query.filter(OSMFeature.region_id.in_(region_id_filter))
        return {region_id: total for region_id, total in query.all()}

    def _fetch_osm_type_counts_by_region(self, region_id_filter):
        query = (
            self.db.session.query(
                OSMFeature.region_id, OSMFeature.osm_type, func.count(OSMFeature.id))
            .filter(OSMFeature.osm_type.in_(list(self.osm_severity_scores.keys())))
            .group_by(OSMFeature.region_id, OSMFeature.osm_type)
        )
        if region_id_filter is not None:
            query = query.filter(OSMFeature.region_id.in_(region_id_filter))

        type_counts = defaultdict(dict)
        for region_id, osm_type, count in query.all():
            type_counts[region_id][osm_type] = count
        return type_counts

    def _target_region_ids(self, region_id_filter, image_count_by_region, total_by_region, label_counts):
        if region_id_filter is not None:
            return region_id_filter
        return set(image_count_by_region.keys()) | set(total_by_region.keys()) | set(label_counts.keys())

    def _build_scores(self, target_region_ids, image_count_by_region, total_by_region, label_counts, apply_image_threshold):
        scores = {}
        severity_count = len(self.severity_scores)
        for region_id in target_region_ids:
            if apply_image_threshold and image_count_by_region.get(region_id, 0) < ScoreConfig.IMAGES_PER_REGION_THRESHOLD:
                scores[region_id] = 0.0
                continue
            scores[region_id] = self._compute_score_for_region(
                region_id,
                total_by_region,
                label_counts,
                severity_count,
            )
        return scores

    def _compute_score_for_region(self, region_id, total_by_region, label_counts, severity_count):
        total = total_by_region.get(region_id, 0)
        labels = label_counts.get(region_id, {})
        if total == 0 or not labels:
            return 0.0
        ccr = len(labels) / severity_count
        sws = sum(
            self.severity_scores[label] * count for label, count in labels.items()) / total
        score = ccr * sws
        if not math.isfinite(score):
            return 0.0
        return score

    def _compute_osm_score_for_region(self, region_id, total_by_region, type_counts):
        total = total_by_region.get(region_id, 0)
        types = type_counts.get(region_id, {})
        if total == 0 or not types:
            return 0.0
        ccr = len(types) / len(self.osm_severity_scores)
        sws = sum(
            self.osm_severity_scores[osm_type] * count for osm_type, count in types.items()) / total
        score = ccr * sws
        if not math.isfinite(score):
            return 0.0
        return score

    def _score_region_naive(self, region_id):
        if not self.severity_scores:
            return 0.0

        labels = defaultdict(int)
        detections_in_region = self.db.get_detections_by_region(region_id)
        imgs_in_region = len(self.db.get_images_by_region(region_id))
        if not detections_in_region or imgs_in_region < ScoreConfig.FEATURES_PER_REGION_THRESHOLD:
            return 0.0

        for det in detections_in_region:
            label = det.label
            if label in self.severity_scores:
                labels[label] += 1

        if not labels:
            return 0.0

        ccr = len(labels)/len(self.severity_scores)
        sws = sum(self.severity_scores[label] *
                  count for label, count in labels.items())/len(detections_in_region)

        score = ccr * sws
        if not math.isfinite(score):
            return 0.0
        return score
