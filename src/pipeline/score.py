from src.config import ScoreConfig
from src.database import DatabaseManager, Image, Detection, OSMFeature
from collections import defaultdict
import math
from sqlalchemy import func


class Scorer:
    CCR_EXCLUDED_DETECTION_LABELS = {"road_sign"}
    CCR_EXCLUDED_OSM_TYPES = {"road_sign", "traffic_sign"}

    def __init__(self, db: DatabaseManager):
        self.severity_scores = ScoreConfig.SEVERITY_SCORES
        self.osm_severity_scores = ScoreConfig.OSM_SEVERITY_SCORES
        self.osm_weight = ScoreConfig.OSM_WEIGHT
        self.db = db

    def score_region(self, region_id, method="vpi"):
        if method == "vpi":
            scores = self.score_regions(
                [region_id], apply_image_threshold=True)
            return scores.get(region_id, 0.0)
        elif method == "osm":
            scores = self.score_regions_with_osm_only([region_id])
            return scores.get(region_id, 0.0)
        elif method == "vpi_osm":
            scores = self.score_regions_with_osm(
                [region_id], apply_image_threshold=True)
            return scores.get(region_id, 0.0)
        return 0.0

    def score_image(self, image_id):
        scores = self.score_images([image_id])
        return scores.get(image_id, 0.0)

    def score_images(self, image_ids=None):
        scores, _ = self.score_images_with_summary(image_ids=image_ids)
        return scores

    def score_images_with_summary(self, image_ids=None):
        if not self.severity_scores:
            if image_ids is None:
                return {}, {}
            return {image_id: 0.0 for image_id in image_ids}, {}

        image_id_filter = self._normalize_image_ids(image_ids)
        weighted_total_by_image, label_weights, severity_detection_counts = (
            self._fetch_image_severity_aggregates(image_id_filter)
        )
        target_image_ids = self._target_image_ids(
            image_id_filter, weighted_total_by_image, label_weights)

        scores = {}
        severity_count = self._ccr_category_count(
            self.severity_scores, self.CCR_EXCLUDED_DETECTION_LABELS)
        for image_id in target_image_ids:
            scores[image_id] = self._compute_score_for_image(
                image_id,
                weighted_total_by_image,
                label_weights,
                severity_count,
            )
        return scores, severity_detection_counts

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

    def score_regions_with_osm_only(self, region_ids=None):
        if not self.osm_severity_scores:
            if region_ids is None:
                return {}
            return {region_id: 0.0 for region_id in region_ids}

        region_id_filter = self._normalize_region_ids(region_ids)
        osm_total_by_region = self._fetch_total_osm_features_by_region(
            region_id_filter)
        osm_type_counts = self._fetch_osm_type_counts_by_region(
            region_id_filter)
        target_region_ids = self._target_osm_region_ids(
            region_id_filter, osm_total_by_region, osm_type_counts)

        scores = {}
        for region_id in target_region_ids:
            if osm_total_by_region.get(region_id, 0) < ScoreConfig.FEATURES_PER_REGION_THRESHOLD:
                scores[region_id] = 0.0
                continue
            scores[region_id] = self._compute_osm_score_for_region(
                region_id, osm_total_by_region, osm_type_counts)
        return scores

    def _normalize_region_ids(self, region_ids):
        if region_ids is None:
            return None
        return set(region_ids)

    def _normalize_image_ids(self, image_ids):
        if image_ids is None:
            return None
        return set(image_ids)

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

    def _fetch_image_severity_aggregates(self, image_id_filter):
        query = (
            self.db.session.query(
                Detection.image_id,
                Detection.label,
                func.count(Detection.id),
                func.sum(
                    func.least(
                        func.greatest(func.coalesce(Detection.confidence, 1.0), 0.0),
                        1.0,
                    )
                ),
            )
            .filter(Detection.label.in_(list(self.severity_scores.keys())))
            .group_by(Detection.image_id, Detection.label)
        )
        if image_id_filter is not None:
            query = query.filter(Detection.image_id.in_(image_id_filter))

        weighted_totals = defaultdict(float)
        label_weights = defaultdict(lambda: defaultdict(float))
        severity_detection_counts = defaultdict(int)
        for image_id, label, count, confidence_sum in query.all():
            detection_count = int(count or 0)
            bounded_confidence_sum = float(confidence_sum or 0.0)
            weighted_totals[image_id] += bounded_confidence_sum
            label_weights[image_id][label] = bounded_confidence_sum
            severity_detection_counts[image_id] += detection_count
        return (
            dict(weighted_totals),
            {image_id: dict(weights) for image_id, weights in label_weights.items()},
            dict(severity_detection_counts),
        )

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

    def _target_image_ids(self, image_id_filter, weighted_total_by_image, label_weights):
        if image_id_filter is not None:
            return image_id_filter
        return set(weighted_total_by_image.keys()) | set(label_weights.keys())

    def _target_osm_region_ids(self, region_id_filter, total_by_region, type_counts):
        if region_id_filter is not None:
            return region_id_filter
        return set(total_by_region.keys()) | set(type_counts.keys())

    def _build_scores(self, target_region_ids, image_count_by_region, total_by_region, label_counts, apply_image_threshold):
        scores = {}
        severity_count = self._ccr_category_count(
            self.severity_scores, self.CCR_EXCLUDED_DETECTION_LABELS)
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
        ccr = self._compute_ccr(
            labels.keys(), severity_count, self.CCR_EXCLUDED_DETECTION_LABELS)
        sws = sum(
            self.severity_scores[label] * count for label, count in labels.items()) / total
        score = ccr * sws
        if not math.isfinite(score):
            return 0.0
        return score

    def _compute_score_for_image(self, image_id, weighted_total_by_image, label_weights, severity_count):
        total = weighted_total_by_image.get(image_id, 0.0)
        labels = label_weights.get(image_id, {})
        if total <= 0.0 or not labels:
            return 0.0
        ccr = self._compute_ccr(
            labels.keys(), severity_count, self.CCR_EXCLUDED_DETECTION_LABELS)
        sws = sum(
            self.severity_scores[label] * weight for label, weight in labels.items()
        ) / total
        score = ccr * sws
        if not math.isfinite(score):
            return 0.0
        return score

    def _compute_osm_score_for_region(self, region_id, total_by_region, type_counts):
        total = total_by_region.get(region_id, 0)
        types = type_counts.get(region_id, {})
        if total == 0 or not types:
            return 0.0
        severity_count = self._ccr_category_count(
            self.osm_severity_scores, self.CCR_EXCLUDED_OSM_TYPES)
        ccr = self._compute_ccr(
            types.keys(), severity_count, self.CCR_EXCLUDED_OSM_TYPES)
        sws = sum(
            self.osm_severity_scores[osm_type] * count for osm_type, count in types.items()) / total
        score = ccr * sws
        if not math.isfinite(score):
            return 0.0
        return score

    def _ccr_category_count(self, severity_scores, excluded_labels):
        return len(set(severity_scores.keys()) - excluded_labels)

    def _compute_ccr(self, labels, severity_count, excluded_labels):
        if severity_count <= 0:
            return 0.0
        included_labels = set(labels) - excluded_labels
        return len(included_labels) / severity_count

