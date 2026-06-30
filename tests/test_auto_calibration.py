# -*- coding: utf-8 -*-
"""Tests unitaires pour la calibration homographique automatique (SIFT + RANSAC)."""

import cv2
import numpy as np
import pytest

from trafficpulse.auto_calibration import (
    AutoCalibrationError,
    AutoHomographyCalibrator,
    ReferenceScale,
    SiftHomographyMatcher,
)


def _make_textured_image(size: int = 400, seed: int = 42) -> np.ndarray:
    """Genere une image synthetique avec assez de texture pour SIFT.

    Un bruit pur donne des points d'interet peu stables sous transformation
    perspective ; des formes nettes (cercles, segments) sont plus realistes
    et donnent des appariements robustes, comme une scene routiere texturee.
    """
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    for _ in range(60):
        x, y = rng.integers(20, size - 20, size=2)
        radius = int(rng.integers(5, 15))
        color = tuple(int(c) for c in rng.integers(0, 180, size=3))
        cv2.circle(image, (int(x), int(y)), radius, color, -1)

    for _ in range(40):
        x1, y1, x2, y2 = rng.integers(0, size, size=4)
        color = tuple(int(c) for c in rng.integers(0, 180, size=3))
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    return image


def _make_matched_pair(size: int = 400):
    """Construit une image de reference et une image source liees par une
    homographie connue, pour verifier que l'estimation la retrouve."""
    reference = _make_textured_image(size)

    h_true = np.array([
        [1.05, 0.05, -30.0],
        [-0.03, 1.10, -20.0],
        [0.0002, 0.0001, 1.0],
    ])
    source = cv2.warpPerspective(reference, np.linalg.inv(h_true), (size, size))

    return source, reference, h_true


def test_reference_scale_from_two_points():
    scale = ReferenceScale.from_two_points((0, 0), (100, 0), real_distance_m=10.0)
    assert scale.meters_per_pixel == pytest.approx(0.1)


def test_reference_scale_round_trip_with_origin():
    scale = ReferenceScale(meters_per_pixel=0.05, origin_px=(50, 80))
    x_m, y_m = scale.pixel_to_world(150, 80)
    assert x_m == pytest.approx(5.0)
    assert y_m == pytest.approx(0.0)

    px, py = scale.world_to_pixel(x_m, y_m)
    assert px == pytest.approx(150)
    assert py == pytest.approx(80)


def test_reference_scale_rejects_identical_points():
    with pytest.raises(ValueError):
        ReferenceScale.from_two_points((10, 10), (10, 10), real_distance_m=5.0)


def test_reference_scale_rejects_non_positive_distance():
    with pytest.raises(ValueError):
        ReferenceScale.from_two_points((0, 0), (10, 0), real_distance_m=0.0)


def test_sift_matcher_recovers_known_homography():
    source, reference, h_true = _make_matched_pair()

    matcher = SiftHomographyMatcher(min_match_count=10, ratio_threshold=0.75, ransac_threshold=5.0)
    result = matcher.estimate(source, reference)

    assert result.num_inliers >= 10
    assert result.inlier_ratio > 0.5

    h_estimated = result.homography / result.homography[2, 2]
    h_expected = h_true / h_true[2, 2]
    assert h_estimated == pytest.approx(h_expected, abs=0.5)


def test_sift_matcher_raises_on_blank_images():
    blank_a = np.zeros((200, 200, 3), dtype=np.uint8)
    blank_b = np.zeros((200, 200, 3), dtype=np.uint8)

    matcher = SiftHomographyMatcher()
    with pytest.raises(AutoCalibrationError):
        matcher.estimate(blank_a, blank_b)


def test_sift_matcher_rejects_invalid_min_match_count():
    with pytest.raises(ValueError):
        SiftHomographyMatcher(min_match_count=2)


def test_auto_calibrator_from_images_and_image_to_world():
    source, reference, _ = _make_matched_pair()
    scale = ReferenceScale(meters_per_pixel=0.1, origin_px=(0, 0))

    calibrator = AutoHomographyCalibrator.from_images(source, reference, scale)

    # Le centre de l'image de reference doit se projeter sur des coordonnees
    # plausibles, du meme ordre de grandeur que la taille de la scene en metres.
    x_m, y_m = calibrator.image_to_world(200, 200)
    assert 0 <= x_m <= 40
    assert 0 <= y_m <= 40


def test_auto_calibrator_serialization_round_trip(tmp_path):
    source, reference, _ = _make_matched_pair()
    scale = ReferenceScale(meters_per_pixel=0.1, origin_px=(5, 5))

    calibrator = AutoHomographyCalibrator.from_images(source, reference, scale)

    path = tmp_path / "auto_calibration.json"
    calibrator.save(path)

    reloaded = AutoHomographyCalibrator.load(path)
    assert reloaded.reference_shape == calibrator.reference_shape
    assert reloaded.reference_scale.meters_per_pixel == pytest.approx(scale.meters_per_pixel)
    assert np.allclose(reloaded.H_img_to_ref, calibrator.H_img_to_ref, atol=1e-6)


def test_auto_calibrator_draw_calibration_zone_preserves_shape():
    source, reference, _ = _make_matched_pair()
    scale = ReferenceScale(meters_per_pixel=0.1)
    calibrator = AutoHomographyCalibrator.from_images(source, reference, scale)

    annotated = calibrator.draw_calibration_zone(source)
    assert annotated.shape == source.shape
