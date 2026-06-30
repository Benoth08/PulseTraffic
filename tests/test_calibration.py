# -*- coding: utf-8 -*-
"""Tests unitaires pour la calibration homographique."""

import numpy as np
import pytest

from trafficpulse.calibration import CalibrationError, HomographyCalibrator


def test_image_to_world_round_trip():
    """Un point projete puis reprojete doit redonner les coordonnees initiales."""
    pts_image = np.float32([[300, 400], [500, 400], [550, 280], [250, 280]])
    pts_world = np.float32([[0, 0], [3.5, 0], [3.5, 10.0], [0, 10.0]])

    calibrator = HomographyCalibrator(pts_image, pts_world)

    x_m, y_m = calibrator.image_to_world(300, 400)
    assert x_m == pytest.approx(0.0, abs=1e-3)
    assert y_m == pytest.approx(0.0, abs=1e-3)

    px, py = calibrator.world_to_image(x_m, y_m)
    assert px == pytest.approx(300, abs=1e-2)
    assert py == pytest.approx(400, abs=1e-2)


def test_known_corner_projects_correctly():
    pts_image = np.float32([[0, 100], [100, 100], [100, 0], [0, 0]])
    pts_world = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])

    calibrator = HomographyCalibrator(pts_image, pts_world)

    x_m, y_m = calibrator.image_to_world(100, 100)
    assert x_m == pytest.approx(10.0, abs=1e-3)
    assert y_m == pytest.approx(0.0, abs=1e-3)


def test_degenerate_points_raise_calibration_error():
    pts_image = np.float32([[100, 100], [100, 100], [100, 100], [100, 100]])
    pts_world = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])

    with pytest.raises(CalibrationError):
        HomographyCalibrator(pts_image, pts_world)


def test_wrong_shape_raises_calibration_error():
    pts_image = np.float32([[0, 0], [1, 1], [2, 2]])
    pts_world = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])

    with pytest.raises(CalibrationError):
        HomographyCalibrator(pts_image, pts_world)


def test_save_and_load_round_trip(tmp_path):
    pts_image = np.float32([[300, 400], [500, 400], [550, 280], [250, 280]])
    pts_world = np.float32([[0, 0], [3.5, 0], [3.5, 10.0], [0, 10.0]])
    calibrator = HomographyCalibrator(pts_image, pts_world)

    path = tmp_path / "calibration.json"
    calibrator.save(path)

    reloaded = HomographyCalibrator.load(path)
    assert np.allclose(reloaded.pts_image, pts_image)
    assert np.allclose(reloaded.pts_world, pts_world)


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        HomographyCalibrator.load("ce_fichier_n_existe_pas.json")
