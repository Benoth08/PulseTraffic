# -*- coding: utf-8 -*-
"""Tests unitaires pour l'estimateur de vitesse."""

import pytest

from trafficpulse.tracking import SpeedEstimator


def test_speed_is_zero_before_enough_history():
    estimator = SpeedEstimator(fps=30, smoothing_window=5)
    for i in range(4):
        estimator.update(track_id=1, x_metres=float(i), y_metres=0.0)
    assert estimator.get_speed(1) == 0.0


def test_known_constant_speed():
    """Un vehicule parcourant 1 metre par frame a 10 fps doit donner 36 km/h.

    10 fps et 1 m/frame -> 10 m/s -> 36 km/h. Sert aussi a verifier que le
    calcul de dt utilise bien (N-1)/fps et non N/fps.
    """
    fps = 10
    estimator = SpeedEstimator(fps=fps, smoothing_window=5)

    for i in range(6):
        estimator.update(track_id=1, x_metres=float(i), y_metres=0.0)

    assert estimator.get_speed(1) == pytest.approx(36.0, abs=0.1)


def test_aberrant_speed_is_filtered_out():
    estimator = SpeedEstimator(fps=30, smoothing_window=2, max_speed_kmh=50.0)

    estimator.update(track_id=1, x_metres=0.0, y_metres=0.0)
    estimator.update(track_id=1, x_metres=0.1, y_metres=0.0)
    first_speed = estimator.get_speed(1)

    # Saut de position irrealiste, simule un changement d'identifiant de suivi
    estimator.update(track_id=1, x_metres=100.0, y_metres=0.0)
    assert estimator.get_speed(1) == first_speed


def test_cleanup_removes_inactive_tracks():
    estimator = SpeedEstimator(fps=30, smoothing_window=2)
    estimator.update(track_id=1, x_metres=0.0, y_metres=0.0)
    estimator.update(track_id=2, x_metres=0.0, y_metres=0.0)

    estimator.cleanup(active_ids={1})

    assert 1 in estimator.positions
    assert 2 not in estimator.positions


def test_history_is_capped_to_max_history():
    estimator = SpeedEstimator(fps=30, smoothing_window=2, max_history=5)
    for i in range(10):
        estimator.update(track_id=1, x_metres=float(i), y_metres=0.0)
    assert len(estimator.positions[1]) == 5


def test_invalid_fps_raises_value_error():
    with pytest.raises(ValueError):
        SpeedEstimator(fps=0)


def test_invalid_smoothing_window_raises_value_error():
    with pytest.raises(ValueError):
        SpeedEstimator(fps=30, smoothing_window=1)
