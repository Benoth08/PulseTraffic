# -*- coding: utf-8 -*-
"""TrafficPulse - detection, suivi et estimation de vitesse de vehicules.

Ce package regroupe toute la logique metier (calibration, tracking,
pipeline de traitement video) independamment de toute interface utilisateur.
Il est utilise a la fois par le CLI (cli.py) et par l'interface Streamlit
(app.py), qui restent de simples consommateurs de cette API.
"""

from trafficpulse.auto_calibration import (
    AutoCalibrationError,
    AutoHomographyCalibrator,
    MatchResult,
    ReferenceScale,
    SiftHomographyMatcher,
)
from trafficpulse.calibration import BaseCalibrator, CalibrationError, HomographyCalibrator, load_calibration
from trafficpulse.config import PipelineConfig
from trafficpulse.pipeline import PipelineStats, TrafficPulsePipeline, VehicleRecord
from trafficpulse.tracking import SpeedEstimator

__all__ = [
    "AutoCalibrationError",
    "AutoHomographyCalibrator",
    "BaseCalibrator",
    "CalibrationError",
    "HomographyCalibrator",
    "MatchResult",
    "PipelineConfig",
    "PipelineStats",
    "ReferenceScale",
    "SiftHomographyMatcher",
    "SpeedEstimator",
    "TrafficPulsePipeline",
    "VehicleRecord",
    "load_calibration",
]

__version__ = "2.0.0"
