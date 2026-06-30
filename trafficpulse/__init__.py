# -*- coding: utf-8 -*-
"""TrafficPulse - detection, suivi et estimation de vitesse de vehicules.

Ce package regroupe toute la logique metier (calibration, tracking,
pipeline de traitement video) independamment de toute interface utilisateur.
Il est utilise a la fois par le CLI (cli.py) et par l'interface Streamlit
(app.py), qui restent de simples consommateurs de cette API.
"""

from trafficpulse.calibration import CalibrationError, HomographyCalibrator
from trafficpulse.config import PipelineConfig
from trafficpulse.pipeline import PipelineStats, TrafficPulsePipeline, VehicleRecord
from trafficpulse.tracking import SpeedEstimator

__all__ = [
    "CalibrationError",
    "HomographyCalibrator",
    "PipelineConfig",
    "PipelineStats",
    "SpeedEstimator",
    "TrafficPulsePipeline",
    "VehicleRecord",
]

__version__ = "2.0.0"
