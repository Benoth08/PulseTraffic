# -*- coding: utf-8 -*-
"""Configuration du pipeline de traitement TrafficPulse."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def detect_device(preferred: Optional[str] = None) -> str:
    """Determine le device a utiliser pour l'inference.

    Args:
        preferred: Device impose explicitement ("cuda" ou "cpu"). Si None,
            le device est detecte automatiquement.

    Returns:
        "cuda" si un GPU est disponible et qu'aucun device n'est impose,
        sinon "cpu".
    """
    if preferred is not None:
        return preferred

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        logger.warning("torch n'est pas installe, utilisation du cpu")
        return "cpu"


@dataclass
class PipelineConfig:
    """Parametres du pipeline de detection, suivi et estimation de vitesse.

    Attributes:
        model_path: Chemin ou nom du poids YOLO a charger.
        device: "cuda" ou "cpu". Si None, detecte automatiquement a la
            construction de l'objet.
        confidence: Seuil de confiance de detection (entre 0 et 1).
        smoothing_window: Nombre de frames utilisees pour lisser la vitesse.
            Doit etre superieur ou egal a 2.
        max_speed_kmh: Vitesse au-dela de laquelle une mesure est consideree
            comme aberrante et ignoree (ex : suite a un changement d'identifiant
            de suivi).
        min_speed_kmh: Vitesse en dessous de laquelle un vehicule est
            considere comme a l'arret et exclu des statistiques de trafic.
        max_history: Nombre maximum de positions conservees par vehicule.
    """

    model_path: str = "yolo11n.pt"
    device: Optional[str] = None
    confidence: float = 0.3
    smoothing_window: int = 5
    max_speed_kmh: float = 200.0
    min_speed_kmh: float = 5.0
    max_history: int = 60

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(f"confidence doit etre dans (0, 1], recu {self.confidence}")
        if self.smoothing_window < 2:
            raise ValueError(
                f"smoothing_window doit etre >= 2 (besoin d'au moins 2 positions "
                f"pour calculer une vitesse), recu {self.smoothing_window}"
            )
        if self.max_speed_kmh <= 0:
            raise ValueError(f"max_speed_kmh doit etre positif, recu {self.max_speed_kmh}")
        if self.min_speed_kmh < 0:
            raise ValueError(f"min_speed_kmh doit etre >= 0, recu {self.min_speed_kmh}")
        if self.max_history < self.smoothing_window:
            raise ValueError("max_history doit etre superieur ou egal a smoothing_window")

        self.device = detect_device(self.device)
