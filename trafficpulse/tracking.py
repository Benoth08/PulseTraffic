# -*- coding: utf-8 -*-
"""Estimation de la vitesse des vehicules a partir de l'historique de positions."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SpeedEstimator:
    """Calcule la vitesse de chaque vehicule suivi a partir de ses positions metriques.

    Pour chaque identifiant de suivi, les positions metriques successives sont
    conservees. La vitesse est obtenue par difference finie lissee sur une
    fenetre de N positions :

        V(t) = ||P(t) - P(t-(N-1))|| / ((N-1) / fps)

    Sur N positions consecutives (une par frame), l'ecart entre la plus
    recente et la plus ancienne couvre N-1 intervalles de temps, pas N : le
    denominateur doit donc utiliser (N-1) et non N pour ne pas biaiser la
    vitesse vers le bas. Ce lissage attenue le bruit introduit par le
    tracker et par la projection homographique.
    """

    def __init__(self, fps: float, smoothing_window: int = 5,
                 max_speed_kmh: float = 200.0, max_history: int = 60):
        """
        Args:
            fps: Images par seconde de la video.
            smoothing_window: Nombre de positions utilisees pour le lissage
                de vitesse. Doit etre superieur ou egal a 2.
            max_speed_kmh: Seuil de vitesse aberrante (filtrage).
            max_history: Nombre max de positions stockees par vehicule.
        """
        if fps <= 0:
            raise ValueError(f"fps doit etre strictement positif, recu {fps}")
        if smoothing_window < 2:
            raise ValueError(f"smoothing_window doit etre >= 2, recu {smoothing_window}")

        self.fps = fps
        self.smoothing_window = smoothing_window
        self.max_speed_kmh = max_speed_kmh
        self.max_history = max_history
        self.positions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.speeds: Dict[int, float] = defaultdict(float)

    def update(self, track_id: int, x_metres: float, y_metres: float) -> None:
        """Enregistre une nouvelle position et met a jour la vitesse estimee."""
        history = self.positions[track_id]
        history.append((x_metres, y_metres))

        if len(history) > self.max_history:
            del history[: len(history) - self.max_history]

        n = self.smoothing_window
        if len(history) < n:
            return

        p_now = np.array(history[-1])
        p_prev = np.array(history[-n])
        distance = float(np.linalg.norm(p_now - p_prev))
        dt = (n - 1) / self.fps

        if dt <= 0:
            return

        speed_kmh = (distance / dt) * 3.6

        if speed_kmh <= self.max_speed_kmh:
            self.speeds[track_id] = round(speed_kmh, 1)
        else:
            logger.debug(
                "Vitesse aberrante ignoree pour le vehicule %d : %.1f km/h", track_id, speed_kmh
            )
            # On garde la derniere vitesse valide plutot que d'effacer la mesure

    def get_speed(self, track_id: int) -> float:
        """Retourne la derniere vitesse valide estimee pour ce vehicule."""
        return self.speeds.get(track_id, 0.0)

    def cleanup(self, active_ids: Set[int]) -> None:
        """Libere l'historique des vehicules qui ne sont plus suivis (occlusion, sortie de champ)."""
        inactive = set(self.positions.keys()) - active_ids
        for track_id in inactive:
            del self.positions[track_id]
            self.speeds.pop(track_id, None)
