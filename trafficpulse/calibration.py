# -*- coding: utf-8 -*-
"""Calibration homographique : projection du plan image vers le plan routier.

L'homographie mappe 4 points du plan image (pixels) vers 4 points du plan
routier (metres). L'hypothese fondamentale est que la route est plane : si
la chaussee est en pente, la projection est biaisee.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Aire minimale (en pixels carres) en dessous de laquelle le quadrilatere
# de calibration est considere comme degenere (points confondus ou alignes).
MIN_QUAD_AREA_PX = 100.0


class CalibrationError(ValueError):
    """Erreur levee lorsque les points de calibration sont invalides."""


def _polygon_area(points: np.ndarray) -> float:
    """Calcule l'aire d'un polygone via la formule du lacet (shoelace)."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def validate_points(pts_image: np.ndarray, pts_world: np.ndarray) -> None:
    """Verifie que les points de calibration forment un quadrilatere exploitable.

    Args:
        pts_image: 4 points en pixels, shape (4, 2).
        pts_world: 4 points en metres correspondants, shape (4, 2).

    Raises:
        CalibrationError: si les points sont mal formes ou degeneres.
    """
    if pts_image.shape != (4, 2) or pts_world.shape != (4, 2):
        raise CalibrationError(
            "pts_image et pts_world doivent contenir exactement 4 points (x, y) chacun"
        )

    if _polygon_area(pts_image) < MIN_QUAD_AREA_PX:
        raise CalibrationError(
            "Le quadrilatere de calibration dans l'image est trop petit ou degenere "
            "(points confondus ou alignes). Verifiez les 4 points cliques."
        )

    if _polygon_area(pts_world) <= 0:
        raise CalibrationError(
            "Le quadrilatere monde (en metres) est degenere (aire nulle)."
        )


class HomographyCalibrator:
    """Gere la transformation de perspective image -> plan metrique.

    Tous les points doivent etre coplanaires (surface de la route).
    """

    def __init__(self, pts_image: np.ndarray, pts_world: np.ndarray):
        """
        Args:
            pts_image: 4 points en pixels, shape (4, 2).
            pts_world: 4 points correspondants en metres, shape (4, 2).

        Raises:
            CalibrationError: si les points sont invalides ou degeneres.
        """
        self.pts_image = np.asarray(pts_image, dtype=np.float32)
        self.pts_world = np.asarray(pts_world, dtype=np.float32)

        validate_points(self.pts_image, self.pts_world)

        self.H = cv2.getPerspectiveTransform(self.pts_image, self.pts_world)
        self.H_inv = cv2.getPerspectiveTransform(self.pts_world, self.pts_image)

    def image_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Projette un point image (px, py) dans le plan metrique.

        Returns:
            (x_metres, y_metres)
        """
        pt = np.array([[[px, py]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, self.H)
        return float(projected[0][0][0]), float(projected[0][0][1])

    def world_to_image(self, mx: float, my: float) -> Tuple[float, float]:
        """Projette un point metrique vers l'image (transformation inverse)."""
        pt = np.array([[[mx, my]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, self.H_inv)
        return float(projected[0][0][0]), float(projected[0][0][1])

    def draw_calibration_zone(self, frame: np.ndarray, color=(0, 255, 255),
                               thickness: int = 2) -> np.ndarray:
        """Dessine le quadrilatere de calibration sur la frame."""
        annotated = frame.copy()
        pts = self.pts_image.astype(int).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)
        for i, pt in enumerate(self.pts_image.astype(int)):
            cv2.circle(annotated, tuple(pt), 6, color, -1)
            cv2.putText(annotated, f"P{i + 1}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return annotated

    def to_dict(self) -> dict:
        """Serialise la calibration en dictionnaire (pour export JSON)."""
        return {
            "pts_image": self.pts_image.tolist(),
            "pts_world": self.pts_world.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HomographyCalibrator":
        """Reconstruit une calibration depuis un dictionnaire."""
        return cls(
            pts_image=np.array(data["pts_image"], dtype=np.float32),
            pts_world=np.array(data["pts_world"], dtype=np.float32),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Sauvegarde la calibration au format JSON."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Calibration sauvegardee dans %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HomographyCalibrator":
        """Charge une calibration depuis un fichier JSON.

        Raises:
            FileNotFoundError: si le fichier n'existe pas.
            CalibrationError: si le contenu est invalide.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier de calibration introuvable : {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except (KeyError, json.JSONDecodeError) as exc:
            raise CalibrationError(f"Fichier de calibration invalide ({path}) : {exc}") from exc
