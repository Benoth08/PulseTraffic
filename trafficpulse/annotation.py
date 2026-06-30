# -*- coding: utf-8 -*-
"""Definitions des classes vehicules et fonctions d'annotation des frames."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

# Classes COCO correspondant aux vehicules routiers (modele YOLO pre-entraine).
# YOLO connait 80 classes au total, on ne garde que celles utiles au trafic routier.
VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASS_IDS: List[int] = list(VEHICLE_CLASSES.keys())

# Couleurs d'affichage par classe (au format BGR utilise par OpenCV)
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    2: (255, 150, 50),
    3: (0, 255, 255),
    5: (0, 200, 0),
    7: (0, 100, 255),
}
DEFAULT_COLOR = (200, 200, 200)


def annotate_frame(
    frame: np.ndarray,
    boxes: np.ndarray,
    ids: np.ndarray,
    classes: np.ndarray,
    speeds: Dict[int, float],
    calibrator=None,
) -> np.ndarray:
    """Annote une frame avec les boites, identifiants et vitesses des vehicules.

    Args:
        frame: Image BGR source.
        boxes: Tableau (n, 4) de boites englobantes [x1, y1, x2, y2].
        ids: Tableau (n,) d'identifiants de suivi.
        classes: Tableau (n,) d'identifiants de classe COCO.
        speeds: dict {track_id: vitesse_kmh}.
        calibrator: Calibrateur utilise pour dessiner la zone de reference,
            si fourni.

    Returns:
        Une copie annotee de la frame.
    """
    annotated = frame.copy()

    if calibrator is not None:
        annotated = calibrator.draw_calibration_zone(annotated)

    for i, raw_id in enumerate(ids):
        track_id = int(raw_id)
        x1, y1, x2, y2 = (int(v) for v in boxes[i])
        cls_id = int(classes[i])
        speed = speeds.get(track_id, 0.0)
        cls_name = VEHICLE_CLASSES.get(cls_id, "vehicle")
        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{track_id} {cls_name}"
        if speed > 0:
            label += f" {speed:.0f}km/h"

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Point de contact au sol : centre de la base de la boite
        cx = (x1 + x2) // 2
        cv2.circle(annotated, (cx, y2), 4, (0, 0, 255), -1)

    return annotated
