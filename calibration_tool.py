# -*- coding: utf-8 -*-
"""Outil interactif de selection des points de calibration.

A executer en local, sur un poste disposant d'un affichage graphique.
Non utilisable dans un environnement sans interface (serveur distant,
notebook Colab, conteneur headless).

Usage :
    python calibration_tool.py chemin/vers/video.mp4 [calibration.json]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from trafficpulse.calibration import CalibrationError, HomographyCalibrator


def select_image_points(video_path: str) -> List[Tuple[int, int]]:
    """Affiche la premiere frame de la video et recupere 4 points cliques.

    L'ordre de clic attendu est : bas-gauche, bas-droite, haut-droite, haut-gauche.
    """
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Impossible de lire la video : {video_path}")

    points: List[Tuple[int, int]] = []

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
            return
        points.append((x, y))
        print(f"Point {len(points)} : ({x}, {y})")
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(frame, f"P{len(points)}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Calibration", frame)

        if len(points) == 4:
            pts = np.array(points, dtype=int).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            cv2.imshow("Calibration", frame)
            print("4 points selectionnes, appuyez sur une touche pour fermer.")

    print("Cliquez sur 4 points au sol, dans l'ordre : bas-gauche, bas-droite, haut-droite, haut-gauche.")
    cv2.imshow("Calibration", frame)
    cv2.setMouseCallback("Calibration", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points


def prompt_world_points() -> List[Tuple[float, float]]:
    """Demande a l'utilisateur les 4 points monde correspondants, en metres."""
    raw = input(
        "Entrez les 4 points monde en metres, format x1,y1 x2,y2 x3,y3 x4,y4 "
        "(Entree pour ignorer) : "
    ).strip()

    if not raw:
        return []

    return [tuple(map(float, p.split(","))) for p in raw.split()]


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage : python calibration_tool.py chemin/vers/video.mp4 [calibration.json]")
        return 1

    video_path = sys.argv[1]
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("calibration.json")

    image_points = select_image_points(video_path)
    if len(image_points) != 4:
        print("Calibration annulee : 4 points sont requis.")
        return 1

    print("\npts_image =", image_points)

    world_points = prompt_world_points()
    if not world_points:
        print("Aucun point monde fourni, calibration non sauvegardee.")
        return 0

    try:
        calibrator = HomographyCalibrator(
            np.array(image_points, dtype=np.float32),
            np.array(world_points, dtype=np.float32),
        )
    except CalibrationError as exc:
        print(f"Calibration invalide : {exc}")
        return 1

    calibrator.save(output_path)
    print(f"Calibration sauvegardee dans {output_path}")
    print("Utilisez --calib-file pour la reutiliser avec cli.py, ou chargez ce fichier dans app.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
