# -*- coding: utf-8 -*-
"""Point d'entree en ligne de commande pour TrafficPulse.

Calibration manuelle (par defaut, ou via un fichier existant) :
    python cli.py --input traffic_video.mp4 --output result.mp4 --calib-file calibration.json

Calibration automatique (par appariement de points d'interet, voir
trafficpulse.auto_calibration) :
    python cli.py --input traffic_video.mp4 --output result.mp4 \\
        --auto-calib --reference-image plan_site.png \\
        --ref-point1 120,80 --ref-point2 420,80 --ref-distance 14.0 \\
        --save-calib calibration_auto.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from trafficpulse.auto_calibration import AutoCalibrationError, AutoHomographyCalibrator, ReferenceScale
from trafficpulse.calibration import BaseCalibrator, CalibrationError, HomographyCalibrator
from trafficpulse.calibration import load_calibration as load_calibration_file
from trafficpulse.config import PipelineConfig
from trafficpulse.logging_utils import setup_logging
from trafficpulse.pipeline import TrafficPulsePipeline

logger = logging.getLogger(__name__)

# Points de calibration manuelle par defaut, fournis a titre d'exemple uniquement.
# Ils doivent etre adaptes a chaque video (voir calibration_tool.py ou --auto-calib).
DEFAULT_PTS_IMAGE = np.float32([[300, 400], [500, 400], [550, 280], [250, 280]])
DEFAULT_PTS_WORLD = np.float32([[0, 0], [3.5, 0], [3.5, 10.0], [0, 10.0]])


def _parse_point(value: str) -> Tuple[float, float]:
    """Parse une chaine 'x,y' en tuple de flottants, pour les arguments CLI."""
    try:
        x_str, y_str = value.split(",")
        return float(x_str), float(y_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Point invalide '{value}', format attendu : x,y") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TrafficPulse : detection, suivi et estimation de vitesse de vehicules."
    )
    parser.add_argument("--input", type=Path, default=Path("traffic_video.mp4"),
                         help="Video source (mp4, avi, mov)")
    parser.add_argument("--output", type=Path, default=Path("output_traffic.mp4"),
                         help="Video annotee de sortie")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                         help="Poids YOLO a utiliser")
    parser.add_argument("--confidence", type=float, default=0.3,
                         help="Seuil de confiance de detection")
    parser.add_argument("--smoothing-window", type=int, default=5,
                         help="Fenetre de lissage de la vitesse, en nombre de positions")
    parser.add_argument("--max-speed", type=float, default=200.0,
                         help="Vitesse maximale plausible en km/h, au-dela consideree comme aberrante")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                         help="Device d'inference, detecte automatiquement si non precise")
    parser.add_argument("--csv-export", type=Path, default=None,
                         help="Chemin d'un fichier CSV recapitulatif par vehicule")
    parser.add_argument("--log-level", type=str, default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    calib_group = parser.add_argument_group("Calibration manuelle")
    calib_group.add_argument("--calib-file", type=Path, default=None,
                              help="Charge une calibration existante (manuelle ou automatique, JSON)")

    auto_group = parser.add_argument_group("Calibration automatique (SIFT)")
    auto_group.add_argument("--auto-calib", action="store_true",
                             help="Calibre automatiquement par appariement de points avec une image de reference")
    auto_group.add_argument("--reference-image", type=Path, default=None,
                             help="Image de reference du site, a l'echelle connue (orthophoto, plan)")
    auto_group.add_argument("--ref-point1", type=_parse_point, default=None,
                             help="Premier point de reference en pixels dans l'image de reference, format x,y")
    auto_group.add_argument("--ref-point2", type=_parse_point, default=None,
                             help="Second point de reference en pixels dans l'image de reference, format x,y")
    auto_group.add_argument("--ref-distance", type=float, default=None,
                             help="Distance reelle en metres entre --ref-point1 et --ref-point2")
    auto_group.add_argument("--ref-origin", type=_parse_point, default=(0.0, 0.0),
                             help="Pixel de l'image de reference correspondant au point (0,0) du repere "
                                  "metrique, format x,y. Defaut : 0,0")
    auto_group.add_argument("--min-matches", type=int, default=10,
                             help="Nombre minimal de correspondances fiables requises")
    auto_group.add_argument("--ratio-threshold", type=float, default=0.75,
                             help="Seuil du test du rapport de Lowe pour le filtrage des correspondances")
    auto_group.add_argument("--ransac-threshold", type=float, default=5.0,
                             help="Tolerance de reprojection (pixels) pour l'estimation RANSAC")
    auto_group.add_argument("--save-calib", type=Path, default=None,
                             help="Sauvegarde la calibration calculee a ce chemin, pour la reutiliser ensuite")

    return parser


def _read_first_frame(video_path: Path) -> np.ndarray:
    """Lit la premiere frame d'une video, utilisee comme image source de calibration."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Impossible de lire la premiere frame de {video_path}")

    return frame


def resolve_calibration(args: argparse.Namespace) -> BaseCalibrator:
    """Determine la calibration a utiliser, selon les arguments fournis.

    Ordre de priorite :
        1. --calib-file : charge une calibration existante (manuelle ou automatique)
        2. --auto-calib : calcule une calibration automatique a partir d'une image
           de reference et de deux points de mise a l'echelle
        3. valeurs manuelles par defaut, a titre d'exemple uniquement
    """
    if args.calib_file is not None:
        return load_calibration_file(args.calib_file)

    if args.auto_calib:
        missing = [
            name for name, value in (
                ("--reference-image", args.reference_image),
                ("--ref-point1", args.ref_point1),
                ("--ref-point2", args.ref_point2),
                ("--ref-distance", args.ref_distance),
            ) if value is None
        ]
        if missing:
            raise CalibrationError(
                f"--auto-calib requiert egalement : {', '.join(missing)}"
            )

        reference_image = cv2.imread(str(args.reference_image))
        if reference_image is None:
            raise CalibrationError(f"Impossible de lire l'image de reference : {args.reference_image}")

        camera_frame = _read_first_frame(args.input)

        scale = ReferenceScale.from_two_points(
            args.ref_point1, args.ref_point2, args.ref_distance, origin_px=args.ref_origin
        )

        calibrator = AutoHomographyCalibrator.from_images(
            camera_frame, reference_image, scale,
            min_match_count=args.min_matches,
            ratio_threshold=args.ratio_threshold,
            ransac_threshold=args.ransac_threshold,
        )
        logger.info(
            "Calibration automatique : %d inliers / %d correspondances (%.0f%%)",
            calibrator.num_inliers, calibrator.num_matches, calibrator.inlier_ratio * 100,
        )
        return calibrator

    logger.warning(
        "Aucune calibration fournie (--calib-file ou --auto-calib), utilisation de points "
        "manuels par defaut. Ces valeurs doivent etre adaptees a votre video."
    )
    return HomographyCalibrator(DEFAULT_PTS_IMAGE, DEFAULT_PTS_WORLD)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if not args.input.exists():
        logger.error("Video introuvable : %s", args.input)
        logger.info("Telechargez une video de trafic libre de droits depuis :")
        logger.info("  https://www.pexels.com/search/videos/traffic/")
        return 1

    try:
        calibrator = resolve_calibration(args)
    except (CalibrationError, AutoCalibrationError, FileNotFoundError, RuntimeError) as exc:
        logger.error("Calibration invalide : %s", exc)
        return 1

    if args.save_calib is not None:
        calibrator.save(args.save_calib)

    config = PipelineConfig(
        model_path=args.model,
        device=args.device,
        confidence=args.confidence,
        smoothing_window=args.smoothing_window,
        max_speed_kmh=args.max_speed,
    )

    pipeline = TrafficPulsePipeline(config)

    try:
        stats = pipeline.run(
            input_path=args.input,
            output_path=args.output,
            calibrator=calibrator,
            csv_export_path=args.csv_export,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Echec du traitement : %s", exc)
        return 1

    logger.info("Resultat : %s", args.output)
    logger.info("Vehicules detectes : %d", stats.unique_vehicles)
    logger.info(
        "Vitesse moyenne : %.1f km/h, mediane : %.1f km/h, V85 : %.1f km/h",
        stats.avg_speed_kmh, stats.median_speed_kmh, stats.p85_speed_kmh,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
