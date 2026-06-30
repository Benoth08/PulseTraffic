# -*- coding: utf-8 -*-
"""Point d'entree en ligne de commande pour TrafficPulse.

Usage :
    python cli.py --input traffic_video.mp4 --output result.mp4 --calib-file calibration.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from trafficpulse.calibration import CalibrationError, HomographyCalibrator
from trafficpulse.config import PipelineConfig
from trafficpulse.logging_utils import setup_logging
from trafficpulse.pipeline import TrafficPulsePipeline

logger = logging.getLogger(__name__)

# Points de calibration par defaut, fournis a titre d'exemple uniquement.
# Ils doivent etre adaptes a chaque video via --calib-file (voir calibration_tool.py).
DEFAULT_PTS_IMAGE = np.float32([[300, 400], [500, 400], [550, 280], [250, 280]])
DEFAULT_PTS_WORLD = np.float32([[0, 0], [3.5, 0], [3.5, 10.0], [0, 10.0]])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TrafficPulse : detection, suivi et estimation de vitesse de vehicules."
    )
    parser.add_argument("--input", type=Path, default=Path("traffic_video.mp4"),
                         help="Video source (mp4, avi, mov)")
    parser.add_argument("--output", type=Path, default=Path("output_traffic.mp4"),
                         help="Video annotee de sortie")
    parser.add_argument("--calib-file", type=Path, default=None,
                         help="Fichier JSON de calibration (genere par app.py ou calibration_tool.py)")
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
    return parser


def load_calibration(calib_file: Optional[Path]) -> HomographyCalibrator:
    """Charge la calibration depuis un fichier JSON, ou utilise les valeurs par defaut."""
    if calib_file is not None:
        return HomographyCalibrator.load(calib_file)

    logger.warning(
        "Aucun fichier de calibration fourni (--calib-file), utilisation de points par defaut. "
        "Ces valeurs doivent etre adaptees a votre video, voir calibration_tool.py."
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
        calibrator = load_calibration(args.calib_file)
    except (CalibrationError, FileNotFoundError) as exc:
        logger.error("Calibration invalide : %s", exc)
        return 1

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
