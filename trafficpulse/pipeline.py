# -*- coding: utf-8 -*-
"""Pipeline complet : detection, suivi et estimation de vitesse sur une video."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from trafficpulse.annotation import VEHICLE_CLASS_IDS, VEHICLE_CLASSES, annotate_frame
from trafficpulse.calibration import HomographyCalibrator
from trafficpulse.config import PipelineConfig
from trafficpulse.tracking import SpeedEstimator

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


@dataclass
class VehicleRecord:
    """Historique resume d'un vehicule suivi durant la video."""

    track_id: int
    vehicle_class: str
    first_frame: int
    last_frame: int
    speeds: List[float] = field(default_factory=list)

    @property
    def avg_speed_kmh(self) -> float:
        return round(float(np.mean(self.speeds)), 1) if self.speeds else 0.0

    @property
    def max_speed_kmh(self) -> float:
        return round(float(np.max(self.speeds)), 1) if self.speeds else 0.0


@dataclass
class PipelineStats:
    """Statistiques agregees produites par un traitement video."""

    total_frames: int
    fps: float
    unique_vehicles: int
    avg_speed_kmh: float = 0.0
    median_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    p85_speed_kmh: float = 0.0
    class_counts: Dict[str, int] = field(default_factory=dict)


def _open_video_writer(
    output_path: Path, fps: float, frame_size: Tuple[int, int], preferred_codec: str = "avc1"
) -> cv2.VideoWriter:
    """Ouvre un VideoWriter en testant plusieurs codecs jusqu'a en trouver un disponible.

    Le codec H264 (avc1) est privilegie car lisible nativement dans la
    plupart des navigateurs. Sur les systemes ou il n'est pas disponible,
    on se replie automatiquement sur mp4v puis XVID.

    Raises:
        RuntimeError: si aucun codec teste ne permet d'ouvrir le writer.
    """
    for codec in (preferred_codec, "mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        if writer.isOpened():
            if codec != preferred_codec:
                logger.warning("Codec %s indisponible, repli sur %s", preferred_codec, codec)
            return writer
        writer.release()

    raise RuntimeError(
        f"Aucun codec video disponible pour ecrire {output_path} (teste : avc1, mp4v, XVID)"
    )


class TrafficPulsePipeline:
    """Pipeline reutilisable : charge le modele une seule fois et traite des videos.

    L'instanciation charge le modele YOLO en memoire (sauf si un modele deja
    charge est fourni). La meme instance peut ensuite traiter plusieurs
    videos sans recharger le modele, ce qui est important pour une
    utilisation repetee (interface Streamlit, API, traitement par lot).
    """

    def __init__(self, config: Optional[PipelineConfig] = None, model=None):
        """
        Args:
            config: Parametres du pipeline. Valeurs par defaut si non fourni.
            model: Instance YOLO deja chargee. Permet de reutiliser un modele
                mis en cache (par exemple via st.cache_resource cote Streamlit)
                sans le recharger a chaque construction du pipeline.
        """
        self.config = config or PipelineConfig()

        if model is not None:
            self.model = model
        else:
            logger.info("Chargement du modele %s sur %s", self.config.model_path, self.config.device)
            from ultralytics import YOLO
            self.model = YOLO(self.config.model_path)

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        calibrator: HomographyCalibrator,
        progress_callback: Optional[ProgressCallback] = None,
        csv_export_path: Optional[Union[str, Path]] = None,
    ) -> PipelineStats:
        """Traite une video complete : detection, suivi, vitesse, annotation.

        Args:
            input_path: Chemin de la video source.
            output_path: Chemin de la video annotee a produire.
            calibrator: Calibration homographique a utiliser pour la projection.
            progress_callback: Fonction(frame_idx, total_frames) appelee a
                chaque frame traitee, utile pour une barre de progression.
            csv_export_path: Si fourni, exporte un recapitulatif par vehicule
                a ce chemin.

        Returns:
            Statistiques agregees du traitement.

        Raises:
            FileNotFoundError: si la video source n'existe pas.
            RuntimeError: si la video ne peut pas etre ouverte ou ecrite.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Video introuvable : {input_path}")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la video : {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info("Video : %dx%d a %.1f fps, %d frames", width, height, fps, total_frames)

        speed_estimator = SpeedEstimator(
            fps=fps,
            smoothing_window=self.config.smoothing_window,
            max_speed_kmh=self.config.max_speed_kmh,
            max_history=self.config.max_history,
        )

        vehicles: Dict[int, VehicleRecord] = {}
        class_counts: Dict[str, int] = defaultdict(int)

        writer = None
        frame_idx = 0

        try:
            writer = _open_video_writer(output_path, fps, (width, height))

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                annotated, active_ids = self._process_frame(
                    frame, frame_idx, calibrator, speed_estimator, vehicles, class_counts
                )

                speed_estimator.cleanup(active_ids)
                writer.write(annotated)

                frame_idx += 1
                if progress_callback:
                    progress_callback(frame_idx, total_frames)
                elif frame_idx % 30 == 0:
                    logger.info(
                        "Frame %d/%d (%.0f%%)", frame_idx, total_frames,
                        100 * frame_idx / max(total_frames, 1),
                    )
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        stats = self._build_stats(total_frames, fps, vehicles, class_counts)

        if csv_export_path:
            self._export_csv(vehicles, csv_export_path)

        logger.info(
            "Traitement termine : %d vehicules, vitesse moyenne %.1f km/h, V85 %.1f km/h",
            stats.unique_vehicles, stats.avg_speed_kmh, stats.p85_speed_kmh,
        )
        return stats

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        calibrator: HomographyCalibrator,
        speed_estimator: SpeedEstimator,
        vehicles: Dict[int, VehicleRecord],
        class_counts: Dict[str, int],
    ) -> Tuple[np.ndarray, Set[int]]:
        """Traite une frame unique : detection, suivi, projection, annotation."""
        results = self.model.track(
            frame,
            persist=True,
            conf=self.config.confidence,
            classes=VEHICLE_CLASS_IDS,
            device=self.config.device,
            verbose=False,
        )
        boxes = results[0].boxes
        active_ids: Set[int] = set()

        if boxes.id is None or len(boxes.id) == 0:
            return calibrator.draw_calibration_zone(frame), active_ids

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i, raw_id in enumerate(ids):
            track_id = int(raw_id)
            cls_id = int(classes[i])
            x1, y1, x2, y2 = xyxy[i]
            cls_name = VEHICLE_CLASSES.get(cls_id, "vehicle")

            active_ids.add(track_id)
            class_counts[cls_name] += 1

            # Point de contact au sol : centre de la base de la boite.
            # C'est le seul point du vehicule reellement situe dans le plan
            # de l'homographie (le centre geometrique de la boite ne l'est pas).
            cx = (x1 + x2) / 2
            cy = y2
            x_m, y_m = calibrator.image_to_world(cx, cy)
            speed_estimator.update(track_id, x_m, y_m)

            if track_id not in vehicles:
                vehicles[track_id] = VehicleRecord(
                    track_id=track_id, vehicle_class=cls_name,
                    first_frame=frame_idx, last_frame=frame_idx,
                )
            record = vehicles[track_id]
            record.last_frame = frame_idx

            speed = speed_estimator.get_speed(track_id)
            if speed > self.config.min_speed_kmh:
                record.speeds.append(speed)

        annotated = annotate_frame(frame, xyxy, ids, classes, speed_estimator.speeds, calibrator)
        return annotated, active_ids

    @staticmethod
    def _build_stats(
        total_frames: int, fps: float, vehicles: Dict[int, VehicleRecord], class_counts: Dict[str, int]
    ) -> PipelineStats:
        all_speeds = [s for record in vehicles.values() for s in record.speeds]

        stats = PipelineStats(
            total_frames=total_frames,
            fps=fps,
            unique_vehicles=len(vehicles),
            class_counts=dict(class_counts),
        )

        if all_speeds:
            arr = np.array(all_speeds)
            stats.avg_speed_kmh = round(float(np.mean(arr)), 1)
            stats.median_speed_kmh = round(float(np.median(arr)), 1)
            stats.max_speed_kmh = round(float(np.max(arr)), 1)
            stats.p85_speed_kmh = round(float(np.percentile(arr, 85)), 1)

        return stats

    @staticmethod
    def _export_csv(vehicles: Dict[int, VehicleRecord], csv_path: Union[str, Path]) -> None:
        """Exporte un recapitulatif par vehicule au format CSV."""
        csv_path = Path(csv_path)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "track_id", "classe", "frame_premiere_detection", "frame_derniere_detection",
                "vitesse_moyenne_kmh", "vitesse_max_kmh", "nb_mesures",
            ])
            for record in vehicles.values():
                writer.writerow([
                    record.track_id, record.vehicle_class, record.first_frame, record.last_frame,
                    record.avg_speed_kmh, record.max_speed_kmh, len(record.speeds),
                ])
        logger.info("Export CSV : %s", csv_path)
