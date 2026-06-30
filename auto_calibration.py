# -*- coding: utf-8 -*-
"""Calibration homographique automatique par appariement de points d'interet.

Principe : au lieu de cliquer manuellement 4 points, on fournit une image de
reference du meme site (par exemple une orthophoto ou un plan a l'echelle
connue, vu de dessus). Des points d'interet (SIFT) sont detectes et appaires
automatiquement entre la frame camera et cette image de reference, puis une
homographie robuste est estimee par RANSAC (qui rejette automatiquement les
correspondances aberrantes).

Cette homographie projette la frame camera vers l'image de reference. Elle
est ensuite combinee avec l'echelle metrique de l'image de reference
(ReferenceScale) pour obtenir directement la projection pixel camera -> metres,
au meme titre que la calibration manuelle (HomographyCalibrator).

L'hypothese reste la meme que pour la calibration manuelle : la scene
observee (la route) doit etre plane.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

from trafficpulse.calibration import BaseCalibrator, CalibrationError

logger = logging.getLogger(__name__)


class AutoCalibrationError(ValueError):
    """Erreur levee lorsque l'appariement automatique de points echoue."""


@dataclass
class ReferenceScale:
    """Echelle metrique d'une image de reference (vue de dessus du site).

    Convertit des coordonnees pixel de l'image de reference en coordonnees
    metriques, a partir d'une distance reelle connue entre deux points de
    cette image (par exemple une largeur de voie mesuree sur un plan, ou
    l'echelle d'une orthophoto).
    """

    meters_per_pixel: float
    origin_px: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.meters_per_pixel <= 0:
            raise ValueError(f"meters_per_pixel doit etre positif, recu {self.meters_per_pixel}")

    @classmethod
    def from_two_points(
        cls,
        point_a_px: Tuple[float, float],
        point_b_px: Tuple[float, float],
        real_distance_m: float,
        origin_px: Tuple[float, float] = (0.0, 0.0),
    ) -> "ReferenceScale":
        """Deduit l'echelle a partir de deux points et de leur distance reelle connue.

        Args:
            point_a_px: Premier point, en pixels, dans l'image de reference.
            point_b_px: Second point, en pixels, dans l'image de reference.
            real_distance_m: Distance reelle entre ces deux points, en metres.
            origin_px: Pixel de l'image de reference correspondant au point
                (0, 0) du repere metrique. Par defaut, le coin haut-gauche.

        Raises:
            ValueError: si les deux points sont confondus ou la distance
                fournie est nulle ou negative.
        """
        if real_distance_m <= 0:
            raise ValueError(f"real_distance_m doit etre positif, recu {real_distance_m}")

        pixel_distance = float(np.linalg.norm(np.array(point_a_px) - np.array(point_b_px)))
        if pixel_distance < 1e-6:
            raise ValueError("point_a_px et point_b_px sont confondus, impossible de deduire une echelle")

        return cls(meters_per_pixel=real_distance_m / pixel_distance, origin_px=origin_px)

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        x_m = (px - self.origin_px[0]) * self.meters_per_pixel
        y_m = (py - self.origin_px[1]) * self.meters_per_pixel
        return x_m, y_m

    def world_to_pixel(self, mx: float, my: float) -> Tuple[float, float]:
        px = mx / self.meters_per_pixel + self.origin_px[0]
        py = my / self.meters_per_pixel + self.origin_px[1]
        return px, py

    def to_dict(self) -> dict:
        return {"meters_per_pixel": self.meters_per_pixel, "origin_px": list(self.origin_px)}

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceScale":
        return cls(meters_per_pixel=data["meters_per_pixel"], origin_px=tuple(data["origin_px"]))


@dataclass
class MatchResult:
    """Resultat brut d'un appariement SIFT + RANSAC entre deux images."""

    homography: np.ndarray
    num_matches: int
    num_inliers: int

    @property
    def inlier_ratio(self) -> float:
        return self.num_inliers / self.num_matches if self.num_matches else 0.0


class SiftHomographyMatcher:
    """Estime une homographie entre deux images par appariement automatique de points.

    Etapes : detection de points d'interet (SIFT), appariement brute-force
    avec filtrage par le test du rapport de Lowe (knnMatch, k=2), puis
    estimation robuste de l'homographie par RANSAC, qui ecarte automatiquement
    les correspondances aberrantes (occlusions partielles, textures repetitives).
    """

    def __init__(self, min_match_count: int = 10, ratio_threshold: float = 0.75,
                 ransac_threshold: float = 5.0):
        """
        Args:
            min_match_count: Nombre minimal de correspondances fiables requises
                pour accepter le resultat (4 est le minimum mathematique pour
                calculer une homographie, mais une marge est preferable).
            ratio_threshold: Seuil du test du rapport de Lowe (plus bas = plus
                strict, moins de correspondances mais plus fiables).
            ransac_threshold: Tolerance de reprojection en pixels pour RANSAC.
        """
        if min_match_count < 4:
            raise ValueError("min_match_count doit etre >= 4 (minimum requis par cv2.findHomography)")

        self.min_match_count = min_match_count
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self._sift = cv2.SIFT_create()
        self._matcher = cv2.BFMatcher()

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def estimate(self, image_source: np.ndarray, image_destination: np.ndarray) -> MatchResult:
        """Estime l'homographie qui projette image_source vers image_destination.

        Args:
            image_source: Image dont on cherche a projeter les points (la
                frame camera dans le cas de TrafficPulse).
            image_destination: Image cible (le plan ou l'orthophoto de
                reference, a l'echelle connue).

        Returns:
            Le resultat de l'appariement : homographie, nombre de
            correspondances retenues, nombre d'inliers selon RANSAC.

        Raises:
            AutoCalibrationError: si pas assez de points sont detectes ou
                appaires, ou si RANSAC ne parvient pas a estimer d'homographie.
        """
        gray_source = self._to_gray(image_source)
        gray_dest = self._to_gray(image_destination)

        kp_source, des_source = self._sift.detectAndCompute(gray_source, None)
        kp_dest, des_dest = self._sift.detectAndCompute(gray_dest, None)

        if des_source is None or des_dest is None or len(kp_source) < 2 or len(kp_dest) < 2:
            raise AutoCalibrationError(
                "Pas assez de points d'interet detectes dans l'une des deux images. "
                "Verifiez qu'elles contiennent suffisamment de texture (eviter les "
                "surfaces uniformes ou trop floues)."
            )

        raw_matches = self._matcher.knnMatch(des_source, des_dest, k=2)

        good_matches = [
            m for m, n in (pair for pair in raw_matches if len(pair) == 2)
            if m.distance < self.ratio_threshold * n.distance
        ]

        logger.info(
            "Correspondances retenues apres test de Lowe : %d / %d",
            len(good_matches), len(raw_matches),
        )

        if len(good_matches) < self.min_match_count:
            raise AutoCalibrationError(
                f"Pas assez de correspondances fiables ({len(good_matches)} trouvees, "
                f"{self.min_match_count} requises). Verifiez que les deux images "
                f"representent bien la meme scene, avec un recouvrement suffisant."
            )

        src_pts = np.float32([kp_source[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_dest[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)

        if homography is None:
            raise AutoCalibrationError("Le calcul de l'homographie a echoue (RANSAC n'a pas converge)")

        num_inliers = int(mask.sum()) if mask is not None else 0
        logger.info("Homographie estimee : %d inliers / %d correspondances", num_inliers, len(good_matches))

        return MatchResult(homography=homography, num_matches=len(good_matches), num_inliers=num_inliers)


class AutoHomographyCalibrator(BaseCalibrator):
    """Calibration homographique obtenue automatiquement par appariement de points.

    Combine une homographie image camera -> image de reference (estimee par
    SiftHomographyMatcher) avec l'echelle metrique de cette image de
    reference, pour obtenir directement la projection pixel camera -> metres.
    """

    def __init__(self, homography: np.ndarray, reference_shape: Tuple[int, int],
                 reference_scale: ReferenceScale, num_matches: int = 0, num_inliers: int = 0):
        """
        Args:
            homography: Homographie image camera -> image de reference.
            reference_shape: (hauteur, largeur) de l'image de reference, pour
                pouvoir projeter son contour dans l'image camera.
            reference_scale: Echelle metrique de l'image de reference.
            num_matches: Nombre de correspondances retenues lors de l'estimation.
            num_inliers: Nombre de correspondances jugees coherentes par RANSAC.
        """
        self.H_img_to_ref = np.asarray(homography, dtype=np.float64)
        self.H_ref_to_img = np.linalg.inv(self.H_img_to_ref)
        self.reference_shape = (int(reference_shape[0]), int(reference_shape[1]))
        self.reference_scale = reference_scale
        self.num_matches = num_matches
        self.num_inliers = num_inliers

    @property
    def inlier_ratio(self) -> float:
        return self.num_inliers / self.num_matches if self.num_matches else 0.0

    @classmethod
    def from_images(
        cls,
        camera_frame: np.ndarray,
        reference_image: np.ndarray,
        reference_scale: ReferenceScale,
        min_match_count: int = 10,
        ratio_threshold: float = 0.75,
        ransac_threshold: float = 5.0,
    ) -> "AutoHomographyCalibrator":
        """Calibre automatiquement a partir d'une frame camera et d'une image de reference.

        Args:
            camera_frame: Une frame de la video a calibrer (typiquement la
                premiere).
            reference_image: Image de reference du meme site, a l'echelle
                connue (orthophoto, plan).
            reference_scale: Echelle metrique de l'image de reference.
            min_match_count: Voir SiftHomographyMatcher.
            ratio_threshold: Voir SiftHomographyMatcher.
            ransac_threshold: Voir SiftHomographyMatcher.

        Raises:
            AutoCalibrationError: si l'appariement automatique echoue.
        """
        matcher = SiftHomographyMatcher(min_match_count, ratio_threshold, ransac_threshold)
        result = matcher.estimate(camera_frame, reference_image)

        if result.inlier_ratio < 0.5:
            logger.warning(
                "Faible taux d'inliers (%.0f%%), la calibration automatique peut etre peu "
                "fiable. Verifiez l'image de reference ou repassez en calibration manuelle.",
                result.inlier_ratio * 100,
            )

        return cls(
            homography=result.homography,
            reference_shape=reference_image.shape[:2],
            reference_scale=reference_scale,
            num_matches=result.num_matches,
            num_inliers=result.num_inliers,
        )

    def image_to_world(self, px: float, py: float) -> Tuple[float, float]:
        pt = np.array([[[px, py]]], dtype=np.float32)
        ref_pt = cv2.perspectiveTransform(pt, self.H_img_to_ref.astype(np.float32))
        ref_x, ref_y = float(ref_pt[0][0][0]), float(ref_pt[0][0][1])
        return self.reference_scale.pixel_to_world(ref_x, ref_y)

    def world_to_image(self, mx: float, my: float) -> Tuple[float, float]:
        ref_px, ref_py = self.reference_scale.world_to_pixel(mx, my)
        pt = np.array([[[ref_px, ref_py]]], dtype=np.float32)
        img_pt = cv2.perspectiveTransform(pt, self.H_ref_to_img.astype(np.float32))
        return float(img_pt[0][0][0]), float(img_pt[0][0][1])

    def draw_calibration_zone(self, frame: np.ndarray, color=(0, 255, 0),
                               thickness: int = 2) -> np.ndarray:
        """Dessine, dans l'image camera, le contour de la zone couverte par l'image de reference."""
        annotated = frame.copy()
        height, width = self.reference_shape
        ref_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
        img_corners = cv2.perspectiveTransform(ref_corners, self.H_ref_to_img.astype(np.float32))
        pts = img_corners.astype(int).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)
        return annotated

    def to_dict(self) -> dict:
        return {
            "type": "auto",
            "homography": self.H_img_to_ref.tolist(),
            "reference_shape": list(self.reference_shape),
            "reference_scale": self.reference_scale.to_dict(),
            "num_matches": self.num_matches,
            "num_inliers": self.num_inliers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AutoHomographyCalibrator":
        return cls(
            homography=np.array(data["homography"], dtype=np.float64),
            reference_shape=tuple(data["reference_shape"]),
            reference_scale=ReferenceScale.from_dict(data["reference_scale"]),
            num_matches=data.get("num_matches", 0),
            num_inliers=data.get("num_inliers", 0),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AutoHomographyCalibrator":
        """Charge une calibration automatique depuis un fichier JSON.

        Raises:
            FileNotFoundError: si le fichier n'existe pas.
            CalibrationError: si le contenu est invalide ou correspond a un
                autre type de calibration.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier de calibration introuvable : {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"Fichier de calibration invalide ({path}) : {exc}") from exc

        if data.get("type") != "auto":
            raise CalibrationError(
                f"{path} n'est pas une calibration automatique (type={data.get('type')!r})"
            )

        try:
            return cls.from_dict(data)
        except KeyError as exc:
            raise CalibrationError(f"Fichier de calibration invalide ({path}) : cle manquante {exc}") from exc
