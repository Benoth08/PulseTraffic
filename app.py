# -*- coding: utf-8 -*-
"""Interface Streamlit de TrafficPulse.

Permet de deposer une video de trafic, regler la calibration homographique
et les parametres de detection, lancer le traitement, puis telecharger la
video annotee et le recapitulatif de calibration.

Lancer avec : streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import streamlit as st

from trafficpulse.calibration import CalibrationError, HomographyCalibrator
from trafficpulse.config import PipelineConfig
from trafficpulse.logging_utils import setup_logging
from trafficpulse.pipeline import TrafficPulsePipeline

setup_logging("INFO")
logger = logging.getLogger(__name__)

DEFAULT_PTS_IMAGE = [[300, 400], [500, 400], [550, 280], [250, 280]]
DEFAULT_PTS_WORLD = [[0, 0], [3.5, 0], [3.5, 10.0], [0, 10.0]]

st.set_page_config(page_title="TrafficPulse - Analyse de trafic", layout="wide")
st.title("TrafficPulse - Analyse de trafic par vision IA")
st.caption("YOLO11 + ByteTrack + homographie pour l'estimation de vitesse")


@st.cache_resource(show_spinner="Chargement du modele...")
def load_model(model_path: str):
    """Charge le modele YOLO une seule fois par session Streamlit.

    Separe du reste de la configuration : un changement de seuil de
    confiance ou de fenetre de lissage ne doit pas declencher un
    rechargement du modele, qui est l'operation couteuse.
    """
    from ultralytics import YOLO
    return YOLO(model_path)


def render_detection_sidebar() -> Dict[str, float]:
    """Affiche les controles de detection et retourne les parametres choisis."""
    st.header("Parametres de detection")
    return {
        "confidence": st.slider("Seuil de confiance", 0.10, 0.90, 0.30, 0.05),
        "smoothing_window": st.slider("Fenetre de lissage (positions)", 3, 15, 5),
        "max_speed": st.number_input("Vitesse maximale plausible (km/h)", value=200.0),
    }


def render_calibration_sidebar() -> Tuple[np.ndarray, np.ndarray]:
    """Affiche les controles de calibration et retourne les points choisis."""
    st.header("Calibration homographique")
    st.caption("4 points image (en pixels) et leur correspondance reelle (en metres)")

    uploaded_calib = st.file_uploader("Charger une calibration existante (JSON)", type=["json"])
    if uploaded_calib is not None:
        try:
            data = json.loads(uploaded_calib.read())
            st.session_state["pts_image"] = data["pts_image"]
            st.session_state["pts_world"] = data["pts_world"]
            st.success("Calibration chargee.")
        except (json.JSONDecodeError, KeyError):
            st.error("Fichier de calibration invalide.")

    default_image = st.session_state.get("pts_image", DEFAULT_PTS_IMAGE)
    default_world = st.session_state.get("pts_world", DEFAULT_PTS_WORLD)

    col_a, col_b = st.columns(2)
    pts_image, pts_world = [], []

    with col_a:
        st.write("Points image (pixels)")
        for i in range(4):
            x = st.number_input(f"P{i + 1} x", value=float(default_image[i][0]), key=f"px{i}")
            y = st.number_input(f"P{i + 1} y", value=float(default_image[i][1]), key=f"py{i}")
            pts_image.append([x, y])

    with col_b:
        st.write("Points monde (metres)")
        for i in range(4):
            x = st.number_input(f"W{i + 1} x", value=float(default_world[i][0]), key=f"wx{i}")
            y = st.number_input(f"W{i + 1} y", value=float(default_world[i][1]), key=f"wy{i}")
            pts_world.append([x, y])

    return np.float32(pts_image), np.float32(pts_world)


def main() -> None:
    with st.sidebar:
        detection_params = render_detection_sidebar()
        st.divider()
        pts_image, pts_world = render_calibration_sidebar()

    try:
        calibrator = HomographyCalibrator(pts_image, pts_world)
    except CalibrationError as exc:
        st.error(f"Calibration invalide : {exc}")
        return

    uploaded_video = st.file_uploader("Deposez une video de trafic", type=["mp4", "avi", "mov"])
    if uploaded_video is None:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_video.read())
        input_path = Path(tmp_input.name)

    output_path = Path(tempfile.gettempdir()) / f"trafficpulse_{uploaded_video.name}"

    cap = cv2.VideoCapture(str(input_path))
    ok, first_frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not ok:
        st.error("Impossible de lire la video deposee.")
        input_path.unlink(missing_ok=True)
        return

    preview = calibrator.draw_calibration_zone(first_frame)
    st.subheader("Apercu et zone de calibration")
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.caption(f"Video : {fps:.0f} fps, {total_frames} frames, duree ~{total_frames / fps:.1f}s")

    calib_json = json.dumps(calibrator.to_dict(), indent=2)
    st.download_button("Telecharger cette calibration (JSON)", data=calib_json,
                        file_name="calibration.json", mime="application/json")

    if not st.button("Lancer l'analyse", type="primary"):
        return

    model = load_model("yolo11n.pt")
    config = PipelineConfig(
        confidence=detection_params["confidence"],
        smoothing_window=detection_params["smoothing_window"],
        max_speed_kmh=detection_params["max_speed"],
    )
    pipeline = TrafficPulsePipeline(config, model=model)

    progress_bar = st.progress(0, text="Traitement en cours...")

    def on_progress(frame_idx: int, total: int) -> None:
        progress_bar.progress(frame_idx / max(total, 1), text=f"Frame {frame_idx}/{total}")

    try:
        stats = pipeline.run(
            input_path=input_path,
            output_path=output_path,
            calibrator=calibrator,
            progress_callback=on_progress,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(f"Echec du traitement : {exc}")
        return
    finally:
        progress_bar.empty()
        input_path.unlink(missing_ok=True)

    st.success("Analyse terminee.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vehicules detectes", stats.unique_vehicles)
    col2.metric("Vitesse moyenne", f"{stats.avg_speed_kmh:.0f} km/h")
    col3.metric("Vitesse mediane", f"{stats.median_speed_kmh:.0f} km/h")
    col4.metric("V85", f"{stats.p85_speed_kmh:.0f} km/h")

    st.subheader("Video annotee")
    st.video(str(output_path))

    with open(output_path, "rb") as f:
        st.download_button("Telecharger la video annotee", data=f,
                            file_name="traffic_analysis.mp4", mime="video/mp4")


if __name__ == "__main__":
    main()
