# -*- coding: utf-8 -*-
"""
Analyse de Trafic et Estimation de Vitesse (Smart City)
Modele : YOLO11n (Ultralytics) + ByteTrack
Methode : Homographie (projection perspective -> plan metrique)
"""

import os
import tempfile
import warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

warnings.filterwarnings("ignore")

from ultralytics import YOLO
import ultralytics

print(f"[OK] Ultralytics version : {ultralytics.__version__}")

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[OK] Device : {DEVICE}")


# ---------------------------------------------------------------------------
# CLASSES COCO POUR LES VEHICULES
# ---------------------------------------------------------------------------
# YOLO11 pre-entraine sur COCO connait 80 classes.
# On filtre uniquement les vehicules routiers.

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_CLASS_IDS = set(VEHICLE_CLASSES.keys())


# ---------------------------------------------------------------------------
# CLASSE HOMOGRAPHIE : PROJECTION IMAGE -> PLAN METRIQUE
# ---------------------------------------------------------------------------

class HomographyCalibrator:
    """Gere la transformation de perspective image -> plan metrique.

    L'homographie mappe 4 points du plan image (pixels) vers 4 points du
    plan routier (metres). Tous les points doivent etre coplanaires (surface
    de la route).

    L'hypothese fondamentale est que la route est un plan. Si la route est
    en pente, la projection est biaisee.
    """

    def __init__(self, pts_image: np.ndarray, pts_world: np.ndarray):
        """
        Args:
            pts_image: 4 points en pixels, shape (4, 2), dtype float32.
            pts_world: 4 points correspondants en metres, shape (4, 2), dtype float32.
        """
        self.pts_image = np.float32(pts_image)
        self.pts_world = np.float32(pts_world)
        self.H = cv2.getPerspectiveTransform(self.pts_image, self.pts_world)
        self.H_inv = cv2.getPerspectiveTransform(self.pts_world, self.pts_image)

    def image_to_world(self, px: float, py: float) -> tuple:
        """Projette un point image (px, py) dans le plan metrique.

        Returns:
            (x_metres, y_metres)
        """
        pt = np.array([[[px, py]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, self.H)
        return float(projected[0][0][0]), float(projected[0][0][1])

    def world_to_image(self, mx: float, my: float) -> tuple:
        """Projette un point metrique vers l'image (inverse)."""
        pt = np.array([[[mx, my]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, self.H_inv)
        return float(projected[0][0][0]), float(projected[0][0][1])

    def draw_calibration_zone(self, frame: np.ndarray, color=(0, 255, 255),
                               thickness=2) -> np.ndarray:
        """Dessine le quadrilatere de calibration sur la frame."""
        annotated = frame.copy()
        pts = self.pts_image.astype(int).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=color,
                      thickness=thickness)
        for i, pt in enumerate(self.pts_image.astype(int)):
            cv2.circle(annotated, tuple(pt), 6, color, -1)
            cv2.putText(annotated, f"P{i+1}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return annotated


# ---------------------------------------------------------------------------
# CLASSE SPEED ESTIMATOR : TRACKING + CINEMATIQUE
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """Estime la vitesse des vehicules a partir de leur historique de positions.

    Pour chaque track_id, on stocke les positions metriques successives et on
    calcule la vitesse par difference finie lissee.

    V(t) = ||P(t) - P(t-N)|| / (N / FPS)

    Le lissage sur N frames reduit le bruit du tracking et de la projection.
    """

    def __init__(self, fps: float, smoothing_window: int = 5,
                 max_speed_kmh: float = 200.0, max_history: int = 60):
        """
        Args:
            fps: Images par seconde de la video.
            smoothing_window: Nombre de frames pour le lissage de vitesse.
            max_speed_kmh: Seuil de vitesse aberrante (filtrage).
            max_history: Nombre max de positions stockees par vehicule.
        """
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.max_speed_kmh = max_speed_kmh
        self.max_history = max_history
        self.positions = defaultdict(list)   # track_id -> [(x_m, y_m), ...]
        self.speeds = defaultdict(float)      # track_id -> derniere vitesse km/h

    def update(self, track_id: int, x_metres: float, y_metres: float):
        """Enregistre une nouvelle position et recalcule la vitesse."""
        self.positions[track_id].append((x_metres, y_metres))

        # Limiter la taille de l'historique
        if len(self.positions[track_id]) > self.max_history:
            self.positions[track_id] = self.positions[track_id][-self.max_history:]

        # Calculer la vitesse si assez d'historique
        history = self.positions[track_id]
        n = self.smoothing_window

        if len(history) >= n:
            p_now = np.array(history[-1])
            p_prev = np.array(history[-n])

            distance = np.linalg.norm(p_now - p_prev)
            dt = n / self.fps

            if dt > 0:
                speed_ms = distance / dt
                speed_kmh = speed_ms * 3.6

                # Filtre anti-aberration (ID switch, occlusion)
                if speed_kmh <= self.max_speed_kmh:
                    self.speeds[track_id] = round(speed_kmh, 1)
                # Si aberrant, on garde la derniere vitesse valide

    def get_speed(self, track_id: int) -> float:
        """Retourne la derniere vitesse estimee pour ce track_id."""
        return self.speeds.get(track_id, 0.0)

    def cleanup(self, active_ids: set):
        """Supprime les tracks inactifs pour liberer la memoire."""
        inactive = set(self.positions.keys()) - active_ids
        for tid in inactive:
            del self.positions[tid]
            if tid in self.speeds:
                del self.speeds[tid]


# ---------------------------------------------------------------------------
# ANNOTATION DES FRAMES
# ---------------------------------------------------------------------------

# Couleurs par classe (BGR)
CLASS_COLORS = {
    2: (255, 150, 50),    # car : bleu clair
    3: (0, 255, 255),     # motorcycle : jaune
    5: (0, 200, 0),       # bus : vert
    7: (0, 100, 255),     # truck : orange
}
DEFAULT_COLOR = (200, 200, 200)


def annotate_frame(frame, boxes, ids, classes, confs, speeds_dict,
                   calibrator=None):
    """Annote une frame avec boxes, IDs, classes et vitesses.

    Args:
        frame: Image BGR.
        boxes: Array (n, 4) de bounding boxes [x1, y1, x2, y2].
        ids: Array (n,) de track IDs.
        classes: Array (n,) de class IDs COCO.
        confs: Array (n,) de scores de confiance.
        speeds_dict: dict {track_id: speed_kmh}.
        calibrator: HomographyCalibrator (pour dessiner la zone).

    Returns:
        Frame annotee.
    """
    annotated = frame.copy()

    # Dessiner la zone de calibration
    if calibrator is not None:
        annotated = calibrator.draw_calibration_zone(annotated)

    for i in range(len(ids)):
        track_id = int(ids[i])
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        cls = int(classes[i])
        conf = float(confs[i])
        speed = speeds_dict.get(track_id, 0.0)
        cls_name = VEHICLE_CLASSES.get(cls, "vehicle")
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label : classe + ID + vitesse
        if speed > 0:
            label = f"ID:{track_id} {cls_name} {speed:.0f}km/h"
        else:
            label = f"ID:{track_id} {cls_name}"

        # Fond du label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)

        # Point de contact au sol (centre bas de la boite)
        cx = (x1 + x2) // 2
        cy = y2
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

    return annotated


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL : TRAITEMENT VIDEO
# ---------------------------------------------------------------------------

def process_video(
    input_path: str,
    output_path: str,
    pts_image: np.ndarray,
    pts_world: np.ndarray,
    confidence: float = 0.3,
    smoothing_window: int = 5,
    max_speed_kmh: float = 200.0,
    progress_callback=None
) -> dict:
    """Traite une video complete : detection + tracking + estimation de vitesse.

    Args:
        input_path: Chemin vers la video d'entree (MP4).
        output_path: Chemin vers la video de sortie annotee.
        pts_image: 4 points de calibration en pixels (4, 2).
        pts_world: 4 points de calibration en metres (4, 2).
        confidence: Seuil de confiance YOLO.
        smoothing_window: Fenetre de lissage pour la vitesse.
        max_speed_kmh: Seuil de vitesse aberrante.
        progress_callback: Fonction(frame_idx, total_frames) pour la progression.

    Returns:
        dict avec les statistiques du traitement.
    """
    # Charger le modele
    model = YOLO("yolo11n.pt")

    # Ouvrir la video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la video : {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video : {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    print(f"  Duree : {total_frames / fps:.1f} secondes")

    # Initialiser les composants
    calibrator = HomographyCalibrator(pts_image, pts_world)
    speed_estimator = SpeedEstimator(
        fps=fps,
        smoothing_window=smoothing_window,
        max_speed_kmh=max_speed_kmh
    )

    # Writer pour la video de sortie
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Statistiques
    stats = {
        "total_frames": total_frames,
        "fps": fps,
        "unique_vehicles": set(),
        "speed_records": [],
        "class_counts": defaultdict(int),
    }

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection + Tracking
        results = model.track(
            frame,
            persist=True,
            conf=confidence,
            classes=list(VEHICLE_CLASS_IDS),
            device=DEVICE,
            verbose=False
        )

        result = results[0]
        boxes_obj = result.boxes

        active_ids = set()

        if boxes_obj.id is not None and len(boxes_obj.id) > 0:
            ids = boxes_obj.id.cpu().numpy().astype(int)
            xyxy = boxes_obj.xyxy.cpu().numpy()
            classes = boxes_obj.cls.cpu().numpy().astype(int)
            confs = boxes_obj.conf.cpu().numpy()

            for i in range(len(ids)):
                track_id = int(ids[i])
                cls = int(classes[i])
                x1, y1, x2, y2 = xyxy[i]

                active_ids.add(track_id)
                stats["unique_vehicles"].add(track_id)
                stats["class_counts"][VEHICLE_CLASSES.get(cls, "other")] += 1

                # Centre de la base de la bounding box (contact au sol)
                cx = (x1 + x2) / 2
                cy = y2

                # Projection dans le plan metrique
                x_m, y_m = calibrator.image_to_world(cx, cy)

                # Mise a jour du speed estimator
                speed_estimator.update(track_id, x_m, y_m)

            # Annoter la frame
            annotated = annotate_frame(
                frame, xyxy, ids, classes, confs,
                speed_estimator.speeds, calibrator
            )

            # Collecter les vitesses pour les stats
            for tid in ids:
                sp = speed_estimator.get_speed(int(tid))
                if sp > 5:  # Ignorer les vehicules quasi-immobiles
                    stats["speed_records"].append(sp)
        else:
            annotated = frame.copy()
            if calibrator is not None:
                annotated = calibrator.draw_calibration_zone(annotated)

        # Nettoyage des tracks inactifs
        speed_estimator.cleanup(active_ids)

        # Ecrire la frame annotee
        writer.write(annotated)

        frame_idx += 1
        if progress_callback:
            progress_callback(frame_idx, total_frames)
        elif frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{total_frames} "
                  f"({100 * frame_idx / total_frames:.0f}%)")

    cap.release()
    writer.release()

    # Finaliser les stats
    stats["unique_vehicles"] = len(stats["unique_vehicles"])
    if stats["speed_records"]:
        speeds_arr = np.array(stats["speed_records"])
        stats["avg_speed_kmh"] = round(float(np.mean(speeds_arr)), 1)
        stats["median_speed_kmh"] = round(float(np.median(speeds_arr)), 1)
        stats["max_speed_kmh"] = round(float(np.max(speeds_arr)), 1)
        stats["p85_speed_kmh"] = round(float(np.percentile(speeds_arr, 85)), 1)
    else:
        stats["avg_speed_kmh"] = 0
        stats["median_speed_kmh"] = 0
        stats["max_speed_kmh"] = 0
        stats["p85_speed_kmh"] = 0

    # Convertir class_counts en comptages uniques
    stats["class_counts"] = dict(stats["class_counts"])

    print(f"\n  [OK] Traitement termine : {output_path}")
    print(f"  Vehicules uniques detectes : {stats['unique_vehicles']}")
    print(f"  Vitesse moyenne : {stats['avg_speed_kmh']} km/h")
    print(f"  Vitesse mediane : {stats['median_speed_kmh']} km/h")
    print(f"  V85 : {stats['p85_speed_kmh']} km/h")

    return stats


# ---------------------------------------------------------------------------
# DEMO : TRAITEMENT D'UNE VIDEO
# ---------------------------------------------------------------------------

# Points de calibration par defaut (A ADAPTER A VOTRE VIDEO)
# Ces valeurs sont des exemples. Vous devez les calibrer sur votre video
# en identifiant un rectangle au sol de dimensions connues.

DEFAULT_PTS_IMAGE = np.float32([
    [300, 400],
    [500, 400],
    [550, 280],
    [250, 280]
])

DEFAULT_PTS_WORLD = np.float32([
    [0, 0],
    [3.5, 0],       # Largeur d'une voie : 3.5 metres
    [3.5, 10.0],    # Profondeur : 10 metres
    [0, 10.0]
])


def run_demo(video_path: str, output_path: str = "output_traffic.mp4"):
    """Lance le traitement sur une video de demo."""
    print("\n" + "=" * 60)
    print("TRAFFICPULSE -- Analyse de trafic et estimation de vitesse")
    print("=" * 60)

    if not os.path.exists(video_path):
        print(f"[ERREUR] Video non trouvee : {video_path}")
        print("Telechargez une video de trafic depuis :")
        print("  https://www.pexels.com/search/videos/traffic/")
        print("  https://pixabay.com/videos/search/traffic/")
        return None

    stats = process_video(
        input_path=video_path,
        output_path=output_path,
        pts_image=DEFAULT_PTS_IMAGE,
        pts_world=DEFAULT_PTS_WORLD,
        confidence=0.3,
        smoothing_window=5,
        max_speed_kmh=200.0
    )

    return stats


# ---------------------------------------------------------------------------
# OUTIL DE CALIBRATION (SELECTION DES POINTS)
# ---------------------------------------------------------------------------

def calibration_tool(video_path: str):
    """Outil interactif pour selectionner les 4 points de calibration.

    Ouvre la premiere frame de la video et affiche les coordonnees
    au survol de la souris. Notez les 4 points et reportez-les
    dans DEFAULT_PTS_IMAGE.

    Utilisable en local (pas dans Colab).
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERREUR] Impossible de lire la video.")
        return

    points_selected = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_selected.append((x, y))
            print(f"  Point {len(points_selected)} : ({x}, {y})")
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"P{len(points_selected)}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Calibration", frame)

            if len(points_selected) == 4:
                pts = np.array(points_selected, dtype=int).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
                cv2.imshow("Calibration", frame)
                print("\n  4 points selectionnes. Appuyez sur une touche pour fermer.")

    print("\nOutil de calibration :")
    print("  Cliquez sur 4 points au sol (dans l'ordre : BG, BD, HD, HG)")
    print("  Les coordonnees seront affichees dans le terminal.")

    cv2.imshow("Calibration", frame)
    cv2.setMouseCallback("Calibration", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points_selected) == 4:
        print(f"\npts_image = np.float32({points_selected})")
        print("Copiez cette ligne dans le code et definissez pts_world correspondant.")

    return points_selected


# ---------------------------------------------------------------------------
# GENERATION DU FICHIER STREAMLIT
# ---------------------------------------------------------------------------

STREAMLIT_CODE = '''# -*- coding: utf-8 -*-
"""
TrafficPulse -- Interface Streamlit
Analyse de trafic et estimation de vitesse (YOLO11 + Homographie)
Lancer avec : streamlit run app.py
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

st.set_page_config(page_title="TrafficPulse -- Analyse de Trafic", layout="wide")
st.title("TrafficPulse -- Analyse de Trafic par Vision IA")
st.caption("YOLO11 + ByteTrack + Homographie (estimation de vitesse)")

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASS_IDS = list(VEHICLE_CLASSES.keys())
CLASS_COLORS = {2: (255, 150, 50), 3: (0, 255, 255), 5: (0, 200, 0), 7: (0, 100, 255)}

# --- Sidebar : Parametres ---
with st.sidebar:
    st.header("Parametres de detection")
    confidence = st.slider("Seuil de confiance", 0.10, 0.90, 0.30, 0.05)
    smoothing = st.slider("Fenetre de lissage (frames)", 3, 15, 5)
    max_speed = st.number_input("Vitesse max plausible (km/h)", value=200.0)

    st.divider()
    st.header("Calibration homographique")
    st.caption("4 points image (px) et leur correspondance reelle (m)")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Points image (pixels)**")
        p1x = st.number_input("P1 x", value=300)
        p1y = st.number_input("P1 y", value=400)
        p2x = st.number_input("P2 x", value=500)
        p2y = st.number_input("P2 y", value=400)
        p3x = st.number_input("P3 x", value=550)
        p3y = st.number_input("P3 y", value=280)
        p4x = st.number_input("P4 x", value=250)
        p4y = st.number_input("P4 y", value=280)
    with col_b:
        st.write("**Points monde (metres)**")
        w1x = st.number_input("W1 x", value=0.0)
        w1y = st.number_input("W1 y", value=0.0)
        w2x = st.number_input("W2 x", value=3.5)
        w2y = st.number_input("W2 y", value=0.0)
        w3x = st.number_input("W3 x", value=3.5)
        w3y = st.number_input("W3 y", value=10.0)
        w4x = st.number_input("W4 x", value=0.0)
        w4y = st.number_input("W4 y", value=10.0)

pts_image = np.float32([[p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y]])
pts_world = np.float32([[w1x, w1y], [w2x, w2y], [w3x, w3y], [w4x, w4y]])

# --- Upload video ---
uploaded = st.file_uploader("Deposez une video de trafic (MP4)", type=["mp4", "avi", "mov"])

if uploaded:
    # Sauvegarder dans un fichier temporaire
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_input.write(uploaded.read())
    tmp_input.close()

    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_output.close()

    # Apercu de la premiere frame
    cap_preview = cv2.VideoCapture(tmp_input.name)
    ret, first_frame = cap_preview.read()
    fps = cap_preview.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_preview.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_preview.release()

    if ret:
        # Dessiner la zone de calibration sur la premiere frame
        H = cv2.getPerspectiveTransform(pts_image, pts_world)
        preview = first_frame.copy()
        pts_draw = pts_image.astype(int).reshape((-1, 1, 2))
        cv2.polylines(preview, [pts_draw], True, (0, 255, 255), 2)
        for i, pt in enumerate(pts_image.astype(int)):
            cv2.circle(preview, tuple(pt), 6, (0, 255, 255), -1)

        st.subheader("Apercu et zone de calibration")
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"Video : {fps:.0f} fps, {total_frames} frames, "
                   f"duree ~{total_frames/fps:.1f}s")

    # Bouton de lancement
    if st.button("Lancer l'analyse", type="primary"):
        model = YOLO("yolo11n.pt")
        H = cv2.getPerspectiveTransform(pts_image, pts_world)

        cap = cv2.VideoCapture(tmp_input.name)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_output.name, fourcc, fps, (w, h))

        positions = defaultdict(list)
        speeds = defaultdict(float)
        unique_ids = set()
        all_speeds = []

        progress_bar = st.progress(0, text="Traitement en cours...")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, conf=confidence,
                                  classes=VEHICLE_CLASS_IDS, verbose=False)
            result = results[0]
            boxes = result.boxes
            annotated = frame.copy()

            # Zone de calibration
            pts_draw = pts_image.astype(int).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts_draw], True, (0, 255, 255), 2)

            if boxes.id is not None and len(boxes.id) > 0:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                for i in range(len(ids)):
                    tid = int(ids[i])
                    cls = int(classes[i])
                    x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                    unique_ids.add(tid)

                    cx = (x1 + x2) / 2
                    cy = float(y2)
                    pt = np.array([[[cx, cy]]], dtype=np.float32)
                    proj = cv2.perspectiveTransform(pt, H)
                    xm, ym = float(proj[0][0][0]), float(proj[0][0][1])

                    positions[tid].append((xm, ym))
                    if len(positions[tid]) > 60:
                        positions[tid] = positions[tid][-60:]

                    if len(positions[tid]) >= smoothing:
                        p_now = np.array(positions[tid][-1])
                        p_prev = np.array(positions[tid][-smoothing])
                        dist = np.linalg.norm(p_now - p_prev)
                        dt = smoothing / fps
                        sp = (dist / dt) * 3.6 if dt > 0 else 0
                        if sp <= max_speed:
                            speeds[tid] = round(sp, 1)

                    speed = speeds.get(tid, 0)
                    cls_name = VEHICLE_CLASSES.get(cls, "vehicle")
                    color = CLASS_COLORS.get(cls, (200, 200, 200))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{tid} {cls_name}"
                    if speed > 0:
                        label += f" {speed:.0f}km/h"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
                    cv2.putText(annotated, label, (x1+2, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    if speed > 5:
                        all_speeds.append(speed)

            writer.write(annotated)
            frame_idx += 1
            progress_bar.progress(frame_idx / total_frames,
                                  text=f"Frame {frame_idx}/{total_frames}")

        cap.release()
        writer.release()
        progress_bar.empty()

        # Resultats
        st.success("Analyse terminee.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Vehicules detectes", len(unique_ids))
        if all_speeds:
            sp_arr = np.array(all_speeds)
            col2.metric("Vitesse moyenne", f"{np.mean(sp_arr):.0f} km/h")
            col3.metric("Vitesse mediane", f"{np.median(sp_arr):.0f} km/h")
            col4.metric("V85", f"{np.percentile(sp_arr, 85):.0f} km/h")

        # Video de sortie
        st.subheader("Video annotee")
        st.video(tmp_output.name)

        # Telechargement
        with open(tmp_output.name, "rb") as f:
            st.download_button(
                "Telecharger la video annotee",
                data=f,
                file_name="traffic_analysis.mp4",
                mime="video/mp4"
            )

    # Nettoyage
    try:
        os.unlink(tmp_input.name)
    except Exception:
        pass
'''


def generate_streamlit_app(output_path: str = "app.py"):
    """Genere le fichier app.py pour Streamlit."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(STREAMLIT_CODE)
    print(f"[OK] Fichier {output_path} genere.")
    print(f"     Lancer avec : streamlit run {output_path}")


# ---------------------------------------------------------------------------
# POINT D'ENTREE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generer l'app Streamlit
    generate_streamlit_app("app.py")

    # Lancer la demo si une video est disponible
    demo_video = "traffic_video.mp4"
    if os.path.exists(demo_video):
        stats = run_demo(demo_video)
    else:
        print("\n[INFO] Pour lancer la demo :")
        print(f"  1. Placez une video de trafic sous le nom '{demo_video}'")
        print("  2. Relancez ce script")
        print("  3. Ou lancez directement : streamlit run app.py")
        print("\nTelechargez une video libre de droits depuis :")
        print("  https://www.pexels.com/search/videos/traffic/")
        print("  https://pixabay.com/videos/search/traffic/")
