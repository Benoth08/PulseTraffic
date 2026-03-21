# TrafficPulse -- Analyse de Trafic et Estimation de Vitesse par Vision IA

> Detection, tracking et estimation de vitesse de vehicules en temps reel par YOLO11 + Homographie

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLO11](https://img.shields.io/badge/Model-YOLO11n-purple)
![OpenCV](https://img.shields.io/badge/Geometrie-Homographie-green)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Portfolio%20PoC-orange)

---

## Le probleme metier

Les etudes de trafic classiques (boucles inductives, comptage pneumatique) coutent 2 000 a 5 000 euros par campagne, ne donnent que des flux agreges, et necessitent des interventions sur chaussee. Les radars fixes coutent 50 000 a 200 000 euros par unite. Les cameras de videosurveillance existantes sont sous-exploitees.

## La solution

Transformer n'importe quelle camera fixe existante en station de comptage et de mesure de vitesse. Le pipeline detecte les vehicules (YOLO11), les suit individuellement entre les frames (ByteTrack), et estime leur vitesse reelle en km/h grace a une transformation de perspective (homographie). Tout tourne en local.

## Architecture

```
Video MP4 (camera fixe)
    |  [OpenCV VideoCapture]
    v
Frame N (image BGR)
    |  [YOLO11n + ByteTrack]
    v
Detections + Track IDs (x, y, w, h, id, class)
    |  [Homographie : cv2.getPerspectiveTransform]
    v
Position dans le plan routier (metres)
    |  [Cinematique : V = delta_d / delta_t]
    v
Vitesse par vehicule (km/h)
    |  [Annotation OpenCV]
    v
Video annotee (boxes + ID + vitesse) --> Streamlit
```

## Stack technique

| Composant | Technologie | Justification |
|---|---|---|
| Detection | YOLO11n (COCO pre-entraine) | Classes vehicules natives, zero fine-tuning |
| Tracking | ByteTrack (Ultralytics natif) | Filtre de Kalman + association IoU |
| Geometrie | OpenCV (getPerspectiveTransform) | Projection image -> plan metrique |
| Cinematique | NumPy | Derivee discrete lissee V = d/dt |
| Video I/O | OpenCV (VideoCapture/Writer) | Lecture et ecriture MP4 |
| Interface | Streamlit | Upload + parametrage + telechargement resultat |

## Resultats

| Metrique | Valeur |
|---|---|
| Detection vehicules | mAP@0.5 > 0.80 (COCO pre-entraine) |
| Tracking (MOTA) | Dependant de la scene |
| Precision vitesse | 5-15% selon la calibration |
| Inference | ~30-60 ms/frame (CPU, YOLO11n) |

## Structure du projet

```
TrafficPulse/
├── README.md
├── requirements.txt
├── traffic_pulse.py             # Code complet (pipeline + Streamlit gen)
├── cours_trafficpulse.md        # Cours pedagogique
├── app.py                       # Interface Streamlit (generee)
└── traffic_video.mp4            # Video de test (a fournir)
```

## Installation et execution

```bash
# Cloner
git clone https://github.com/<votre-username>/TrafficPulse.git
cd TrafficPulse

# Installer
pip install -r requirements.txt

# Telecharger une video de trafic (camera fixe) depuis :
#   https://www.pexels.com/search/videos/traffic/
# et la nommer traffic_video.mp4

# Option 1 : Script direct
python traffic_pulse.py

# Option 2 : Interface Streamlit
streamlit run app.py
```

## Calibration de l'homographie

C'est l'etape critique qui requiert une intervention manuelle :

1. Identifier dans la video un rectangle au sol de dimensions connues (passage pieton, marquage de voie, espacement entre bandes blanches)
2. Relever les 4 coins en pixels (P1 bas-gauche, P2 bas-droit, P3 haut-droit, P4 haut-gauche)
3. Definir les coordonnees reelles correspondantes en metres
4. Reporter ces valeurs dans l'interface Streamlit (sidebar)

Un outil de calibration interactif (clic sur l'image) est inclus dans le code pour faciliter cette etape en local.

## Concepts physiques cles

### Homographie (transformation de perspective)
Matrice 3x3 qui mappe le plan image vers le plan routier. Hypothese : la route est plane. Necessite 4 correspondances point image <-> point reel.

### Point de contact au sol
On projette le centre de la BASE de la bounding box (x_center, y_bottom), pas le centre geometrique. Le bas de la boite est le contact vehicule/route, seul point dans le plan de l'homographie.

### Vitesse par derivee discrete lissee
V(t) = ||P(t) - P(t-N)|| / (N/FPS). Le lissage sur N frames (defaut: 5) reduit le bruit du tracking. A 30 fps, N=5 correspond a un lissage sur 167 ms.

### V85
Le 85e percentile des vitesses mesurees. Indicateur standard en ingenierie du trafic pour dimensionner les infrastructures.

## Limitations

- **Camera fixe obligatoire** : tout mouvement invalide l'homographie
- **Plan routier** : les pentes biaisent la projection
- **Non homologue** : precision de 5-15%, pas un radar certifie
- **ID switches** : le tracker peut perdre un vehicule (occlusion) et generer des vitesses aberrantes (filtrees par seuil)
- **FPS critique** : en dessous de 15 fps, l'estimation de vitesse se degrade

## Extensions possibles

- Comptage de flux directionnels (entrees/sorties par voie)
- Detection d'incidents (vehicule arrete, contresens)
- Heatmap de densite de trafic
- Temps de parcours entre deux cameras
- Export des donnees en CSV pour analyse statistique
- Integration avec un systeme de gestion de trafic (SCADA)

## Auteur

Projet realise dans le cadre d'un portfolio Data Science / IA industrielle.
Profil : Docteur-Ingenieur, expertise physique/optique/metrologie, reconversion IA.

---

*Ce projet demontre la maitrise de la chaine complete vision temps reel (detection, tracking, geometrie projective) et la capacite a passer du plan image au plan metrique -- competence de physicien appliquee a la Smart City.*
