# TrafficPulse -- Analyse de Trafic et Estimation de Vitesse par Vision IA

> Detection, tracking et estimation de vitesse de vehicules en temps reel par YOLO11 + Homographie

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLO11](https://img.shields.io/badge/Model-YOLO11n-purple)
![OpenCV](https://img.shields.io/badge/Geometrie-Homographie-green)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-red)
![Tests](https://img.shields.io/badge/Tests-pytest-yellow)
![Status](https://img.shields.io/badge/Status-Portfolio%20PoC-orange)

---

## Le probleme metier

Les etudes de trafic classiques (boucles inductives, comptage pneumatique) coutent 2 000 a 5 000 euros par campagne, ne donnent que des flux agreges, et necessitent des interventions sur chaussee. Les radars fixes coutent 50 000 a 200 000 euros par unite. Les cameras de videosurveillance existantes sont sous-exploitees.

## La solution

Transformer n'importe quelle camera fixe existante en station de comptage et de mesure de vitesse. Le pipeline detecte les vehicules (YOLO11), les suit individuellement entre les frames (ByteTrack), et estime leur vitesse reelle en km/h grace a une transformation de perspective (homographie). Tout tourne en local.

## Architecture du traitement

```
Video MP4 (camera fixe)
    |  [OpenCV VideoCapture]
    v
Frame N (image BGR)
    |  [YOLO11n + ByteTrack]
    v
Detections + identifiants de suivi (x, y, w, h, id, classe)
    |  [Homographie : cv2.getPerspectiveTransform]
    v
Position dans le plan routier (metres)
    |  [Cinematique : V = delta_d / delta_t]
    v
Vitesse par vehicule (km/h)
    |  [Annotation OpenCV]
    v
Video annotee (boites + ID + vitesse)
```

## Architecture logicielle

Le projet separe strictement la logique metier de l'interface utilisateur. Le package `trafficpulse` ne depend d'aucune librairie d'interface (ni Streamlit, ni argparse) : il peut etre teste, importe et reutilise independamment.

| Module | Role |
|---|---|
| `trafficpulse/config.py` | Parametres du pipeline (dataclass validee), detection automatique du device |
| `trafficpulse/calibration.py` | Calibration manuelle (4 points), interface commune `BaseCalibrator`, chargement JSON generique |
| `trafficpulse/auto_calibration.py` | Calibration automatique par appariement de points d'interet (SIFT + RANSAC) |
| `trafficpulse/tracking.py` | Estimation de vitesse a partir de l'historique de positions |
| `trafficpulse/annotation.py` | Classes vehicules et annotation visuelle des frames |
| `trafficpulse/pipeline.py` | Orchestration complete : detection, suivi, vitesse, export CSV |
| `cli.py` | Point d'entree ligne de commande (traitement par lot, scripts) |
| `app.py` | Interface Streamlit, consommateur du package, aucune logique dupliquee |
| `calibration_tool.py` | Outil interactif de selection des points de calibration manuelle |
| `tests/` | Tests unitaires de la calibration (manuelle et automatique) et de l'estimation de vitesse |

`app.py` et `cli.py` appellent tous les deux `TrafficPulsePipeline`, ce qui garantit un comportement identique quelle que soit l'interface utilisee. Le pipeline accepte indifferemment une calibration manuelle ou automatique : les deux implementent la meme interface (`BaseCalibrator`).

## Stack technique

| Composant | Technologie | Justification |
|---|---|---|
| Detection | YOLO11n (COCO pre-entraine) | Classes vehicules natives, zero fine-tuning |
| Tracking | ByteTrack (Ultralytics natif) | Filtre de Kalman + association IoU |
| Geometrie | OpenCV (getPerspectiveTransform) | Projection image -> plan metrique |
| Calibration automatique | OpenCV (SIFT + BFMatcher + RANSAC) | Mise en correspondance automatique avec une image de reference a l'echelle connue |
| Cinematique | NumPy | Derivee discrete lissee V = d/dt |
| Video I/O | OpenCV (VideoCapture/Writer), repli automatique de codec | Lecture et ecriture MP4 |
| Interface | Streamlit | Upload + parametrage + telechargement resultat |
| Tests | pytest | Calibration et estimation de vitesse testees sans dependance au modele |

## Resultats

| Metrique | Valeur |
|---|---|
| Detection vehicules | mAP@0.5 > 0.80 (COCO pre-entraine) |
| Tracking (MOTA) | Dependant de la scene |
| Precision vitesse | 5-15% selon la calibration |
| Inference | ~30-60 ms/frame (CPU, YOLO11n) |

## Structure du projet

```
PulseTraffic/
  README.md
  requirements.txt
  requirements-dev.txt
  conftest.py
  cli.py                       (point d'entree CLI, argparse)
  app.py                       (interface Streamlit)
  calibration_tool.py          (outil interactif de calibration)
  trafficpulse/
    __init__.py
    config.py
    calibration.py
    auto_calibration.py
    tracking.py
    annotation.py
    pipeline.py
    logging_utils.py
  tests/
    test_calibration.py
    test_auto_calibration.py
    test_tracking.py
  traffic_video.mp4            (video de test a fournir, non versionnee)
```

## Installation et execution

```bash
# Cloner
git clone https://github.com/Benoth08/PulseTraffic.git
cd PulseTraffic

# Installer (utiliser requirements-dev.txt pour lancer aussi les tests)
pip install -r requirements.txt

# Telecharger une video de trafic (camera fixe) depuis :
#   https://www.pexels.com/search/videos/traffic/
# et la nommer traffic_video.mp4

# Option 1 : CLI
python cli.py --input traffic_video.mp4 --output result.mp4

# Option 2 : Interface Streamlit
streamlit run app.py
```

Le CLI accepte plusieurs options utiles en production :

```bash
python cli.py \
    --input traffic_video.mp4 \
    --output result.mp4 \
    --calib-file calibration.json \
    --confidence 0.35 \
    --smoothing-window 5 \
    --max-speed 150 \
    --csv-export vehicules.csv \
    --log-level INFO
```

## Calibration de l'homographie

Deux approches sont disponibles, toutes deux produisant un objet calibrateur compatible avec le pipeline (interface `BaseCalibrator`).

### Calibration manuelle (4 points cliques)

Methode de reference, fiable et previsible :

1. Identifier dans la video un rectangle au sol de dimensions connues (passage pieton, marquage de voie, espacement entre bandes blanches)
2. Relever les 4 coins en pixels (P1 bas-gauche, P2 bas-droit, P3 haut-droit, P4 haut-gauche)
3. Definir les coordonnees reelles correspondantes en metres
4. Sauvegarder cette calibration au format JSON, reutilisable ensuite par le CLI (`--calib-file`) ou l'interface Streamlit (chargement de fichier)

Deux methodes pour obtenir ces points :

- `python calibration_tool.py traffic_video.mp4 calibration.json` : outil interactif en local (clic sur l'image), genere directement le fichier JSON
- Saisie manuelle des 8 valeurs dans la barre laterale de l'interface Streamlit, avec export possible vers JSON

### Calibration automatique (SIFT + RANSAC)

Plutot que de cliquer des points, cette methode appaire automatiquement la frame camera avec une image de reference du meme site dont l'echelle est connue (orthophoto, plan, capture satellite). Le principe :

1. Detection de points d'interet (SIFT) dans la frame camera et dans l'image de reference
2. Mise en correspondance brute-force (BFMatcher) filtree par le test du rapport de Lowe
3. Estimation robuste de l'homographie par RANSAC, qui rejette automatiquement les correspondances aberrantes
4. Combinaison avec l'echelle de l'image de reference (deduite de deux points dont la distance reelle est connue) pour obtenir directement la projection pixel camera -> metres

```bash
python cli.py \
    --input traffic_video.mp4 \
    --output result.mp4 \
    --auto-calib \
    --reference-image plan_site.png \
    --ref-point1 120,80 \
    --ref-point2 420,80 \
    --ref-distance 14.0 \
    --save-calib calibration_auto.json
```

`--ref-point1` et `--ref-point2` sont deux points en pixels dans l'image de reference, `--ref-distance` leur distance reelle en metres (par exemple la largeur connue d'une chaussee sur le plan). `--save-calib` permet de conserver le resultat pour le reutiliser tel quel via `--calib-file`, sans recalculer l'appariement a chaque execution.

Cette methode depend de la qualite et du recouvrement de l'image de reference : un faible taux d'inliers (journalise a l'execution) signale une calibration peu fiable, auquel cas la methode manuelle reste preferable.

### Validation commune

Quelle que soit la methode, les points de calibration manuelle sont valides automatiquement avant tout traitement : un quadrilatere degenere (points confondus ou alignes) est rejete avec un message explicite plutot que de produire des vitesses silencieusement fausses. Pour la calibration automatique, un nombre insuffisant de correspondances fiables declenche egalement une erreur explicite plutot qu'une homographie aberrante.

## Concepts physiques cles

### Homographie (transformation de perspective)
Matrice 3x3 qui mappe le plan image vers le plan routier. Hypothese : la route est plane. Necessite 4 correspondances point image <-> point reel.

### Point de contact au sol
On projette le centre de la base de la boite englobante (x_center, y_bottom), pas le centre geometrique. Le bas de la boite est le contact vehicule/route, seul point reellement situe dans le plan de l'homographie.

### Vitesse par derivee discrete lissee
Sur une fenetre de N positions consecutives (une par frame), l'ecart entre la position la plus recente et la plus ancienne couvre N-1 intervalles de temps, pas N :

```
V(t) = ||P(t) - P(t-(N-1))|| / ((N-1) / fps)
```

A 30 fps, N=5 correspond a un lissage sur 4 intervalles, soit 133 ms.

### V85
Le 85e percentile des vitesses mesurees. Indicateur standard en ingenierie du trafic pour dimensionner les infrastructures.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Les tests couvrent la calibration manuelle (precision de projection sur points connus, rejet des quadrilateres degeneres, sauvegarde/chargement JSON), la calibration automatique (recuperation d'une homographie connue sur un cas synthetique, rejet des images sans texture exploitable, serialisation) et l'estimation de vitesse (calcul sur deplacement connu, filtrage des vitesses aberrantes, liberation memoire des vehicules inactifs). Ils ne dependent ni du modele YOLO ni d'une video reelle, et s'executent en moins de deux secondes.

## Limitations

- **Camera fixe obligatoire** : tout mouvement invalide l'homographie
- **Plan routier** : les pentes biaisent la projection
- **Non homologue** : precision de 5-15%, pas un radar certifie
- **Changements d'identifiant de suivi** : le tracker peut perdre un vehicule (occlusion) et generer des vitesses aberrantes, filtrees par seuil configurable
- **FPS critique** : en dessous de 15 fps, l'estimation de vitesse se degrade
- **Codec video** : le writer tente H264 (avc1) en priorite pour une lecture native dans le navigateur, avec repli automatique sur mp4v puis XVID selon les codecs disponibles sur le systeme
- **Calibration automatique** : la fiabilite depend du recouvrement et de la texture de l'image de reference ; une scene peu texturee (chaussee uniforme, vue trop eloignee) peut ne pas fournir assez de correspondances fiables, auquel cas la calibration manuelle reste la methode de repli

## Extensions possibles

- Comptage de flux directionnels (entrees/sorties par voie)
- Detection d'incidents (vehicule arrete, contresens)
- Heatmap de densite de trafic
- Temps de parcours entre deux cameras
- Integration avec un systeme de gestion de trafic (SCADA)
- Conteneurisation (Dockerfile) pour deploiement sur poste distant

L'export CSV par vehicule (`--csv-export`) est deja disponible via le CLI.

## Auteur

Projet realise dans le cadre d'un portfolio Data Science / IA industrielle.
Profil : Docteur-Ingenieur, expertise physique/optique/metrologie, reconversion IA.

---

*Ce projet demontre la maitrise de la chaine complete vision temps reel (detection, tracking, geometrie projective) et la capacite a passer du plan image au plan metrique -- competence de physicien appliquee a la Smart City.*
