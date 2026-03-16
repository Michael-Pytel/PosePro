# PosePro Coach

A Django web application for automated push-up technique assessment using computer vision and machine learning. The system processes a video recording of a push-up set, segments it into individual repetitions, extracts 77 biomechanical features via pose estimation, and evaluates each repetition across three criteria: hip alignment, head position, and range of motion.

This project was developed as an engineering thesis at the Warsaw University of Technology, Faculty of Mathematics and Information Science (MiNI PW), in the field of Data Science.

Authors: Jakub Półtorak, Michał Pytel  
Supervisor: dr Barbara Żogała-Siudem

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Processing Pipeline](#processing-pipeline)
- [Machine Learning Models](#machine-learning-models)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)

---

## Overview

Manual assessment of exercise form is time-consuming and subjective. PosePro Coach addresses this by providing per-repetition biomechanical feedback automatically from a side-view video recording. The system is designed as a demo prototype that is accessible to non-technical users and extensible for future development.

The application accepts a video of a push-up set (MP4, MOV, or AVI, up to 150 MB), processes it entirely on the backend, and presents a report showing which repetitions were performed correctly and which contained technical errors, along with metrics such as eccentric and concentric phase times.

---

## Architecture

The backend consists of six sequentially executed modules, with two modules running in parallel in the final stage:

```
Video Upload
    |
    v
Landmark Extraction Module
(MediaPipe heavy model, multithreaded ProcessPool)
    |
    v
Push-up Repetition Detection Module
(Gaussian smoothing + SciPy peak detection on shoulder trajectory)
    |
    v
Calculating Metrics Module
(Time, ROM, Head, Plank metrics — 77 features total)
    |
    +---------------------------+
    v                           v
Video Cut Module           ML Model Processing Module
(FFmpeg-based clip          (3 serialized sklearn pipelines)
 segmentation)
    |                           |
    +---------------------------+
    v
Generating Report Module
(Django frontend)
```

---

## Processing Pipeline

### 1. Landmark Extraction

Video pre-processing uses FFmpeg (via ffprobe) for reliable rotation detection, achieving 100% accuracy across 110 test recordings. Frames are converted from BGR to RGB before being passed to MediaPipe.

Pose estimation uses the MediaPipe Pose Landmarker (heavy model), which provides 33 body keypoints in normalized 3D coordinates along with per-keypoint visibility scores. The heavy model was chosen over the full model due to superior stability when body parts leave the frame.

To bring the heavy model's processing time to an acceptable level, a multithreaded ProcessPool approach was implemented. The video is split into overlapping segments (10-frame overlap for boundary context), each processed by a separate MediaPipe instance in parallel. This reduced mean extraction time from 46.06 seconds (baseline heavy model) to 17.88 seconds — a 61.2% improvement — while maintaining the quality of the heaviest model.

### 2. Repetition Detection

The analytical approach uses the y-axis position of the more visible shoulder (determined by mean visibility across the recording) as the primary signal. The signal is inverted (to align with Cartesian intuition, since MediaPipe's y-axis increases downward) and smoothed with a Gaussian filter (sigma = fps * 0.04).

Local maxima and minima are found using SciPy's `find_peaks` with a minimum inter-peak distance of `floor(0.5 * fps)` frames and a prominence threshold of 0.04. The detector searches for (peak, valley, peak) patterns and applies an amplitude constraint:

```
amp_l >= 0.18 * min(q_95 - q_05, 0.8)
```

where `q_95` and `q_05` are the 0.95 and 0.05 quantiles of the signal. Edge repetitions that produce (valley, peak) or (peak, valley) boundary patterns are recovered by gradient-based lookback/lookahead logic.

Detection accuracy on a curated set of 82 clean recordings: **92.7% exact match**, **98.8% within ±1 repetition**.

### 3. Metrics Calculation

For each detected repetition, 77 biomechanical features are computed across four categories:

**Time metrics** (not used for ML, presented to the user):
- Total repetition duration
- Eccentric phase time (descent to minimum position)
- Concentric phase time (minimum position to lockout)
- Bottom pause time

**Range of Motion metrics** (12 features):
- Elbow angle (wrist–elbow–shoulder triplet) statistics: max, min, mean, quantiles (q10, q25, q50, q75, q90), std, range of motion angle, binary full_depth and full_lockout flags

**Head metrics** (33 features):
- Torso-head angle aggregates (bottom, min, max, range, mean, quantiles, std)
- Head-forward normalized deviation aggregates
- Head-tilt angle aggregates

**Plank / Hip metrics** (32 features):
- Hip deviation from the heel-to-shoulder axis (sagging/piking), aggregated over the repetition
- Torso angle aggregates (shoulder–hip–knee)
- Body angle aggregates (heel–hip–shoulder, assessing overall line straightness)

3D angle calculation uses the dot product formula for three keypoints A, B, C where B is the vertex:

```
theta = arccos( (BA · BC) / (|BA| * |BC|) )
```

### 4. Video Segmentation

Each repetition is physically cut from the original recording using an FFmpeg-based approach. Processing time was optimized from approximately 10 seconds to 4.3 seconds on average while maintaining original video quality.

### 5. ML Model Processing

Three pre-trained scikit-learn pipelines (serialized as `.pkl` files) are applied to the feature vectors of each detected repetition. Each pipeline produces a binary prediction (Correct / Incorrect) and a confidence score that is displayed to the user.

---

## Machine Learning Models

### Training Setup

- Dataset: 1,104 side-view repetitions from 33 subjects after filtering
- Split: Subject-wise (GroupKFold) to prevent data leakage from temporal/morphological correlation between repetitions of the same person
- Test subjects: IDs 1, 11, 22, 25, 33 (manually selected to preserve class balance)
- Cross-validation: StratifiedGroupKFold with 3 folds
- Optimization: RandomizedSearchCV over scaler type, feature selection method, number of features (k in [5, 32]), and model hyperparameters
- Primary metric: F1 score on the minority (Incorrect) class

Models evaluated: Random Forest, XGBoost, Logistic Regression, SVM (RBF/linear/polynomial kernels).

Scalers evaluated: StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer (Yeo-Johnson), and no scaling (baseline for tree-based models).

Feature selection: SelectKBest with either `f_classif` (ANOVA F-statistic) or `mutual_info_classif`.

### Hip Alignment Pipeline

Predicts whether the hips maintain a straight line between heels and shoulder during the repetition. Deviations are classified as piking (hips above axis) or sagging (hips below axis).

| Component | Configuration |
|---|---|
| Scaler | QuantileTransformer |
| Feature selection | mutual_info_classif, k=13 |
| Model | Logistic Regression (ElasticNet, C=0.936, l1_ratio=0.448) |
| Class weighting | balanced |

### Head Position Pipeline

Predicts whether the head remains in a neutral position aligned with the torso throughout the repetition. Forward head posture (FHP) is the primary error detected.

| Component | Configuration |
|---|---|
| Scaler | RobustScaler |
| Feature selection | f_classif (ANOVA), k=15 |
| Model | SVM (linear kernel, C=0.651) |
| Class weighting | balanced |

### Range of Motion Pipeline

Predicts whether the push-up reaches full depth (upper arm parallel to ground, elbow angle ≤ 90°) and achieves full lockout at the top position.

| Component | Configuration |
|---|---|
| Scaler | RobustScaler |
| Feature selection | f_classif (ANOVA), k=8 |
| Model | SVM (linear kernel, C=1.470) |
| Class weighting | balanced |

---

## Dataset

The dataset was collected specifically for this project, as no suitable public dataset was found.

- 33 subjects (32 male, 1 female)
- Mean age: 23.3 ± 1.7 years
- Mean height: 180 ± 7.0 cm
- Mean weight: 77.5 ± 10.9 kg
- 1,297 total repetitions collected; 1,104 retained after filtering for side-view angle

Videos were recorded from a side perspective at angles chosen by participants. Recordings from non-side perspectives were excluded after visibility analysis, as they precluded reliable assessment of hip alignment and head position.

**Annotation** was performed by two internal annotators (the thesis authors), each labeling each repetition as Correct, Incorrect, Not visible, or Not a rep for all three criteria independently. A custom Django annotation tool was built for this purpose, with per-repetition video playback and iterative correction support.

**Inter-annotator agreement** (Cohen's kappa, side-view filtered, binary labels only):

| Criterion | Kappa (κ) | Agreement level |
|---|---|---|
| Hip alignment | 0.524 | Weak |
| Head position | 0.707 | Moderate |
| Range of motion | 0.816 | Strong |

Final labels used for training were those of Annotator A.

---

## Model Performance

Results on the held-out subject-wise test set:

| Criterion | Model | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Hip alignment | Logistic Regression | 0.76 | 0.65 | 0.90 | 0.87 |
| Head position | SVM (linear) | 0.58 | 0.48 | 0.74 | 0.69 |
| Range of motion | SVM (linear) | 0.56 | 0.57 | 0.50 | 0.70 |

The hip alignment classifier achieves the strongest results, with a recall of 0.90 ensuring that most actual form errors are flagged. Head position and range of motion are harder to predict reliably, partly due to MediaPipe's difficulty estimating the Z axis from a single monocular camera and the inherent subjectivity of annotations.

---

## Requirements

- Python 3.9 or higher
- FFmpeg (must be available on PATH)
- See `requirements.txt` for Python dependencies

Core Python dependencies include:

- Django
- mediapipe
- opencv-python
- scikit-learn
- xgboost
- scipy
- numpy
- matplotlib

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Michael-Pytel/PosePro.git
cd PosePro
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**4. Install FFmpeg**

FFmpeg is required for video rotation detection and clip segmentation. Install it via your system package manager:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

**5. Apply database migrations**

```bash
python manage.py migrate
```

**6. Collect static files (optional, for production)**

```bash
python manage.py collectstatic
```

---

## Running the Application

**Development server:**

```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`.

For deployment instructions (Gunicorn, Nginx, GCP Cloud SQL, Secret Manager, SSL), refer to `Deployment_documentation_and_users_manual.pdf` included in the repository.

---

## Usage

1. Open the application in a browser and navigate to the home page.
2. Click the "Try demo" button to go to the upload view.
3. Upload a video of a push-up set in MP4, MOV, or AVI format (maximum 150 MB).
4. The video must be recorded from a **side view** — this is required for the biomechanical analysis to produce meaningful results.
5. Click "Send and process video". Processing takes approximately 18–60 seconds depending on recording length and hardware.
6. Review the generated report:
   - A summary shows total repetitions, number performed correctly, and per-criterion correctness counts.
   - Each repetition is listed with color-coded indicators (green = correct, red = incorrect) for hip alignment, head position, and range of motion, plus timing information for eccentric and concentric phases.
   - Clicking "Show details" on any repetition reveals model confidence scores and raw biomechanical metric values.

---

## Project Structure

```
PosePro/
├── fitness_app/                       
|   ├── core/
|   |  ├── models/                           # saved pipelines & mediapipe models
|   |  ├── compute_signals.py                # calculating angles and other metrics
|   |  ├── detecting_repetitions.py          # detection of repetitions
|   |  ├── feature_extractor.py              # extracting aggregated features from metrics/report_repetition_tools_*.py
|   |  ├── landmark_extraction.py            # extraction of landmarks using mediapipe
|   |  ├── pose_initialising.py              # initialization of mediapipe pose module
|   |  ├── predictor.py                      # uses the uploaded models and predicts for each repetition hips, range_of_motion, and head correctness
│   ├── metrics/
|   |  ├── report_repetition_tools_basic.py  # aggregation of timing and range_of_motion metrics
|   |  ├── report_repetition_tools_head.py   # aggregation of head metics
|   |  ├── report_repetition_tools_plank.py  # aggregation of hip metrics
│   ├── utils/
│   ├── models.py                            # Database models
│   ├── views.py                             # Request handling and pipeline orchestration
│   ├── urls.py                              # URL routing
|   ├── apps.py                              # uploads ml models at the deploy of the web app. Makes the process of predicting much faster 
|   ├── uploading_processor.py               # orchestrates the full uploading to results pipeline
|   ├── middleware.py                        # makes automated deletion of videos upladed and cut before any new upload
├── static/                                  # Static files (CSS, JS, images)
├── templates/                               # Django HTML templates
├── manage.py
├── requirements.txt
├── Deployment_documentation_and_users_manual.pdf
└── README.md
```

---

## Known Limitations

**Video angle requirement.** The system requires a side-view recording. Front or back perspectives make it impossible to assess hip alignment and head position reliably.

**MediaPipe Z-axis accuracy.** MediaPipe's 3D keypoint estimation from a single monocular camera has known inaccuracies on the Z axis. This particularly affects elbow angle calculation, where the `full_depth` binary feature often fails to match annotator labels despite the repetition being classified as correct. Published methods for improving monocular Z-axis estimation exist but were not reproducible without withheld parameters.

**Dataset homogeneity.** All 33 subjects were university students, predominantly male, with a narrow age range. Model generalization to populations with different demographics, body types, or fitness backgrounds is uncertain.

**Single-exercise scope.** The repetition detection module is designed specifically for push-ups. Extending to other exercises would require partial rebuilding of the detection logic.

**Annotation reliability.** Inter-annotator agreement for hip alignment was weak (κ = 0.524), which places a ceiling on model performance for that criterion regardless of algorithm choice. The high recall (0.90) on hip alignment was prioritized to ensure errors are flagged when they occur.

---

## Future Work

- Expand the exercise library beyond push-ups.
- Improve the repetition detection module to distinguish between different exercise types.
- Incorporate temporal sequence models (TCN, GCN) for feature extraction from movement sequences, which may capture dynamics that per-repetition aggregate features miss.
- Collect a larger and more demographically diverse dataset with standardized recording conditions.
- Investigate improved monocular 3D pose estimation to address Z-axis inaccuracies.
- Add user account functionality, historical session storage, and the ability to share results with a trainer or physiotherapist.
- Explore model calibration to improve the reliability of confidence scores displayed in the report.

