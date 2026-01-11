# PosePro
The html templates are still temporary, text for now is just for aesthetic reasons.

# HOW TO SETUP (python=3.12)
``` bash
git clone https://github.com/Michael-Pytel/PosePro.git
cd PosePro
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

The web app should be availabe now at: http://127.0.0.1:8000

After configuring once all needed from now on is
``` bash
python manage.py runserver
```

# Most important modules
fitness_app/
 - core/ - all of the core functionalities
    - compute_signals.py - calculating angles and different metrics that will later be aggregated
    - detecting_repetitions.py - detection of repetitions
    - feature_extractor.py - extracting aggregated features from metrics/report_repetition_tools_*.py
    - landmark_extraction.py - extraction of landmarks using mediapipe
    - pose_initialising.py - initialization of mediapipe pose module
    - predictor.py - uses the uploaded models and predicts for each repetition hips, range_of_motion, and head correctness
- metrics/
    - report_repetition_tools_basic.py - aggregation of timing and range_of_motion metrics
    - report_repetition_tools_head.py - aggregation of head metics
    - report_repetition_tools_plank.py - aggregation of hip metrics
- utils/
    - interpolation.py - for interpolating signals before aggregation
    - video_cut.py - script for cutting the video that was uploaded so later these videos will be shown on the results page
    - visibility_utils.py - calculating visibility for a given angle
- middleware.py - makes automated deletion of videos upladed and cut before any new upload
- apps.py - uploads ml models at the deploy of the web app. Makes the process of predicting much faster 
- uploading_processor.py - orchestrates the full uploading to results pipeline 
- views.py - views for uplad and results page
