"""Domyślne ustawienia dla procesora ćwiczeń"""

# Domyślne wartości czasu trwania powtórzeń
DEFAULT_DURATIONS = {
    'pushup': {
        'min': 0.6,
        'max': 6.0
    },
    'squat': {
        'min': 0.7,
        'max': 8.0
    }
}

# Ustawienia MediaPipe
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'model_complexity': 2,
    'smooth_landmarks': True,
    'enable_segmentation': True,
    'min_detection_confidence': 0.7,
    'min_tracking_confidence': 0.7
}

# Ustawienia detekcji
DETECTION_CONFIG = {
    'peak_prominence': 0.08,
    'min_distance_multiplier': 0.35,  # * fps
    'smoothing_sigma_multiplier': 0.08,  # * fps
    'min_amplitude_ratio': 0.10
}

# Ustawienia jakości wideo
VIDEO_QUALITY = {
    'low': '28',
    'medium': '23',
    'high': '18',
    'max': '15'
}

# Rozszerzenia plików wideo
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.MOV']

# Kluczowe punkty ciała dla różnych ćwiczeń
KEY_POINTS = {
    'pushup': [11, 12, 13, 14, 15, 16, 23, 24, 27, 28],  # Ramiona, łokcie, nadgarstki, biodra, kostki
    'squat': [11, 12, 23, 24, 25, 26, 27, 28, 29, 30]  # Ramiona, biodra, kolana, kostki, pięty
}

# Progi widoczności
VISIBILITY_THRESHOLD = 0.5
MIN_VISIBLE_KEY_POINTS_RATIO = 0.7

# Progi jakości
QUALITY_THRESHOLDS = {
    'pushup': 0.3,
    'squat': 0.25
}

# Progi konsensusu
CONSENSUS_SUPPORT_THRESHOLD = 2
CONSENSUS_AMPLITUDE_THRESHOLD = 0.25