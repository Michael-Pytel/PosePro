import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_model_path(model_complexity='heavy'):
    script_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    model_name = f'pose_landmarker_{model_complexity}.task'

    # Preferred location: ../model relative to this script
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    path = os.path.join(model_dir, model_name)

    return path

def initialize_pose(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    return landmarker

def pose_initialization_processor():
    model_path = get_model_path('heavy')
    return initialize_pose(model_path)