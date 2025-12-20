"""Extracting pose landmarks from video using MediaPipe."""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
# from compute_signals import compute_pushup_signals
# from butterworth_filtration import butterworth_filter_matrix, extract_xyz_matrix, apply_filtered_matrix

# from ..config import (
#     MEDIAPIPE_CONFIG,
#     KEY_POINTS,
#     VISIBILITY_THRESHOLD,
#     MIN_VISIBLE_KEY_POINTS_RATIO
# )
from video_rotation import detect_video_rotation, rotate_frame
debug = True

def extract_landmarks(landmarker, video_path: str) -> List[Dict]:
    """Landmark extraction from video"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Turning off automatic orientation correction
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    
    # Detecting rotation from metadata
    rotation = detect_video_rotation(video_path, debug=debug)
    
    # Preparing all necessary variables
    landmarks_data = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detection_stats = {
        'total_frames': 0,
        'detected_frames': 0,
        'failed_frames': 0
    }
    i = 0
    # Reading each frame and processing
    while cap.isOpened():            
        ret, frame = cap.read()
        if not ret:
            break
        # Copying frame for mediapipe processing
        mp_frame = frame.copy()
        detection_stats['total_frames'] += 1
        if rotation != 0:
            mp_frame = rotate_frame(mp_frame, rotation, debug = debug)
#   #   #  # To validate, whether it's needed        
        # processed = cv2.convertScaleAbs(mp_frame, alpha=1.3, beta=15)
        
        # RGB conversion
        mp_frame_rgb = cv2.cvtColor(mp_frame, cv2.COLOR_BGR2RGB)
        # Mediapipe processing
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_frame_rgb)
        
        pose_landmarker_result = landmarker.detect_for_video(mp_image, int((frame_idx / fps) * 1000))

        #Processing normalized landmarks
        # landmarks_mp = pose_landmarker_result.pose_landmarks[0]
        if pose_landmarker_result.pose_landmarks:
            landmarks_mp = pose_landmarker_result.pose_landmarks[0]
            key_points = [11, 12, 13, 14, 15, 16, 23, 24, 27, 28]  # Shoulder, elbow, wrist, hip, ankle
            visible_key_points = 0
            
            landmarks = {}
            for idx, lm in enumerate(landmarks_mp):
                landmarks[idx] = {
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                }
                # Validating key points, whether they are visible enough
                if idx in key_points and lm.visibility > 0.5:
                    visible_key_points += 1
            # Thresholding based on visible key points
            if visible_key_points >= len(key_points) * 0.1:
                landmarks_data.append({
                    'frame': frame_idx,
                    'time': frame_idx / fps,
                    'landmarks': landmarks,
                    'visibility_score': visible_key_points / len(key_points),
                    'rotation_applied': rotation
                })
                detection_stats['detected_frames'] += 1
            else:
                detection_stats['failed_frames'] += 1
        else:
            detection_stats['failed_frames'] += 1
        
        frame_idx += 1
        
        if frame_idx % 50 == 0 and debug:
            progress = (frame_idx / total_frames) * 100
            detection_rate = (detection_stats['detected_frames'] / detection_stats['total_frames']) * 100
            print(f"  Progress: {progress:.1f}% | Detection: {detection_rate:.1f}%", end='\r') 
    
    cap.release()

    # Detection statistics summary
    if detection_stats['total_frames'] > 0:
        detection_rate = (detection_stats['detected_frames'] / detection_stats['total_frames']) * 100
        print(f"\n  Detection rate: {detection_stats['detected_frames']}/{detection_stats['total_frames']} "
                f"({detection_rate:.1f}%)")
    
    # landmark_array = extract_xyz_matrix(landmarks_data)
    # filtered_array = butterworth_filter_matrix(landmark_array, fs=fps)
    # filtered_landmarks_data = apply_filtered_matrix(landmarks_data, filtered_array)
    
    return landmarks_data

def visualize_keypoint_y_over_frames(landmarks_data, key_points):
    """
    Rysuje wykres: X = numer klatki, Y = pozycja Y znormalizowana dla wybranych landmarków.
    """
    # przygotowanie słownika trajektorii
    trajectories = {kp: {'frame': [], 'y': []} for kp in key_points}

    # zbieranie danych
    for entry in landmarks_data:
        frame = entry['frame']
        lm = entry['landmarks']

        for kp in key_points:
            if kp in lm:
                trajectories[kp]['frame'].append(frame)
                trajectories[kp]['y'].append(lm[kp]['y'])  # znormalizowane Y

    # rysowanie wykresu
    plt.figure(figsize=(12, 6))
    plt.title("Trajektorie pozycji Y kluczowych landmarków w czasie (klatki)")
    
    for kp in key_points:
        plt.plot(trajectories[kp]['frame'],
                 trajectories[kp]['y'],
                 linewidth=1,
                 label=f"LM {kp}")

    plt.xlabel("Frame (klatki)")
    plt.ylabel("Y (znormalizowane)")
    plt.gca().invert_yaxis()   # bo w obrazie Y rośnie w dół
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()   # ⭐ POJAWI SIĘ OKNO Z WYKRESEM


if __name__ == "__main__":
    a = 0
    # model_path = 'C:\\Users\\jakub\\Documents\\Inzynierka\\django-app\\static\\backend-scripts\\exercise_processor\\model\\pose_landmarker_full.task'
    # BaseOptions = mp.tasks.BaseOptions
    # PoseLandmarker = mp.tasks.vision.PoseLandmarker
    # PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    # VisionRunningMode = mp.tasks.vision.RunningMode

    # # Create a pose landmarker instance with the video mode:
    # options = PoseLandmarkerOptions(
    #     base_options=BaseOptions(model_asset_path=model_path),
    #     running_mode=VisionRunningMode.VIDEO)

    # with PoseLandmarker.create_from_options(options) as landmarker:
    # pose = mp.solutions.pose.Pose(
    #     static_image_mode=False,
    #     model_complexity=2,
    #     smooth_landmarks=True,
    #     enable_segmentation=True,
    #     min_detection_confidence=0.7,
    #     min_tracking_confidence=0.7
    # )
    #     test = extract_landmarks(landmarker, "C:\\Users\\jakub\\Documents\\Inzynierka\\camera-based-exercise-evaluation\\data\\recordings\\own_recordings\\pushups\\filiplis\\filiplis2.mp4")
    #     test2 = compute_pushup_signals(test, 30)
    #     print(test2['avg_hip_y'])
    # visualize_keypoint_y_over_frames(test, [11,12,13,14,15,16,23,24,27,28])
