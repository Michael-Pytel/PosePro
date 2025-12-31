import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

# Połączenia MediaPipe Pose
POSE_CONNECTIONS = [
    # (0, 1), (1, 2), (2, 3), (3, 7),  # Twarz
    # (0, 4), (4, 5), (5, 6), (6, 8),  # Twarz
    # (9, 10),  # Usta
    # (11, 12),  # Ramiona
    # (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Lewa ręka
    # (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Prawa ręka
    (11, 23), (12, 24), (23, 24),  # Tułów
    (23, 27),  # Lewa noga
    (24, 28)  # Prawa noga
]


def draw_landmarks_on_video(
        video_path: str,
        frames_data: List[Dict],
        output_dir: str,
        output_name: str = None
):
    """
    Rysuje keypointy MediaPipe na video i zapisuje wynik.

    Args:
        video_path: ścieżka do wideo źródłowego
        frames_data: lista dict z kluczem 'landmarks' (np.array landmarków)
        output_dir: katalog wyjściowy
        output_name: opcjonalna nazwa pliku wyjściowego
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_name is None:
        output_name = Path(video_path).stem + '_keypoints.mp4'

    output_path = str(Path(output_dir) / output_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(frames_data):
            landmarks = frames_data[frame_idx]['landmarks']

            # Rysuj połączenia
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]

                    if start['visibility'] > 0.1 and end['visibility'] > 0.1:
                        start_point = (int(start['x'] * width), int(start['y'] * height))
                        end_point = (int(end['x'] * width), int(end['y'] * height))
                        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

            # Rysuj punkty
            for landmark in landmarks:
                if landmark['visibility'] > 0.1:
                    x = int(landmark['x'] * width)
                    y = int(landmark['y'] * height)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    return output_path