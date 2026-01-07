import subprocess
import re
import numpy as np
import cv2

def detect_video_rotation(video_path: str, debug: bool) -> int:
    """Detecting video rotation from metadata using ffprobe"""
    try:
        # Using: check display_matrix for rotation
        cmd_matrix = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'side_data=rotation',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result_matrix = subprocess.run(cmd_matrix, capture_output=True, text=True)
        
        if result_matrix.returncode == 0:
            txt = result_matrix.stdout.strip()
            if txt:
                match = re.search(r'-?\d+', txt)
                if match:
                    rotation = abs(int(float(match.group()))) if int(float(match.group()))<= 0 else 270
                    if debug:
                        print("Detected rotation from display matrix:", rotation)
                    return rotation
        
        # No rotation found
        if debug:
            print("No rotation metadata found")
        return 0

    except Exception as e:
        if debug:
            print(f"Error detecting rotation: {e}")
        return 0
        
def rotate_frame(frame: np.ndarray, rotation: int, debug: bool = False) -> np.ndarray:
    """Rotate frame based on detected rotation angle"""
    if rotation == 0:
        return frame
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        if debug:
            print(f"Unsupported rotation angle: {rotation}. No rotation applied.")
        return frame