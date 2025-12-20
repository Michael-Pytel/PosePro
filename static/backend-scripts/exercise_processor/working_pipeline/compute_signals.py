import numpy as np
from typing import Dict, List
from interpolation import interpolate_nans
from visibility_utils import compute_visibility_scores

debug = True


def compute_pushup_signals(landmarks_data: List[Dict], fps) -> Dict[str, np.ndarray]:
    """Compute motion signals for push-ups"""
    
    # To consider wheter use it or not
    # camera_orientation = detect_camera_orientation(landmarks_data)

    # Computing signals visibility score
    key_points = [11, 12, 13, 14, 15, 16, 23, 24, 27, 28]  # Shoulder, elbow, wrist, hip, ankle
    visibility_scores = compute_visibility_scores(landmarks_data, key_points)
    print("Visibility scores:", visibility_scores)
    signals = {}
    
    # BASIC AVERAGE Y-POSITIONS
    avg_hip_y = []
    avg_shoulder_y = []
    avg_wrist_y = []
    avg_elbow_y = []
    avg_knee_y = []
    nose_y = []
    
    # X-POSITIONS 
    avg_hip_x = []
    avg_shoulder_x = []
    torso_center_x = []
    
    # Z-POSITIONS
    avg_hip_z = []
    avg_shoulder_z = []
    torso_center_z = []
    
    # JOINT ANGLES
    left_elbow_angles = []
    right_elbow_angles = []
    left_shoulder_angles = []
    right_shoulder_angles = []
    left_wrist_angles = []
    right_wrist_angles = []
    
    # TORSO AND BODY ANGLES
    torso_angles = []
    body_angles = []
    plank_angles = []  
    
    # SHOULDER HIP DISTANCE AND OTHER 3D DISTANCES
    shoulder_hip_distances = []  
    elbow_knee_distances = []  
    wrist_shoulder_distances = []  
    chest_ground_distances = []  
    
    # WIDTHS
    shoulder_widths = []
    hip_widths = []
    elbow_widths = []
    
    # SYMMETRY
    left_right_shoulder_diff = []
    left_right_hip_diff = []
    
    for data in landmarks_data:
        lm = data['landmarks']
        
        # ===== WYSOKOŚCI Y =====
        if 23 in lm and 24 in lm:
            avg_hip_y.append((lm[23]['y'] + lm[24]['y']) / 2)
            avg_hip_x.append((lm[23]['x'] + lm[24]['x']) / 2)
            avg_hip_z.append((lm[23]['z'] + lm[24]['z']) / 2)
            hip_widths.append(abs(lm[23]['x'] - lm[24]['x']))
            left_right_hip_diff.append(lm[23]['y'] - lm[24]['y'])
        else:
            avg_hip_y.append(np.nan)
            avg_hip_x.append(np.nan)
            avg_hip_z.append(np.nan)
            hip_widths.append(np.nan)
            left_right_hip_diff.append(np.nan)
        
        if 11 in lm and 12 in lm:
            avg_shoulder_y.append((lm[11]['y'] + lm[12]['y']) / 2)
            avg_shoulder_x.append((lm[11]['x'] + lm[12]['x']) / 2)
            avg_shoulder_z.append((lm[11]['z'] + lm[12]['z']) / 2)
            shoulder_widths.append(abs(lm[11]['x'] - lm[12]['x']))
            left_right_shoulder_diff.append(lm[11]['y'] - lm[12]['y'])
        else:
            avg_shoulder_y.append(np.nan)
            avg_shoulder_x.append(np.nan)
            avg_shoulder_z.append(np.nan)
            shoulder_widths.append(np.nan)
            left_right_shoulder_diff.append(np.nan)
        
        if 15 in lm and 16 in lm:
            avg_wrist_y.append((lm[15]['y'] + lm[16]['y']) / 2)
        else:
            avg_wrist_y.append(np.nan)
        
        if 13 in lm and 14 in lm:
            avg_elbow_y.append((lm[13]['y'] + lm[14]['y']) / 2)
            elbow_widths.append(abs(lm[13]['x'] - lm[14]['x']))
        else:
            avg_elbow_y.append(np.nan)
            elbow_widths.append(np.nan)
        
        if 25 in lm and 26 in lm:
            avg_knee_y.append((lm[25]['y'] + lm[26]['y']) / 2)
        else:
            avg_knee_y.append(np.nan)
        
        if 0 in lm:  # Nos
            nose_y.append(lm[0]['y'])
        else:
            nose_y.append(np.nan)
        
        # ===== CENTRUM TUŁOWIA =====
        if 11 in lm and 12 in lm and 23 in lm and 24 in lm:
            torso_center_x.append((lm[11]['x'] + lm[12]['x'] + lm[23]['x'] + lm[24]['x']) / 4)
            torso_center_z.append((lm[11]['z'] + lm[12]['z'] + lm[23]['z'] + lm[24]['z']) / 4)
        else:
            torso_center_x.append(np.nan)
            torso_center_z.append(np.nan)
        
        # ===== KĄTY ŁOKCI =====
        if 11 in lm and 13 in lm and 15 in lm:
            left_elbow_angles.append(calculate_angle(lm[11], lm[13], lm[15]))
        else:
            left_elbow_angles.append(np.nan)
        
        if 12 in lm and 14 in lm and 16 in lm:
            right_elbow_angles.append(calculate_angle(lm[12], lm[14], lm[16]))
        else:
            right_elbow_angles.append(np.nan)
        
        # ===== KĄTY RAMION =====
        if 13 in lm and 11 in lm and 23 in lm:
            left_shoulder_angles.append(calculate_angle(lm[13], lm[11], lm[23]))
        else:
            left_shoulder_angles.append(np.nan)
        
        if 14 in lm and 12 in lm and 24 in lm:
            right_shoulder_angles.append(calculate_angle(lm[14], lm[12], lm[24]))
        else:
            right_shoulder_angles.append(np.nan)
        
        # ===== KĄTY NADGARSTKÓW =====
        if 13 in lm and 15 in lm and 19 in lm:  # 19 = pinky finger
            left_wrist_angles.append(calculate_angle(lm[13], lm[15], lm[19]))
        else:
            left_wrist_angles.append(np.nan)
        
        if 14 in lm and 16 in lm and 20 in lm:
            right_wrist_angles.append(calculate_angle(lm[14], lm[16], lm[20]))
        else:
            right_wrist_angles.append(np.nan)
        
        # ===== KĄT TUŁOWIA =====
        if 11 in lm and 12 in lm and 23 in lm and 24 in lm:
            shoulder_center_y = (lm[11]['y'] + lm[12]['y']) / 2
            hip_center_y = (lm[23]['y'] + lm[24]['y']) / 2
            shoulder_center_x = (lm[11]['x'] + lm[12]['x']) / 2
            hip_center_x = (lm[23]['x'] + lm[24]['x']) / 2
            
            torso_angle = np.degrees(np.arctan2(
                hip_center_y - shoulder_center_y,
                hip_center_x - shoulder_center_x
            ))
            torso_angles.append(abs(torso_angle))
        else:
            torso_angles.append(np.nan)
        
        # ===== KĄT CIAŁA (ramiona-biodra-kostki) =====
        if 11 in lm and 12 in lm and 23 in lm and 24 in lm and 27 in lm and 28 in lm:
            shoulder_avg = {'x': (lm[11]['x'] + lm[12]['x'])/2, 'y': (lm[11]['y'] + lm[12]['y'])/2}
            hip_avg = {'x': (lm[23]['x'] + lm[24]['x'])/2, 'y': (lm[23]['y'] + lm[24]['y'])/2}
            ankle_avg = {'x': (lm[27]['x'] + lm[28]['x'])/2, 'y': (lm[27]['y'] + lm[28]['y'])/2}
            
            body_angles.append(calculate_angle(ankle_avg, hip_avg, shoulder_avg))
        else:
            body_angles.append(np.nan)
        
        # ===== KĄT "DESKI" (nos-biodra-kostki) =====
        if 0 in lm and 23 in lm and 24 in lm and 27 in lm and 28 in lm:
            nose_pt = {'x': lm[0]['x'], 'y': lm[0]['y']}
            hip_avg = {'x': (lm[23]['x'] + lm[24]['x'])/2, 'y': (lm[23]['y'] + lm[24]['y'])/2}
            ankle_avg = {'x': (lm[27]['x'] + lm[28]['x'])/2, 'y': (lm[27]['y'] + lm[28]['y'])/2}
            
            plank_angles.append(calculate_angle(ankle_avg, hip_avg, nose_pt))
        else:
            plank_angles.append(np.nan)
        
        # ===== DYSTANSE 3D =====
        if 11 in lm and 12 in lm and 23 in lm and 24 in lm:
            shoulder_center = np.array([
                (lm[11]['x'] + lm[12]['x']) / 2,
                (lm[11]['y'] + lm[12]['y']) / 2,
                (lm[11]['z'] + lm[12]['z']) / 2
            ])
            hip_center = np.array([
                (lm[23]['x'] + lm[24]['x']) / 2,
                (lm[23]['y'] + lm[24]['y']) / 2,
                (lm[23]['z'] + lm[24]['z']) / 2
            ])
            shoulder_hip_distances.append(np.linalg.norm(shoulder_center - hip_center))
        else:
            shoulder_hip_distances.append(np.nan)
        
        if 13 in lm and 14 in lm and 25 in lm and 26 in lm:
            elbow_center = np.array([
                (lm[13]['x'] + lm[14]['x']) / 2,
                (lm[13]['y'] + lm[14]['y']) / 2,
                (lm[13]['z'] + lm[14]['z']) / 2
            ])
            knee_center = np.array([
                (lm[25]['x'] + lm[26]['x']) / 2,
                (lm[25]['y'] + lm[26]['y']) / 2,
                (lm[25]['z'] + lm[26]['z']) / 2
            ])
            elbow_knee_distances.append(np.linalg.norm(elbow_center - knee_center))
        else:
            elbow_knee_distances.append(np.nan)
        
        if 15 in lm and 16 in lm and 11 in lm and 12 in lm:
            wrist_center = np.array([
                (lm[15]['x'] + lm[16]['x']) / 2,
                (lm[15]['y'] + lm[16]['y']) / 2,
                (lm[15]['z'] + lm[16]['z']) / 2
            ])
            shoulder_center = np.array([
                (lm[11]['x'] + lm[12]['x']) / 2,
                (lm[11]['y'] + lm[12]['y']) / 2,
                (lm[11]['z'] + lm[12]['z']) / 2
            ])
            wrist_shoulder_distances.append(np.linalg.norm(wrist_center - shoulder_center))
        else:
            wrist_shoulder_distances.append(np.nan)
        
        # Wysokość klatki (szacunkowa - punkt między ramionami)
        if 11 in lm and 12 in lm:
            chest_y = (lm[11]['y'] + lm[12]['y']) / 2
            chest_ground_distances.append(1.0 - chest_y)  # Odległość od "podłoża" (Y=1)
        else:
            chest_ground_distances.append(np.nan)
    
    # ========== INTERPOLACJA I ZAPISANIE ==========
    # Wysokości Y
    signals['avg_hip_y'] = interpolate_nans(np.array(avg_hip_y))
    signals['avg_shoulder_y'] = interpolate_nans(np.array(avg_shoulder_y))
    signals['avg_wrist_y'] = interpolate_nans(np.array(avg_wrist_y))
    signals['avg_elbow_y'] = interpolate_nans(np.array(avg_elbow_y))
    signals['avg_knee_y'] = interpolate_nans(np.array(avg_knee_y))
    signals['nose_y'] = interpolate_nans(np.array(nose_y))
    
    # Pozycje X i Z
    signals['avg_hip_x'] = interpolate_nans(np.array(avg_hip_x))
    signals['avg_shoulder_x'] = interpolate_nans(np.array(avg_shoulder_x))
    signals['torso_center_x'] = interpolate_nans(np.array(torso_center_x))
    signals['avg_hip_z'] = interpolate_nans(np.array(avg_hip_z))
    signals['avg_shoulder_z'] = interpolate_nans(np.array(avg_shoulder_z))
    signals['torso_center_z'] = interpolate_nans(np.array(torso_center_z))
    
    # Kąty
    signals['left_elbow_angle'] = interpolate_nans(np.array(left_elbow_angles))
    signals['right_elbow_angle'] = interpolate_nans(np.array(right_elbow_angles))
    signals['avg_elbow_angle'] = (signals['left_elbow_angle'] + signals['right_elbow_angle']) / 2
    signals['left_shoulder_angle'] = interpolate_nans(np.array(left_shoulder_angles))
    signals['right_shoulder_angle'] = interpolate_nans(np.array(right_shoulder_angles))
    signals['avg_shoulder_angle'] = (signals['left_shoulder_angle'] + signals['right_shoulder_angle']) / 2
    signals['left_wrist_angle'] = interpolate_nans(np.array(left_wrist_angles))
    signals['right_wrist_angle'] = interpolate_nans(np.array(right_wrist_angles))
    signals['torso_angle'] = interpolate_nans(np.array(torso_angles))
    signals['body_angle'] = interpolate_nans(np.array(body_angles))
    signals['plank_angle'] = interpolate_nans(np.array(plank_angles))
    
    # Dystanse 3D
    signals['shoulder_hip_distance'] = interpolate_nans(np.array(shoulder_hip_distances))
    signals['elbow_knee_distance'] = interpolate_nans(np.array(elbow_knee_distances))
    signals['wrist_shoulder_distance'] = interpolate_nans(np.array(wrist_shoulder_distances))
    signals['chest_ground_distance'] = interpolate_nans(np.array(chest_ground_distances))
    
    # Szerokości
    signals['shoulder_width'] = interpolate_nans(np.array(shoulder_widths))
    signals['hip_width'] = interpolate_nans(np.array(hip_widths))
    signals['elbow_width'] = interpolate_nans(np.array(elbow_widths))
    
    # Symetria
    signals['left_right_shoulder_diff'] = interpolate_nans(np.array(left_right_shoulder_diff))
    signals['left_right_hip_diff'] = interpolate_nans(np.array(left_right_hip_diff))
    
    # Prędkości
    signals['hip_velocity'] = np.gradient(signals['avg_hip_y']) * fps
    signals['shoulder_velocity'] = np.gradient(signals['avg_shoulder_y']) * fps
    signals['elbow_velocity'] = np.gradient(signals['avg_elbow_y']) * fps
    signals['chest_velocity'] = np.gradient(signals['chest_ground_distance']) * fps
    
    # Przyspieszenia (drugie pochodne)
    signals['hip_acceleration'] = np.gradient(signals['hip_velocity']) * fps
    
    # Synchronizacja
    signals['elbow_sync'] = np.abs(signals['left_elbow_angle'] - signals['right_elbow_angle'])
    signals['shoulder_sync'] = np.abs(signals['left_shoulder_angle'] - signals['right_shoulder_angle'])
    
    # Zapisz orientację dla późniejszego użycia
    # signals['camera_orientation'] = camera_orientation
    
    return signals


def calculate_angle(p1: Dict, p2: Dict, p3: Dict) -> float:
    """Calculate angle"""
    a = np.array([p1['x'], p1['y']])
    b = np.array([p2['x'], p2['y']])
    c = np.array([p3['x'], p3['y']])
    
    ba = a - b
    bc = c - b
    
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    
    return np.degrees(angle)