import numpy as np
from typing import Dict, List
from interpolation import interpolate_nans
from visibility_utils import compute_visibility_scores

# debug = True

def _append_nan(*lists):
    for lst in lists:
        lst.append(np.nan)

def compute_pushup_signals(landmarks_data, fps):
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
    left_shoulder_y = []
    right_shoulder_y = []
    left_elbow_y = []
    right_elbow_y = []
    left_wrist_y = []
    right_wrist_y = []
    left_hip_y = []
    right_hip_y = []
    left_ankle_y = []
    right_ankle_y = []

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

        valid = len(lm) == 33
        if not valid:
            # Y
            _append_nan(
                avg_hip_y, avg_shoulder_y, avg_wrist_y, avg_elbow_y, avg_knee_y,
                nose_y, left_shoulder_y, right_shoulder_y, left_elbow_y, right_elbow_y,
                left_wrist_y, right_wrist_y, left_hip_y, right_hip_y, left_ankle_y, right_ankle_y
            )

            # X
            _append_nan(avg_hip_x, avg_shoulder_x, torso_center_x)

            # Z
            _append_nan(avg_hip_z, avg_shoulder_z, torso_center_z)

            # ANGLES
            _append_nan(
                left_elbow_angles, right_elbow_angles,
                left_shoulder_angles, right_shoulder_angles,
                left_wrist_angles, right_wrist_angles,
                torso_angles, body_angles, plank_angles
            )

            # DISTANCES
            _append_nan(
                shoulder_hip_distances, elbow_knee_distances,
                wrist_shoulder_distances, chest_ground_distances
            )

            # WIDTHS
            _append_nan(shoulder_widths, hip_widths, elbow_widths)

            # SYMMETRY
            _append_nan(left_right_shoulder_diff, left_right_hip_diff)

            continue

            # === od tego miejsca: masz komplet 33 punktów, więc ZERO ifów ===

            # Y-POSITIONS
        left_shoulder_y.append(lm[11]['y'])
        right_shoulder_y.append(lm[12]['y'])
        left_elbow_y.append(lm[13]['y'])
        right_elbow_y.append(lm[14]['y'])
        left_wrist_y.append(lm[15]['y'])
        right_wrist_y.append(lm[16]['y'])
        left_hip_y.append(lm[23]['y'])
        right_hip_y.append(lm[24]['y'])
        left_ankle_y.append(lm[27]['y'])
        right_ankle_y.append(lm[28]['y'])
        nose_y.append(lm[0]['y'])

        # AVG Y + WIDTHS + SYMMETRY + AVG X/Z
        avg_hip_y.append((lm[23]['y'] + lm[24]['y']) / 2)
        avg_hip_x.append((lm[23]['x'] + lm[24]['x']) / 2)
        avg_hip_z.append((lm[23]['z'] + lm[24]['z']) / 2)
        hip_widths.append(abs(lm[23]['x'] - lm[24]['x']))
        left_right_hip_diff.append(lm[23]['y'] - lm[24]['y'])

        avg_shoulder_y.append((lm[11]['y'] + lm[12]['y']) / 2)
        avg_shoulder_x.append((lm[11]['x'] + lm[12]['x']) / 2)
        avg_shoulder_z.append((lm[11]['z'] + lm[12]['z']) / 2)
        shoulder_widths.append(abs(lm[11]['x'] - lm[12]['x']))
        left_right_shoulder_diff.append(lm[11]['y'] - lm[12]['y'])

        avg_wrist_y.append((lm[15]['y'] + lm[16]['y']) / 2)

        avg_elbow_y.append((lm[13]['y'] + lm[14]['y']) / 2)
        elbow_widths.append(abs(lm[13]['x'] - lm[14]['x']))

        avg_knee_y.append((lm[25]['y'] + lm[26]['y']) / 2)

        # TORSO CENTER POSITIONS
        torso_center_x.append((lm[11]['x'] + lm[12]['x'] + lm[23]['x'] + lm[24]['x']) / 4)
        torso_center_z.append((lm[11]['z'] + lm[12]['z'] + lm[23]['z'] + lm[24]['z']) / 4)

        # ELBOW ANGLES
        left_elbow_angles.append(calculate_angle(lm[11], lm[13], lm[15]))
        right_elbow_angles.append(calculate_angle(lm[12], lm[14], lm[16]))

        # SHOULDER ANGLES
        left_shoulder_angles.append(calculate_angle(lm[13], lm[11], lm[23]))
        right_shoulder_angles.append(calculate_angle(lm[14], lm[12], lm[24]))

        # WRIST ANGLES
        left_wrist_angles.append(calculate_angle(lm[13], lm[15], lm[19]))
        right_wrist_angles.append(calculate_angle(lm[14], lm[16], lm[20]))

        # TORSO ANGLE
        shoulder_center_y = (lm[11]['y'] + lm[12]['y']) / 2
        hip_center_y = (lm[23]['y'] + lm[24]['y']) / 2
        shoulder_center_x_ = (lm[11]['x'] + lm[12]['x']) / 2
        hip_center_x_ = (lm[23]['x'] + lm[24]['x']) / 2

        torso_angle = np.degrees(np.arctan2(
            hip_center_y - shoulder_center_y,
            hip_center_x_ - shoulder_center_x_
        ))
        torso_angles.append(abs(torso_angle))

        # BODY ANGLE (ANKLE - HIP - SHOULDER)
        shoulder_avg = {'x': (lm[11]['x'] + lm[12]['x']) / 2, 'y': (lm[11]['y'] + lm[12]['y']) / 2}
        hip_avg = {'x': (lm[23]['x'] + lm[24]['x']) / 2, 'y': (lm[23]['y'] + lm[24]['y']) / 2}
        ankle_avg = {'x': (lm[27]['x'] + lm[28]['x']) / 2, 'y': (lm[27]['y'] + lm[28]['y']) / 2}
        body_angles.append(calculate_angle(ankle_avg, hip_avg, shoulder_avg))

        # PLANK ANGLE (NOSE - HIP - ANKLE)
        nose_pt = {'x': lm[0]['x'], 'y': lm[0]['y']}
        plank_angles.append(calculate_angle(ankle_avg, hip_avg, nose_pt))

        # 3D DISTANCES
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

        wrist_center = np.array([
            (lm[15]['x'] + lm[16]['x']) / 2,
            (lm[15]['y'] + lm[16]['y']) / 2,
            (lm[15]['z'] + lm[16]['z']) / 2
        ])
        wrist_shoulder_distances.append(np.linalg.norm(wrist_center - shoulder_center))

        # CHEST TO GROUND DISTANCE
        chest_y = (lm[11]['y'] + lm[12]['y']) / 2
        chest_ground_distances.append(1.0 - chest_y)

    # INTERPOLATION AND SAVING SIGNALS
    # Y-POSITIONS
    signals['left_shoulder_y'] = interpolate_nans(np.array(left_shoulder_y))
    signals['right_shoulder_y'] = interpolate_nans(np.array(right_shoulder_y))
    signals['left_elbow_y'] = interpolate_nans(np.array(left_elbow_y))
    signals['right_elbow_y'] = interpolate_nans(np.array(right_elbow_y))
    signals['left_wrist_y'] = interpolate_nans(np.array(left_wrist_y))
    signals['right_wrist_y'] = interpolate_nans(np.array(right_wrist_y))
    signals['left_hip_y'] = interpolate_nans(np.array(left_hip_y))
    signals['right_hip_y'] = interpolate_nans(np.array(right_hip_y))
    signals['left_ankle_y'] = interpolate_nans(np.array(left_ankle_y))
    signals['right_ankle_y'] = interpolate_nans(np.array(right_ankle_y))

    signals['avg_hip_y'] = interpolate_nans(np.array(avg_hip_y))
    signals['avg_shoulder_y'] = interpolate_nans(np.array(avg_shoulder_y))
    signals['avg_wrist_y'] = interpolate_nans(np.array(avg_wrist_y))
    signals['avg_elbow_y'] = interpolate_nans(np.array(avg_elbow_y))
    signals['avg_knee_y'] = interpolate_nans(np.array(avg_knee_y))
    signals['nose_y'] = interpolate_nans(np.array(nose_y))

    # X AND Z POSITIONS
    signals['avg_hip_x'] = interpolate_nans(np.array(avg_hip_x))
    signals['avg_shoulder_x'] = interpolate_nans(np.array(avg_shoulder_x))
    signals['torso_center_x'] = interpolate_nans(np.array(torso_center_x))
    signals['avg_hip_z'] = interpolate_nans(np.array(avg_hip_z))
    signals['avg_shoulder_z'] = interpolate_nans(np.array(avg_shoulder_z))
    signals['torso_center_z'] = interpolate_nans(np.array(torso_center_z))

    # ANGLES
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

    # 3D DISTANCES
    signals['shoulder_hip_distance'] = interpolate_nans(np.array(shoulder_hip_distances))
    signals['elbow_knee_distance'] = interpolate_nans(np.array(elbow_knee_distances))
    signals['wrist_shoulder_distance'] = interpolate_nans(np.array(wrist_shoulder_distances))
    signals['chest_ground_distance'] = interpolate_nans(np.array(chest_ground_distances))

    # WIDTH
    signals['shoulder_width'] = interpolate_nans(np.array(shoulder_widths))
    signals['hip_width'] = interpolate_nans(np.array(hip_widths))
    signals['elbow_width'] = interpolate_nans(np.array(elbow_widths))

    # SYMMETRY
    signals['left_right_shoulder_diff'] = interpolate_nans(np.array(left_right_shoulder_diff))
    signals['left_right_hip_diff'] = interpolate_nans(np.array(left_right_hip_diff))

    # VELOCITIES (GRADIENTS)
    signals['hip_velocity'] = np.gradient(signals['avg_hip_y']) * fps
    signals['shoulder_velocity'] = np.gradient(signals['avg_shoulder_y']) * fps
    signals['elbow_velocity'] = np.gradient(signals['avg_elbow_y']) * fps
    signals['chest_velocity'] = np.gradient(signals['chest_ground_distance']) * fps

    # ACCELERATIONS (GRADIENTS OF VELOCITIES)
    signals['hip_acceleration'] = np.gradient(signals['hip_velocity']) * fps

    # SYNC
    signals['elbow_sync'] = np.abs(signals['left_elbow_angle'] - signals['right_elbow_angle'])
    signals['shoulder_sync'] = np.abs(signals['left_shoulder_angle'] - signals['right_shoulder_angle'])
    return signals, visibility_scores


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