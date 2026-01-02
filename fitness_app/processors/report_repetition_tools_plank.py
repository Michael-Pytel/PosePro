import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass(frozen=True)
class PlankMetrics:
    hip_deviation_bottom: float
    hip_deviation_max_sag: float
    hip_deviation_max_pike: float
    hip_deviation_mean_abs: float
    torso_angle_bottom_deg: float
    torso_angle_max_deg: float
    torso_angle_min_deg: float
    torso_angle_mean_deg: float
    torso_angle_range_deg: float
    body_angle_bottom_deg: float
    body_angle_max_deg: float
    body_angle_min_deg: float
    body_angle_mean_deg: float
    body_angle_range_deg: float
    body_angle: List[float]

def choose_side_visibility(visibility_scores):
    left_side_mean = np.mean([visibility_scores[11], visibility_scores[23], visibility_scores[27]])
    right_side_mean = np.mean([visibility_scores[12], visibility_scores[24], visibility_scores[28]])
    return "left" if left_side_mean > right_side_mean else "right"

def distance_of_hip_to_shoulder_ankle_line(shoulder_x, shoulder_y, hip_x, hip_y, ankle_x, ankle_y):
    """+ means that hip is below shoulder-ankle line
        - means that hip is above shoulder-ankle line"""

    #Shoulder-ankle vector
    dx = ankle_x - shoulder_x
    dy = ankle_y - shoulder_y

    denominator = np.sqrt(dx*dx + dy*dy)
    denominator = np.where(denominator < 1e-9, np.nan, denominator)

    cross = dx * (hip_y - shoulder_y) - dy * (hip_x - shoulder_x)
    d_perpendicular = np.abs(cross) / denominator

    with np.errstate(divide="ignore" ,invalid="ignore"):
        slope = dy / dx
        y_line = shoulder_y + slope * (hip_x - shoulder_x)

    sign = np.sign(hip_y - y_line)
    return d_perpendicular * sign

def compute_plank_metrics(signal_args, start_frame, end_frame, bottom_frame, fps):

    (shoulder_x, shoulder_y, hip_x, hip_y, ankle_x,
        ankle_y, torso_angle, body_angle) = signal_args

    sl = slice(start_frame, end_frame + 1)

    shoulder_x_segment, shoulder_y_segment = shoulder_x[sl], shoulder_y[sl]
    hip_x_segment, hip_y_segment = hip_x[sl], hip_y[sl]
    ankle_x_segment, ankle_y_segment = ankle_x[sl], ankle_y[sl]

    torso_angle_segment = torso_angle[sl]
    body_angle_segment = body_angle[sl]

    hip_deviation = distance_of_hip_to_shoulder_ankle_line(shoulder_x_segment,
                                                           shoulder_y_segment,
                                                           hip_x_segment,
                                                           hip_y_segment,
                                                           ankle_x_segment,
                                                           ankle_y_segment)

    relative_bottom = bottom_frame - start_frame
    hip_deviation_bottom = float(hip_deviation[relative_bottom])
    hip_deviation_max_sag = float(np.nanmax(hip_deviation))
    hip_deviation_max_pike = float(np.nanmin(hip_deviation))
    hip_deviation_mean_abs = float(np.nanmean(np.abs(hip_deviation)))

    torso_angle_bottom_deg = float(torso_angle[relative_bottom])
    torso_angle_max_deg = float(np.nanmax(torso_angle_segment))
    torso_angle_min_deg = float(np.nanmin(torso_angle_segment))
    torso_angle_mean_deg = float(np.nanmean(torso_angle_segment))
    torso_angle_range_deg = float(torso_angle_max_deg - torso_angle_min_deg)

    body_angle_bottom_deg = float(body_angle[relative_bottom])
    body_angle_max_deg = float(np.nanmax(body_angle_segment))
    body_angle_min_deg = float(np.nanmin(body_angle_segment))
    body_angle_mean_deg = float(np.nanmean(body_angle_segment))
    body_angle_range_deg = float(body_angle_max_deg - body_angle_min_deg)

    return PlankMetrics(
        hip_deviation_bottom = hip_deviation_bottom,
        hip_deviation_max_sag = hip_deviation_max_sag,
        hip_deviation_max_pike = hip_deviation_max_pike,
        hip_deviation_mean_abs = hip_deviation_mean_abs,
        torso_angle_bottom_deg = torso_angle_bottom_deg,
        torso_angle_max_deg = torso_angle_max_deg,
        torso_angle_min_deg = torso_angle_min_deg,
        torso_angle_mean_deg = torso_angle_mean_deg,
        torso_angle_range_deg = torso_angle_range_deg,
        body_angle_bottom_deg = body_angle_bottom_deg,
        body_angle_max_deg = body_angle_max_deg,
        body_angle_min_deg = body_angle_min_deg,
        body_angle_mean_deg = body_angle_mean_deg,
        body_angle_range_deg = body_angle_range_deg,
        body_angle= body_angle_segment,
    )

def velocity_threshold(velocity):
    median = np.nanmedian(velocity)
    mad = np.median(np.abs(velocity - median))
    return float(max(1e-6, 2.5 * mad))

def find_start_of_motion(signal, start_frame, end_frame, fps, direction):
    segment = signal[start_frame:end_frame + 1].copy()

    velocity_segment = np.diff(segment) * fps
    vel_threshold = velocity_threshold(velocity_segment)

    if direction == "up":
        mask = (velocity_segment <= -vel_threshold)
    else:
        mask = (velocity_segment >= vel_threshold)

    consecutive_frames = 3
    for i in range(mask.size):
        if mask[i]:
            consecutive_frames -= 1
            if consecutive_frames == 0:
                return start_frame + (i + 1)
        else:
            consecutive_frames = 3
    return False

def normalized_cross_correlation_between_signals(signal_one, signal_two, max_lag_frames):
    n = signal_one.size
    signal_one_norm = signal_one - np.nanmean(signal_one)
    signal_two_norm = signal_two - np.nanmean(signal_two)

    # best_lag, best_corr =


def compute_worming_phase(shoulder_y, hip_y, start_frame, end_frame, fps, direction, is_position):
    shoulder_start_of_motion = find_start_of_motion(shoulder_y, start_frame, end_frame, fps, direction)
    hip_start_of_motion = find_start_of_motion(hip_y, start_frame, end_frame, fps, direction)

    if not shoulder_start_of_motion or not hip_start_of_motion:
        start_lag_frames = None
        start_lag_s = None
    else:
        start_lag_frames = int(hip_start_of_motion - shoulder_start_of_motion)
        start_lag_s = float(start_lag_frames / fps)

    segment_shoulder = shoulder_y[start_frame: end_frame + 1].copy()
    segment_hip = hip_y[start_frame: end_frame + 1].copy()

    if not is_position:
        segment_shoulder = np.diff(segment_shoulder) * fps
        segment_hip = np.diff(segment_hip) * fps

    max_lag_frames = int(round(fps / 2))



def test(signal, visibility_scores, repetition, fps):
    side = choose_side_visibility(visibility_scores)
    print(side)
    signal_args = ((signal['left_shoulder_x'], signal['left_shoulder_y'],
                   signal['left_hip_x'], signal['left_hip_y'],
                   signal['left_ankle_x'], signal['left_ankle_y'],
                   signal['torso_left_angle'], signal['body_left_angle']) if side == "left" else
                   (signal['right_shoulder_x'], signal['right_shoulder_y'],
                    signal['right_hip_x'], signal['right_hip_y'],
                    signal['right_ankle_x'], signal['right_ankle_y'],
                    signal['torso_right_angle'], signal['body_right_angle'])
                   )
    print(compute_plank_metrics(signal_args, repetition['start_frame'], repetition['end_frame'],
                                repetition['bottom_frame'], fps))
    
def get_plank(signal, visibility_scores, repetition, fps) -> Dict[str, Any]:
    side = choose_side_visibility(visibility_scores)
    signal_args = ((signal['left_shoulder_x'], signal['left_shoulder_y'],
                   signal['left_hip_x'], signal['left_hip_y'],
                   signal['left_ankle_x'], signal['left_ankle_y'],
                   signal['torso_left_angle'], signal['body_left_angle']) if side == "left" else
                   (signal['right_shoulder_x'], signal['right_shoulder_y'],
                    signal['right_hip_x'], signal['right_hip_y'],
                    signal['right_ankle_x'], signal['right_ankle_y'],
                    signal['torso_right_angle'], signal['body_right_angle'])
                   )
    plank = compute_plank_metrics(signal_args, repetition['start_frame'], repetition['end_frame'],
                                 repetition['bottom_frame'], fps)
    
    return {
        'hip_deviation_bottom': plank.hip_deviation_bottom,
        'hip_deviation_max_sag': plank.hip_deviation_max_sag,
        'hip_deviation_max_pike': plank.hip_deviation_max_pike,
        'hip_deviation_mean_abs': plank.hip_deviation_mean_abs,
        'torso_angle_bottom_deg': plank.torso_angle_bottom_deg,
        'torso_angle_max_deg': plank.torso_angle_max_deg,
        'torso_angle_min_deg': plank.torso_angle_min_deg,
        'torso_angle_mean_deg': plank.torso_angle_mean_deg,
        'torso_angle_range_deg': plank.torso_angle_range_deg,
        'body_angle_bottom_deg': plank.body_angle_bottom_deg,
        'body_angle_max_deg': plank.body_angle_max_deg,
        'body_angle_min_deg': plank.body_angle_min_deg,
        'body_angle_mean_deg': plank.body_angle_mean_deg,
        'body_angle_range_deg': plank.body_angle_range_deg,
        'body_angle': plank.body_angle.tolist(),
    }