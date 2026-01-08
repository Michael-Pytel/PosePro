import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class HeadMetrics:
    torso_head_angle_bottom: float
    torso_head_angle_min: float
    torso_head_angle_max: float
    torso_head_angle_range: float
    torso_head_angle_mean: float
    torso_head_angle_q_10: float
    torso_head_angle_q_25: float
    torso_head_angle_q_50: float
    torso_head_angle_q_75: float
    torso_head_angle_q_90: float
    torso_head_angle_std: float
    head_forward_norm_bottom: float
    head_forward_norm_min: float
    head_forward_norm_max: float
    head_forward_norm_range: float
    head_forward_norm_mean: float
    head_forward_norm_q_10: float
    head_forward_norm_q_25: float
    head_forward_norm_q_50: float
    head_forward_norm_q_75: float
    head_forward_norm_q_90: float
    head_forward_norm_std: float
    head_tilt_angle_bottom: float
    head_tilt_angle_min: float
    head_tilt_angle_max: float
    head_tilt_angle_range: float
    head_tilt_angle_mean: float
    head_tilt_angle_q_10: float
    head_tilt_angle_q_25: float
    head_tilt_angle_q_50: float
    head_tilt_angle_q_75: float
    head_tilt_angle_q_90: float
    head_tilt_angle_std: float

def choose_side_visibility(visibility_scores):
    left_side_mean = np.mean([visibility_scores[7], visibility_scores[11], visibility_scores[27]])
    right_side_mean = np.mean([visibility_scores[8], visibility_scores[12], visibility_scores[28]])
    return "left" if left_side_mean > right_side_mean else "right"


def compute_head_metrics(args, start_frame, end_frame, bottom_frame):
    (ear_x, ear_y, shoulder_x, shoulder_y, ankle_x, ankle_y, torso_head_angle, head_tilt_angle) = args

    sl = slice(start_frame, end_frame + 1)
    ear_y_segment, ear_x_segment = ear_y[sl].copy(), ear_x[sl].copy()
    shoulder_y_segment, shoulder_x_segment = shoulder_y[sl].copy(), shoulder_x[sl].copy()
    ankle_y_segment, ankle_x_segment = ankle_y[sl].copy(), ankle_x[sl].copy()
    torso_head_angle_segment = torso_head_angle[sl].copy()
    head_tilt_angle_segment = head_tilt_angle[sl].copy()

    plank_x_segment = ankle_x_segment - shoulder_x_segment
    plank_y_segment = ankle_y_segment - shoulder_y_segment
    head_x_segment = ear_x_segment - shoulder_x_segment
    head_y_segment = ear_y_segment - shoulder_y_segment

    plank_len = np.sqrt(plank_x_segment ** 2 + plank_y_segment ** 2)

    cross = plank_x_segment * head_y_segment - plank_y_segment * head_x_segment
    forward_norm = np.abs(cross) / (plank_len * plank_len)

    relative_bottom = bottom_frame - start_frame

    torso_head_angle_bottom = torso_head_angle_segment[relative_bottom]
    torso_head_angle_min = np.nanmin(torso_head_angle_segment)
    torso_head_angle_max = np.nanmax(torso_head_angle_segment)
    torso_head_angle_range = torso_head_angle_max - torso_head_angle_min
    torso_head_angle_mean = np.nanmean(torso_head_angle_segment)
    torso_head_q_10 = np.percentile(torso_head_angle_segment, 10)
    torso_head_q_25 = np.percentile(torso_head_angle_segment, 25)
    torso_head_q_50 = np.percentile(torso_head_angle_segment, 50)
    torso_head_q_75 = np.percentile(torso_head_angle_segment, 75)
    torso_head_q_90 = np.percentile(torso_head_angle_segment, 90)
    torso_head_std = np.nanstd(torso_head_angle_segment)

    head_forward_norm_bottom = forward_norm[relative_bottom]
    head_forward_norm_min = np.nanmin(forward_norm)
    head_forward_norm_max = np.nanmax(forward_norm)
    head_forward_norm_range = head_forward_norm_max - head_forward_norm_min
    head_forward_norm_mean = np.nanmean(forward_norm)
    head_forward_norm_q_10 = np.percentile(forward_norm, 10)
    head_forward_norm_q_25 = np.percentile(forward_norm, 25)
    head_forward_norm_q_50 = np.percentile(forward_norm, 50)
    head_forward_norm_q_75 = np.percentile(forward_norm, 75)
    head_forward_norm_q_90 = np.percentile(forward_norm, 90)
    head_forward_norm_std = np.nanstd(forward_norm)

    head_tilt_angle_bottom = head_tilt_angle_segment[relative_bottom]
    head_tilt_angle_min = np.nanmin(head_tilt_angle_segment)
    head_tilt_angle_max = np.nanmax(head_tilt_angle_segment)
    head_tilt_angle_range = head_tilt_angle_max - head_tilt_angle_min
    head_tilt_angle_mean = np.nanmean(head_tilt_angle_segment)
    head_tilt_q_10 = np.percentile(head_tilt_angle_segment, 10)
    head_tilt_q_25 = np.percentile(head_tilt_angle_segment, 25)
    head_tilt_q_50 = np.percentile(head_tilt_angle_segment, 50)
    head_tilt_q_75 = np.percentile(head_tilt_angle_segment, 75)
    head_tilt_q_90 = np.percentile(head_tilt_angle_segment, 90)
    head_tilt_std = np.nanstd(head_tilt_angle_segment)

    return HeadMetrics(torso_head_angle_bottom=torso_head_angle_bottom,
                       torso_head_angle_min=torso_head_angle_min,
                       torso_head_angle_max=torso_head_angle_max,
                       torso_head_angle_range=torso_head_angle_range,
                       torso_head_angle_mean=torso_head_angle_mean,
                       torso_head_angle_q_10=torso_head_q_10,
                       torso_head_angle_q_25 = torso_head_q_25,
                       torso_head_angle_q_50 = torso_head_q_50,
                       torso_head_angle_q_75 = torso_head_q_75,
                       torso_head_angle_q_90=torso_head_q_90,
                       torso_head_angle_std=torso_head_std,
                       head_forward_norm_bottom=head_forward_norm_bottom,
                       head_forward_norm_min=head_forward_norm_min,
                       head_forward_norm_max=head_forward_norm_max,
                       head_forward_norm_range=head_forward_norm_range,
                       head_forward_norm_mean=head_forward_norm_mean,
                       head_forward_norm_q_10=head_forward_norm_q_10,
                       head_forward_norm_q_25=head_forward_norm_q_25,
                       head_forward_norm_q_50=head_forward_norm_q_50,
                       head_forward_norm_q_75=head_forward_norm_q_75,
                       head_forward_norm_q_90=head_forward_norm_q_90,
                       head_forward_norm_std=head_forward_norm_std,
                       head_tilt_angle_bottom=head_tilt_angle_bottom,
                       head_tilt_angle_min=head_tilt_angle_min,
                       head_tilt_angle_max=head_tilt_angle_max,
                       head_tilt_angle_range=head_tilt_angle_range,
                       head_tilt_angle_mean=head_tilt_angle_mean,
                       head_tilt_angle_q_10=head_tilt_q_10,
                       head_tilt_angle_q_25=head_tilt_q_25,
                       head_tilt_angle_q_50=head_tilt_q_50,
                       head_tilt_angle_q_75=head_tilt_q_75,
                       head_tilt_angle_q_90=head_tilt_q_90,
                       head_tilt_angle_std=head_tilt_std)


def get_head(signal, visibility_scores, repetition) -> Dict[str, Any]:
    side = choose_side_visibility(visibility_scores)
    signal_args = ((signal['left_ear_x'], signal['left_ear_y'],
                   signal['left_shoulder_x'], signal['left_shoulder_y'],
                   signal['left_ankle_x'], signal['left_ankle_y'],
                   signal['left_torso_head_angle'],
                    signal['left_head_tilt_angle']) if side == "left" else
                   (signal['right_ear_x'], signal['right_ear_y'],
                    signal['right_shoulder_x'], signal['right_shoulder_y'],
                    signal['right_ankle_x'], signal['right_ankle_y'],
                    signal['right_torso_head_angle'],
                    signal['right_head_tilt_angle'])
                   )
    head = compute_head_metrics(signal_args, repetition['start_frame'], repetition['end_frame'],
                                 repetition['bottom_frame'])

    return {
        'torso_head_angle_bottom': head.torso_head_angle_bottom,
        'torso_head_angle_min': head.torso_head_angle_min,
        'torso_head_angle_max': head.torso_head_angle_max,
        'torso_head_angle_range': head.torso_head_angle_range,
        'torso_head_angle_mean': head.torso_head_angle_mean,
        'torso_head_angle_q_10': head.torso_head_angle_q_10,
        'torso_head_angle_q_25': head.torso_head_angle_q_25,
        'torso_head_angle_q_50': head.torso_head_angle_q_50,
        'torso_head_angle_q_75': head.torso_head_angle_q_75,
        'torso_head_angle_q_90': head.torso_head_angle_q_90,
        'torso_head_angle_std': head.torso_head_angle_std,
        'head_forward_norm_bottom': head.head_forward_norm_bottom,
        'head_forward_norm_min': head.head_forward_norm_min,
        'head_forward_norm_max': head.head_forward_norm_max,
        'head_forward_norm_range': head.head_forward_norm_range,
        'head_forward_norm_mean': head.head_forward_norm_mean,
        'head_forward_norm_q_10': head.head_forward_norm_q_10,
        'head_forward_norm_q_25': head.head_forward_norm_q_25,
        'head_forward_norm_q_50': head.head_forward_norm_q_50,
        'head_forward_norm_q_75': head.head_forward_norm_q_75,
        'head_forward_norm_q_90': head.head_forward_norm_q_90,
        'head_forward_norm_std': head.head_forward_norm_std,
        'head_tilt_angle_bottom': head.head_tilt_angle_bottom,
        'head_tilt_angle_min': head.head_tilt_angle_min,
        'head_tilt_angle_max': head.head_tilt_angle_max,
        'head_tilt_angle_range': head.head_tilt_angle_range,
        'head_tilt_angle_mean': head.head_tilt_angle_mean,
        'head_tilt_angle_q_10': head.head_tilt_angle_q_10,
        'head_tilt_angle_q_25': head.head_tilt_angle_q_25,
        'head_tilt_angle_q_50': head.head_tilt_angle_q_50,
        'head_tilt_angle_q_75': head.head_tilt_angle_q_75,
        'head_tilt_angle_q_90': head.head_tilt_angle_q_90,
        'head_tilt_angle_std': head.head_tilt_angle_std
    }