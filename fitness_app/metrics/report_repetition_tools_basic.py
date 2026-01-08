import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass

@dataclass(frozen=True)
class TimingMetrics:
    down_time_s: float
    up_time_s: float
    rep_time_s: float
    bottom_pause_s: float
    pause_start_frame: Optional[int] = None
    pause_end_frame: Optional[int] = None

@dataclass(frozen=True)
class RangeOfMotionMetrics:
    max_elbow_angle: float
    min_elbow_angle: float
    mean_elbow_angle: float
    elbow_angle_q_10: float
    elbow_angle_q_25: float
    elbow_angle_q_50: float
    elbow_angle_q_75: float
    elbow_angle_q_90: float
    elbow_angle_std: float
    range_of_motion: float
    full_depth: bool
    full_lockout: bool
    is_rep_full: bool


def choose_elbow_visibility(visibility_scores):
    left_mean = np.mean([visibility_scores[11], visibility_scores[13], visibility_scores[15]])
    right_mean = np.mean([visibility_scores[12], visibility_scores[14], visibility_scores[16]])
    return "left_elbow_angle" if left_mean > right_mean else "right_elbow_angle"

def continous_block_by_mask(mask, frame):
    start_frame = frame
    while start_frame > 0 and mask[start_frame - 1]:
        start_frame -= 1
    end_frame = frame
    while end_frame < mask.size - 1 and mask[end_frame + 1]:
        end_frame += 1
    return start_frame, end_frame

def compute_timing_metrics(elbow_angle_signal, start_frame, end_frame, bottom_frame, fps):
    down_time_s = (bottom_frame - start_frame) / fps
    up_time_s = (end_frame - bottom_frame) / fps
    rep_time_s = (end_frame - start_frame) / fps

    elbow_segment = elbow_angle_signal[start_frame : end_frame + 1]
    min_elbow_angle, min_elbow_angle_frame = np.nanmin(elbow_segment), np.nanargmin(elbow_segment)
    angle_threshold = 5

    angle_mask_min = elbow_segment <= (min_elbow_angle + angle_threshold)
    block_min = continous_block_by_mask(angle_mask_min, min_elbow_angle_frame)

    if block_min == (min_elbow_angle_frame, min_elbow_angle_frame):
        bottom_pause_s = 0.0
        pause_start_frame = start_frame + min_elbow_angle_frame
        pause_end_frame = start_frame + min_elbow_angle_frame
    else:
        (pause_start_frame_relative, pause_end_frame_relative) = block_min
        bottom_pause_s = (pause_end_frame_relative - pause_start_frame_relative) / fps

        pause_start_frame = start_frame + pause_start_frame_relative
        pause_end_frame = start_frame + pause_end_frame_relative

    return TimingMetrics(down_time_s = float(down_time_s),
                         up_time_s = float(up_time_s),
                         rep_time_s = float(rep_time_s),
                         bottom_pause_s = float(bottom_pause_s),
                         pause_start_frame = pause_start_frame,
                         pause_end_frame = pause_end_frame)

def compute_rom_metrics(elbow_angle_signal, start_frame, end_frame, bottom_frame, fps):
    elbow_segment = elbow_angle_signal[start_frame : end_frame + 1]
    maximum_elbow_angle = np.nanmax(elbow_segment)
    minimum_elbow_angle = np.nanmin(elbow_segment)

    mean_elbow_angle = np.nanmean(elbow_segment)
    elbow_angle_q_10 = np.nanpercentile(elbow_segment, 10)
    elbow_angle_q_25 = np.nanpercentile(elbow_segment, 25)
    elbow_angle_q_50 = np.nanpercentile(elbow_segment, 50)
    elbow_angle_q_75 = np.nanpercentile(elbow_segment, 75)
    elbow_angle_q_90 = np.nanpercentile(elbow_segment, 90)
    elbow_angle_std = np.nanstd(elbow_segment)

    range_of_motion = float(maximum_elbow_angle - minimum_elbow_angle)
    full_depth = minimum_elbow_angle <= 95
    full_lockout = maximum_elbow_angle >= 160

    is_rep_full = full_depth and full_lockout

    return RangeOfMotionMetrics(max_elbow_angle = maximum_elbow_angle,
                                min_elbow_angle= minimum_elbow_angle,
                                mean_elbow_angle=mean_elbow_angle,
                                elbow_angle_q_10 = elbow_angle_q_10,
                                elbow_angle_q_25 = elbow_angle_q_25,
                                elbow_angle_q_50 = elbow_angle_q_50,
                                elbow_angle_q_75 = elbow_angle_q_75,
                                elbow_angle_q_90 = elbow_angle_q_90,
                                elbow_angle_std= elbow_angle_std,
                                range_of_motion = range_of_motion,
                                full_depth = full_depth,
                                full_lockout = full_lockout,
                                is_rep_full = is_rep_full)

def get_timing(signal, visibility_scores, repetition, fps) -> Dict[str, Any]:

    which_signal = choose_elbow_visibility(visibility_scores)
    
    timing = compute_timing_metrics(
        signal[which_signal],
        repetition["start_frame"],
        repetition["end_frame"],
        repetition["bottom_frame"],
        fps
    )
    
    return {
        'down_time': timing.down_time_s,
        'up_time': timing.up_time_s,
        'rep_time': timing.rep_time_s,
        'bottom_pause': timing.bottom_pause_s,
        'pause_frames': [timing.pause_start_frame, timing.pause_end_frame] 
                       if timing.pause_start_frame is not None else None
    }


def get_rom(signal, visibility_scores, repetition, fps) -> Dict[str, Any]:
    
    which_signal = choose_elbow_visibility(visibility_scores)
    
    rom = compute_rom_metrics(
        signal[which_signal],
        repetition["start_frame"],
        repetition["end_frame"],
        repetition["bottom_frame"],
        fps
    )
    
    return {
        'max_elbow_angle': rom.max_elbow_angle,
        'min_elbow_angle': rom.min_elbow_angle,
        'mean_elbow_angle': rom.mean_elbow_angle,
        'elbow_angle_q_10': rom.elbow_angle_q_10,
        'elbow_angle_q_25': rom.elbow_angle_q_25,
        'elbow_angle_q_50': rom.elbow_angle_q_50,
        'elbow_angle_q_75': rom.elbow_angle_q_75,
        'elbow_angle_q_90': rom.elbow_angle_q_90,
        'elbow_angle_std': rom.elbow_angle_std,
        'range_of_motion': rom.range_of_motion,
        'full_depth': rom.full_depth,
        'full_lockout': rom.full_lockout,
        'is_complete_rep': rom.is_rep_full
    }