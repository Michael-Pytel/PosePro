import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def support_from_other_signals(signals, start: int, end: int):
    support = 0
    window = slice(start, end + 1)

    backup_signals = [
        'avg_hip_y',
        'chest_ground_distance', 'avg_elbow_angle',
        'avg_elbow_y', 'torso_angle'
    ]

    for s in backup_signals:
        if s not in signals:
            continue

        sig = np.array(signals[s])[window]
        if np.ptp(sig) < 1e-4:
            continue

        r = sig - np.min(sig)
        r = r / (np.max(sig) - np.min(sig) + 1e-6)

        if abs(np.max(r) - np.min(r)) < 0.25:
            continue
        support += 1

    return support >= 2

def detect_pushup_repetitions(signals, landmark_data, visibility_scores, fps):

    # Choosing main signal
    base_signal_candidates = [
        'left_shoulder_y',
        'right_shoulder_y'
    ]
    additional_candidates = [
        'left_elbow_angle',
        'right_elbow_angle',
    ]

    signal_for_landmarks = {'left_hip_y' : 23, 'right_hip_y' : 24, 'left_shoulder_y' : 11, 'right_shoulder_y' : 12}
    scores = {s: visibility_scores[signal_for_landmarks[s]] for s in base_signal_candidates}
    # scores2 = {s: np.ptp(signals[s]) for s in base_signal_candidates_2}
    # print(scores)
    # print(scores2)
    primary_signal = max(scores, key=scores.get)
    # primary_signal = 'right_elbow_angle'
    raw = np.array(-signals[primary_signal])

    sigma = fps * 0.08
    smoothed = gaussian_filter1d(raw, sigma=sigma)
    min_distance = int(fps * 0.5)

    peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=0.04)
    valleys, _ = find_peaks(-smoothed, distance=min_distance, prominence=0.04)

    # if len(peaks) == 0 or len(valleys) == 0:
    #     return []
    is_main_left = 11 if primary_signal == 'left_shoulder_y' else 12
    #if landmark_data[int(p)]["landmarks"][is_main_left]["visibility"] >= 0.1
    #if landmark_data[int(p)]["landmarks"].size != 0 and landmark_data[int(p)]["landmarks"][is_main_left]["visibility"] >= 0.5
    events = [('peak', int(p)) for p in peaks ] + \
             [('valley', int(v)) for v in valleys]
    events.sort(key=lambda x: x[1])

    print(events)
    repetitions = []
    rep_id = 1
    if len(events) == 1 and events[0][0] == 'valley':
        mid = events[0][1]
        lookback_start, lookback_end = min(100, mid), min(100, len(smoothed) - mid - 1)
        gradient_start, gradient_end = np.diff(smoothed[mid - lookback_start: mid]), np.diff(smoothed[mid: mid + lookback_end])
        sign_changes_start, sign_changes_end = np.diff(np.sign(gradient_start)), np.diff(np.sign(gradient_end))
        change_indices_start, change_indices_end = np.where(sign_changes_start != 0)[0], np.where(sign_changes_end != 0)[0]
        if change_indices_start.size != 0:
            start = mid - lookback_start + change_indices_start[-1] + 2
        else:
            start = mid - lookback_start
        if change_indices_end.size != 0:
            end = mid + change_indices_end[0]
        else:
            end = mid + lookback_end

        repetitions.append({
            'rep_id': rep_id,
            'start_frame': int(start),
            'end_frame': int(end),
            'start_time': start / fps,
            'end_time': end / fps,
            'duration': (end - start) / fps,
            'signal_used': primary_signal,
        })
    else:
        i = 0
        while i <= len(events) - 2:
            e1, e2 = events[i], events[i + 1]
            if i != len(events) - 2:
                e3 = events[i + 2]
                if (e1[0], e2[0], e3[0]) == ("peak", "valley", "peak"):
                    start = e1[1]
                    mid = e2[1]
                    end = e3[1]
                elif (e1[0], e2[0]) == ("valley", "peak") and len(repetitions) == 0:
                    mid = e1[1]
                    end = e2[1]
                    lookback = min(100, mid)
                    gradient = np.diff(smoothed[mid - lookback : mid])
                    sign_changes = np.diff(np.sign(gradient))
                    change_indices = np.where(sign_changes != 0)[0]
                    if change_indices.size != 0:
                        start = mid - lookback + change_indices[-1] + 2
                    else:
                        start = mid - lookback
                else:
                    i += 1
                    continue
            elif (e1[0], e2[0]) == ("peak", "valley"):
                start = e1[1]
                mid = e2[1]
                lookback = min(100, len(smoothed) - mid - 1)
                gradient = np.diff(smoothed[mid: mid + lookback])
                sign_changes = np.diff(np.sign(gradient))
                change_indices = np.where(sign_changes != 0)[0]
                if change_indices.size != 0:
                    end = mid + change_indices[0]
                else:
                    end = mid + lookback
            else:
                i += 1
                continue

            dur = (end - start) / fps
            if not (0.6 <= dur <= 6):
                i += 1
                continue

            up = smoothed[end] if len(repetitions) == 0 else smoothed[start]
            down = smoothed[mid]
            amp = abs(up - down)

            # Amplitude must be rational
            if amp < 0.18 * min(np.ptp(smoothed[~np.isnan(smoothed)]), 0.8):
                i += 1
                continue
            if not support_from_other_signals(signals, start, end):
                i += 1
                continue

            repetitions.append({
                'rep_id': rep_id,
                'start_frame': int(start),
                'bottom_frame': int(mid),
                'end_frame': int(end),
                'start_time': start / fps,
                'end_time': end / fps,
                'duration': dur,
                'signal_used': primary_signal,
            })

            rep_id += 1
            i += 1 if len(repetitions) in {0, 1} else 2
    print(repetitions)
    return repetitions