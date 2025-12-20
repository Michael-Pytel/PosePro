import numpy as np

def compute_visibility_scores(landmarks_data, key_points):
    scores = {}

    # Przygotowanie tablic visibility i presence
    for kp in key_points:
        vis = []
        missing = 0
        xs = []
        ys = []

        for frame in landmarks_data:
            lm = frame["landmarks"]
            if kp not in lm:
                missing += 1
                continue
            
            vis.append(lm[kp].get("visibility", 0))
            xs.append(lm[kp]["x"])
            ys.append(lm[kp]["y"])

        total_frames = len(landmarks_data)

        if len(vis) == 0:
            scores[kp] = 0
            continue

        mean_visibility = np.mean(vis)
        missing_ratio = missing / total_frames

        # Noise (opcjonalnie)
        if len(xs) > 2:
            dx = np.diff(xs)
            dy = np.diff(ys)
            noise = (np.std(dx) + np.std(dy)) / 2
        else:
            noise = 0

        # Final score
        score = mean_visibility * (1 - missing_ratio) / (1 + noise)

        scores[kp] = score

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
