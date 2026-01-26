import numpy as np

def compute_visibility_scores(landmarks_data, key_points):
    scores = {}

    for kp in key_points:
        vis = []
        missing = 0
        xs = []
        ys = []

        for frame in landmarks_data:
            lm = frame["landmarks"]
            lm_len = len(lm)
            if kp >= lm_len:
                missing += 1
                continue

            vis.append(lm[kp].get("visibility", 0))
            xs.append(lm[kp]["x"])
            ys.append(lm[kp]["y"])

        if len(vis) == 0:
            scores[kp] = 0
            continue

        scores[kp] = np.mean(vis)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
