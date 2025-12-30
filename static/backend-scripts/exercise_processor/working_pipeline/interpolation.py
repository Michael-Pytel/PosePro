import numpy as np

def cubic_interp(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2*p1) +
                  (-p0 + p2) * t +
                  (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                  (-p0 + 3*p1 - 3*p2 + p3) * t3)


def interpolate_nans(signal):
    y = np.array(signal, float, copy=True)
    known_idx = np.where(~np.isnan(y))[0]
    if known_idx.size == 0:
        return y

    first_known, last_known = known_idx[0], known_idx[-1]
    y[:first_known] = y[first_known]
    y[last_known+1:] = y[last_known]

    known_idx = np.where(~np.isnan(y))[0]
    for missing_i in np.where(np.isnan(y))[0]:
        left_known = known_idx[known_idx < missing_i][-1]
        right_known = known_idx[known_idx > missing_i][0]
        left_support = max(left_known - 1, known_idx[0])
        right_support = min(right_known + 1, known_idx[-1])
        t = (missing_i - left_known) / (right_known - left_known)
        y[missing_i] = cubic_interp(y[left_support], y[left_known], y[right_known], y[right_support], t)
    return y

# def interpolate_nans(signal):
#     nans = np.isnan(signal)
#     if nans.all():
#         return signal
#     x = np.arange(len(signal))
#     signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])
#     return signal