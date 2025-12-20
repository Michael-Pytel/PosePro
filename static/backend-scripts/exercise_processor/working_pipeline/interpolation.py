import numpy as np

def cubic_interp(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2*p1) +
                  (-p0 + p2) * t +
                  (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                  (-p0 + 3*p1 - 3*p2 + p3) * t3)

def interpolate_nans(signal):
    y = signal.copy()
    n = len(y)

    idx = np.arange(n)
    mask = ~np.isnan(y)
    known = idx[mask]
    for i in np.where(~mask)[0]:
        lefti = known[known < i][-1]
        righti = known[known > i][0]

        left2 = max(lefti - 1, known[0])
        right2 = min(righti + 1, known[-1])

        # pobieramy wartości
        p0 = y[left2]
        p1 = y[lefti]
        p2 = y[righti]
        p3 = y[right2]

        # Normalized parameter
        t = (i - lefti) / (righti - lefti)
        y[i] = cubic_interp(p0, p1, p2, p3, t)

    return y

# def interpolate_nans(signal):
#     nans = np.isnan(signal)
#     if nans.all():
#         return signal
#     x = np.arange(len(signal))
#     signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])
#     return signal