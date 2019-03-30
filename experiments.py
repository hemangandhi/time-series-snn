import numpy as np
import matplotlib.pyplot as plt

class Sine:
    def __init__(self, amplitude, freq, phase_shift, y_shift):
        self.a = amplitude
        self.p = freq
        self.s = phase_shift
        self.y = y_shift
    def __call__(self, arg):
        return np.sin(self.p * arg + self.s) * self.a + self.y
    def time_series(self, start, stop, num):
        return self(np.linspace(start, stop, num=num))
    def period(self):
        return 2 * np.pi / self.p

#THIS MIGHT BE POINTLESS
def poincare_map(ts, peak=True):
    max_val = min_val = -1
    pv = -1
    nv = -1
    for i, v in enumerate(ts):
        i_is_max = (i == 0 or ts[i - 1] < ts[i]) and (i == len(ts) - 1 or ts[i + 1] > ts[i])
        i_is_min = (i == 0 or ts[i - 1] > ts[i]) and (i == len(ts) - 1 or ts[i + 1] < ts[i])
        if i_is_max:
            if peak:
                pv = nv
                nv = i - max_val
            max_val = i
        if i_is_min:
            if not peak:
                pv = nv
                nv = i - min_val
            min_val = i
        yield pv, nv
