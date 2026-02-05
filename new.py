import numpy as np
from scipy.optimize import curve_fit

def mzi_transmission(I0, ph0, phase):
    return I0 * (1 + np.cos(phase + ph0)) / 2
