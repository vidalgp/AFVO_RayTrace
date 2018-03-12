import numpy as np
from seisclass import *
from plots import *
from obspy.signal.tf_misfit import cwt

def AVO(gather, tmin, tmax, dt):
    Ri = np.zeros(gather.shape[0], dtype='float64')
    for i, trace in enumerate(gather):
        Ri[i] = peak_amplitude(trace, [tmin[i], tmax[i]], dt)
    return Ri

def FVO(gather, tmin, tmax, dt):
    Fi = np.zeros(gather.shape[0], dtype='float64')
    for i, trace in enumerate(gather):
        Fi[i] = peak_frequency(trace, [tmin[i], tmax[i]], dt)
    return Fi

def peak_frequency(trace, loc, dt, fmin=1, fmax=90, w0=6):
    dsl = int(loc[0] / dt)
    dsh = int(loc[1] / dt)
    scalogram = cwt(trace[dsl:dsh], dt, w0, fmin, fmax)
    nf = np.logspace(np.log10(fmin), np.log10(fmax))
    idx = np.argmax(np.abs(scalogram), axis = None)
    idxmax = np.unravel_index(idx, scalogram.shape)
    return nf[idxmax[0]//2]

def peak_amplitude(trace, loc, dt):
    dsl = int(loc[0] / dt)
    dsh = int(loc[1] / dt)
    return max(trace[dsl:dsh].max(), trace[dsl:dsh].min(), key=abs)

