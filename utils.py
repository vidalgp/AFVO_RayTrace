import numpy as np
from obspy.signal.tf_misfit import cwt

def AVO(seismic, timelocmin, timelocmax, dts):
    if len(seismic.shape) == 3:
        zLen, xTraces = seismic.shape[:2]
    else:
        zLen = 1
        xTraces, ySamples = seismic.shape
        seismic = seismic.reshape([zLen, xTraces, ySamples])
    peakA = np.zeros(xTraces * zLen, dtype='float64')
    for z in range(zLen):
        for x in range(xTraces):
            ix = z * xTraces + x
            peakA[ix] = peak_amplitude(trace=seismic[z][x],\
                    loc=[timelocmin[ix], timelocmax[ix]], dt=dts)
    return peakA

def FVO(seismic, timelocmin, timelocmax, dts, wo=6):
    if len(seismic.shape) == 3:
        zLen, xTraces = seismic.shape[:2]
    else:
        zLen = 1
        xTraces, ySamples = seismic.shape
        seismic = seismic.reshape([zLen, xTraces, ySamples])
    pFreq = np.zeros(xTraces * zLen, dtype='float64')
    for z in range(zLen):
        for x in range(xTraces):
            ix = z * xTraces + x
            pFreq[ix] = peak_frequency(trace=seismic[z][x],\
                    loc=[timelocmin[ix], timelocmax[ix]], dt=dts, w0=wo)
    return pFreq

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

