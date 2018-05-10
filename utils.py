import numpy as np
from obspy.signal.tf_misfit import cwt
from seisclass import*
from simplemodel import digitize_top_base
from wedgemodel import digitize_wedge


def create_timewedgeModel(seismic, model, dtc, Theta, Beta, tBottom, tTop, wedgeAngle, Q=False):
    print('\n\n Synthetic wedge time model calculations\n\n')
    FreqVel = model.vp[0]
    for z in range(seismic.zLen):
        print(('Computing cdp #{} of {}').format(z+1, seismic.zLen))
        for x in range(seismic.xTraces):
            R = ReflectivityS( ns = seismic.ySamples )
            #top reflector
            digitize_wedge(R, model, dtc, Theta[z][x], Beta[z][x], tTop[z][x], 'top', wedgeAngle)
            Wv = Wavelet(wtype='bp', wf=[5, 10, 40, 80], duration=0.28, wdt=seismic.dt)
            #Wv = Wavelet(wtype='r', wf=75)
            if Q:
                Wv.apply_Q(tTop[z][x], FreqVel)
            trace = Trace(wavelet = Wv, rseries = R.rserie)
            #base reflector
            R = ReflectivityS( ns = seismic.ySamples )
            digitize_wedge(R, model, dtc, Theta[z][x], Beta[z][x], tBottom[z][x], 'base', wedgeAngle)
            Wv = Wavelet(wtype='bp', wf=[5, 10, 40, 80], duration=0.28, wdt=seismic.dt)
            #Wv = Wavelet(wtype='r', wf=75)
            if Q:
                Wv.apply_Q(tBottom[z][x], FreqVel)
            traceB = Trace(wavelet = Wv, rseries = R.rserie)
            trace+= traceB
            #calculation made on individual trace are made here
            seismic.add_trace(trace, x, z)

def create_timeModel(seismic, model, dtc, Theta, tBottom, tTop, Q=False):
    print('\n\n Synthetic simple time model calculations\n\n')
    FreqVel = model.vp[0]
    for z in range(seismic.zLen):
        print(('Computing dh #{} of {}').format(z+1, seismic.zLen))
        for x in range(seismic.xTraces):
            R = ReflectivityS( ns = seismic.ySamples )
            #top reflector
            digitize_top_base(R, model, dtc, Theta[z][x], tTop[z][x], 'top')
            Wv = Wavelet(wtype='bp', wf=[5, 10, 40, 80], duration=0.28, wdt=seismic.dt)
            #Wv = Wavelet(wtype='r', wf=75)
            if Q:
                Wv.apply_Q(tTop[z][x], FreqVel)
            trace = Trace(wavelet = Wv, rseries = R.rserie)
            #base reflector
            R = ReflectivityS( ns = seismic.ySamples )
            digitize_top_base(R, model, dtc, Theta[z][x], tBottom[z][x], 'base')
            Wv = Wavelet(wtype='bp', wf=[5, 10, 40, 80], duration=0.28, wdt=seismic.dt)
            #Wv = Wavelet(wtype='r', wf=75)
            if Q:
                Wv.apply_Q(tBottom[z][x], FreqVel)
            traceB = Trace(wavelet = Wv, rseries = R.rserie)
            trace+= traceB
            #calculation made on individual trace are made here
            seismic.add_trace(trace, x, z)


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
                    loc=np.asarray([timelocmin[ix], timelocmax[ix]]), dt=dts)
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
                    loc=np.asarray([timelocmin[ix], timelocmax[ix]]), dt=dts, w0=wo)
    return pFreq

def peak_frequency(trace, loc, dt, fmin=1, fmax=90, w0=6):
    if not all(loc>0):
        return 0.0
    else:
        dsl = int(loc[0] / dt)
        dsh = int(loc[1] / dt)
        scalogram = cwt(trace[dsl:dsh], dt, w0, fmin, fmax)
        nf = np.logspace(np.log10(fmin), np.log10(fmax))
        idx = np.argmax(np.abs(scalogram), axis = None)
        idxmax = np.unravel_index(idx, scalogram.shape)
        return nf[idxmax[0]//2]

def peak_amplitude(trace, loc, dt):
    if not all(loc>0):
        return 0.0
    else:
        dsl = int(loc[0] / dt)
        dsh = int(loc[1] / dt)
        return max(trace[dsl:dsh].max(), trace[dsl:dsh].min(), key=abs)

