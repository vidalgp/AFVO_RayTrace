#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: wedgefunctions.py
#  Purpose: A.FVO wedge Analysis
#   Author: Vidal Gonzalez P
#    Email: vidalgonz8@gmail.com
# --------------------------------------------------------------------
"""
Code to analyze synthetic seismic data considering AVO in a \
        wedge model using CWT, for amplitude and frequency analysis.
"""
__author__ = "Vidal Gonzalez P"
__email__ = "vidalgonz8@gmail.com"
__status__ = "En Desarrollo"

import sys
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

def digitize_top_base(Rs, model, dt, iAngle, t, interf):
    vp = model.vp[:-1]
    vs = model.vs[:-1]
    rho = model.rho[:-1]
    if (not np.isnan(t) or t):
        t_ix = int(t / dt)
        Rs.add_layer_reflection(vp, vs, rho, iAngle, t_ix, interf)
    else:
        pass

def alpha_tr(angle_in, V1, V2):
    return np.arcsin(np.sin(angle_in) * V2 / V1)

def simple_offset(angle_in, topDepth, dh, V1, V2):
    return topDepth * np.tan(angle_in) + dh*np.tan(alpha_tr(angle_in, V1, V2))

def theta_equivalent(angle_in, topDepth, dh, V1, V2):
    return np.arctan(0.5 * simple_offset(angle_in, topDepth, dh, V1, V2) / topDepth)

def cmp_gather_simple(dh, maxAng, step, topDepth, velocities):
    '''
    shot gather equivalent to cmp gather considering flat paralell reflectors
    '''
    import warnings
    warnings.filterwarnings("error")
    
    v1, v2 = velocities[0], velocities[1] #m/s

    try:
        radmax_downwards = min(np.arcsin(v1/v2), np.radians(maxAng))
    except:
        radmax_downwards = np.radians(maxAng)

    try:
        radmax_upwards = min(np.arcsin(v2/v1), alpha_tr(radmax_downwards, v1, v2))
    except:
        radmax_upwards = alpha_tr(radmax_downwards, v1, v2)
    
    print(('Max down: {} - Max up: {}').format(radmax_downwards, radmax_upwards))
    
    Angles_in = np.zeros(0, dtype='float')
    Angles_top = np.zeros(0, dtype='float')
    Angles_base = np.zeros(0, dtype='float')

    i = 0
    while True:
        rad_in = np.radians(i * step)
        alpha = alpha_tr(rad_in, v1, v2)
        theta = theta_equivalent(rad_in, topDepth, dh, v1, v2)
        Angles_in = np.append(Angles_in, rad_in)
        Angles_base = np.append(Angles_base, alpha)
        Angles_top = np.append(Angles_top, theta)
        if ((theta >= radmax_downwards) or (alpha >= radmax_upwards)):
            break
        i += 1
    
    RayPath_top =  2 * topDepth / np.cos(Angles_top)
    RayPath_base1 = 2 * topDepth / np.cos(Angles_in)
    RayPath_base2 = 2 * dh / np.cos(Angles_base)
    RayPath_base =  RayPath_base1 + RayPath_base2   
    TopTime = RayPath_top / v1
    BaseTime = ((RayPath_base1 / v1) + (RayPath_base2 / v2))
    return Angles_top, Angles_base, RayPath_top, RayPath_base, TopTime, BaseTime

def create_timeModel(seismic, model, dtc, Theta, tBottom, tTop, Q=False):
    print('\n\n Synthetic elastic TIME wedge calculations\n\n')
    FreqVel = model.vp[0]
    for z in range(seismic.zLen):
        print(('Computing dh #{}').format(z))
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


###################################################################################

def amplitudes_calc(seismic, timelocmin, timelocmax):
    peakA = np.zeros(seismic.xTraces * seismic.zLen, dtype='float')
    for z in range(seismic.zLen):
        print(('Computing attributes for shot #{}').format(z))
        for x in range(seismic.xTraces):
            ix = z*seismic.xTraces+x
            peakA[ix] = peak_amplitude(trace=seismic.get_amplitude(x, z),\
                    loc=[timelocmin[ix], timelocmax[ix]], dt=seismic.dt)
    return peakA

def spectral_calc(seismic, timelocmin, timelocmax):
    pFreq = np.zeros(seismic.xTraces * seismic.zLen, dtype='float')
    for z in range(seismic.zLen):
        print(('Computing attributes for shot #{}').format(z))
        for x in range(seismic.xTraces):
            ix = z*seismic.xTraces+x
            pFreq[ix] = peak_frequency(trace=seismic.get_amplitude(x, z),\
                    loc=[timelocmin[ix], timelocmax[ix]], dt=seismic.dt, w0=6)
    return pFreq

def simple_array_maker(model, dhmax, dhstep, angMax, angStep, topDepth):
    angMin = 0
    dhmin = 0.5
    dhVec = np.arange(dhmin, dhmax+dhstep, dhstep)
    print("Vector DH:")
    print(dhVec)
    angVec = np.arange(0, angMax, angStep)
    for i in range(dhVec.shape[0]):
        th, be, ru, rl, tt, tb = cmp_gather_simple(dhVec[i], angMax, angStep, \
                topDepth, model.vp)
        if i == 0:
            sizeX = th.shape[0]
            TH = th
            BE = be
            RU = ru
            RL = rl
            TT = tt
            TB = tb
            DH = dhVec[i] * np.ones(th.shape[0], dtype='float')
        else:
            aux = np.zeros(sizeX, dtype='float')
            aux[:th.shape[0]] = th
            TH = np.vstack([TH, aux])
            aux[:be.shape[0]] = be
            BE = np.vstack([BE, aux])
            aux[:ru.shape[0]] = ru
            RU = np.vstack([RU, aux])
            aux[:rl.shape[0]] = rl
            RL = np.vstack([RL, aux])
            aux[:tt.shape[0]] = tt
            TT = np.vstack([TT, aux])
            aux[:tb.shape[0]] = tb
            TB = np.vstack([TB, aux])
            aux[:th.shape[0]] = dhVec[i] * np.ones(th.shape[0], dtype='float')
            DH = np.vstack([DH, aux])
            del(aux)
    return TH, BE, RU, RL, TT, TB, DH

###################################################################################

def main():
    import time
    start = time.time()
    if len(sys.argv) > 1:
        mtype, Aq = sys.argv[1], sys.argv[2]
    
    mod = Model(mtype)
    dt = 0.0001 #ms
    topdepth = 2005
    angmax = 40
    angstep = 1
    dhmax = 50
    dhstep = 1.5 
    #global TH, B, RU, RL, TT, TB, DH
    TH, B, RU, RL, TT, TB, DH= simple_array_maker(mod, dhmax, dhstep, angmax, angstep, \
            topdepth)

    dimX = TH.shape[1]
    dimY = int(TB[-1].max()/dt * (1.10))
    dimZ = TH.shape[0]

    global seismik, ymin, ymax
    ymax = dimY*dt
    ymin = TT[0].min()* 0.95
    seismik = Seismic(dt, dimX, dimY, dimZ)
    create_timeModel(seismik, mod, dt, TH, TB, TT, Aq)
    
    Tmin = TT - 0.1
    Tmax = 0.5 * (TT + TB)
    Bmin = Tmax
    Bmax = TB + 0.1
    
    for dh in range(0, 31, 3):
        seismik.plot_seismogram(ymin=ymin, ymax=ymax, excursion=3, z=dh)
        plot_AFVO(seismik._SEIS[dh], np.degrees(TH[dh]), Tmin[dh], Tmax[dh], Bmin[dh], \
                Bmax[dh], seismik.dt,('TopBase_{}').format(dh*dhstep+0.5))
        plt.close('all') 
    
    dh = seismik.zLen-1
    seismik.plot_seismogram(ymin=ymin, ymax=ymax, excursion=3, z=dh)
    plot_AFVO(seismik._SEIS[dh], np.degrees(TH[dh]), Tmin[dh], Tmax[dh], Bmin[dh], \
            Bmax[dh], seismik.dt,('TopBase_{}').format(dh*dhstep))
    
    #global fullArray
    totalTraces = seismik.zLen * seismik.xTraces
    fullArray = np.zeros([totalTraces, 4], dtype='float') 
    tt = TT.reshape(totalTraces)
    tb = TB.reshape(totalTraces)
    theta = np.degrees(TH.reshape(totalTraces))
    dh = DH.reshape(totalTraces) 
    
    #top
    tminT = tt - 0.1
    tmaxT = 0.5 * (tt + tb)

    fullArray.T[0] = theta
    fullArray.T[1] = dh
    fullArray.T[2] = amplitudes_calc(seismik, tminT, tmaxT)
    fullArray.T[3] = spectral_calc(seismik, tminT, tmaxT)

    fullArray = fullArray[~np.isnan(fullArray).any(axis=1)]

    xmin = np.floor(fullArray.T[1].min())
    xmax = np.ceil(fullArray.T[1].max())
    ymin = np.floor(fullArray.T[0].min())
    ymax = np.ceil(fullArray.T[0].max())
    
    plot_map(fullArray.T[1], fullArray.T[0], fullArray.T[2], xmin, xmax, ymin, ymax,\
            ['dhT','angleT'], 'amp', 'BsimpleTop')
    plot_map(fullArray.T[1], fullArray.T[0], fullArray.T[3], xmin, xmax, ymin, ymax,\
            ['dhT','angleT'], 'freq', 'BsimpleTop')

    #base
    tminB = 0.5 * (tb + tt)
    tmaxB = tb + 0.1 
    
    fullArray.T[2] = amplitudes_calc(seismik, tminB, tmaxB)
    fullArray.T[3] = spectral_calc(seismik, tminB, tmaxB)

    fullArray = fullArray[~np.isnan(fullArray).any(axis=1)]

    plot_map(fullArray.T[1], fullArray.T[0], fullArray.T[2], xmin, xmax, ymin, ymax,\
            ['dhT','angleT'], 'amp', 'BsimpleBase')
    plot_map(fullArray.T[1], fullArray.T[0], fullArray.T[3], xmin, xmax, ymin, ymax,\
            ['dhT','angleT'], 'freq', 'BsimpleBase')
 
    end = time.time()
    print(("Elapsed time {}s").format(end - start))

    plt.close('all') 
    return 0

if (__name__ == '__main__'):
    main()
