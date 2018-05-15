#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
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

import numpy as np
from seisclass import*


def create_timeModel(seismic, model, dtc, Theta, tBottom, tTop, Q=False):
    print('\n\n Cálculos del modelo de cuña sintético\n\n')
    FreqVel = model.vp[0]
    for z in range(seismic.zLen):
        print(('DH #{} de {}').format(z+1, seismic.zLen))
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


def alpha_tr(angle_in, V1, V2):
    return np.arcsin(np.sin(angle_in) * V2 / V1)

def simple_offset(angle_in, topDepth, dh, V1, V2):
    return topDepth * np.tan(angle_in) + dh*np.tan(alpha_tr(angle_in, V1, V2))

def theta_equivalent(angle_in, topDepth, dh, V1, V2):
    return np.arctan(0.5 * simple_offset(angle_in, topDepth, dh, V1, V2) / topDepth)

def cmp_gather_simple(dh, maxAng, step, topDepth, velocities, critical=False):
    '''
    shot gather equivalent to cmp gather considering flat paralell reflectors
    critical: consider critical angles in ray tracing 
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
        if critical:
            if ((theta >= radmax_downwards) or (alpha >= radmax_upwards)):
                break
        else:    
            try:
                if ((theta >= np.radians(maxAng)) or (rad_in >= np.radians(maxAng))):
                    break
            except:
                break
        i += 1
    
    RayPath_top =  2 * topDepth / np.cos(Angles_top)
    RayPath_base1 = 2 * topDepth / np.cos(Angles_in)
    RayPath_base2 = 2 * dh / np.cos(Angles_base)
    RayPath_base =  RayPath_base1 + RayPath_base2   
    TopTime = RayPath_top / v1
    BaseTime = ((RayPath_base1 / v1) + (RayPath_base2 / v2))
    return Angles_top, Angles_base, RayPath_top, RayPath_base, TopTime, BaseTime

def digitize_top_base(Rs, model, dt, iAngle, t, interf):
    vp = model.vp[:-1]
    vs = model.vs[:-1]
    rho = model.rho[:-1]
    if (t>0):
        t_ix = int(t / dt)
        Rs.add_layer_reflection(vp, vs, rho, iAngle, t_ix, interf)
    else:
        pass

def simple_array_maker(model, dhmin, dhmax, dhstep, angMax, angStep, topDepth):
    angMin = 0
    dhVec = np.arange(dhmin, dhmax+dhstep, dhstep)
    print("Vector DH:")
    print(dhVec)
    spanSize = np.zeros(dhVec.size, dtype='int')
    for i in range(dhVec.size):
        th, be, ru, rl, tt, tb = cmp_gather_simple(dhVec[i], angMax, angStep, \
                topDepth, model.vp)
        spanSize[i] = th.size
        if i == 0:
            sizeX = th.size
            TH = th
            BE = be
            RU = ru
            RL = rl
            TT = tt
            TB = tb
            DH = dhVec[i] * np.ones(th.shape[0], dtype='float')
        else:
            aux = th[-1] * np.ones(sizeX, dtype='float')
            aux[:th.shape[0]] = th
            TH = np.vstack([TH, aux])
            del(aux)
            
            aux = be[-1] * np.ones(sizeX, dtype='float')
            aux[:be.shape[0]] = be
            BE = np.vstack([BE, aux])
            del(aux)
            
            aux = ru[-1] * np.ones(sizeX, dtype='float')
            aux[:ru.shape[0]] = ru
            RU = np.vstack([RU, aux])
            del(aux)
            
            aux = rl[-1] * np.ones(sizeX, dtype='float')
            aux[:rl.shape[0]] = rl
            RL = np.vstack([RL, aux])
            del(aux)
            
            aux = tt[-1] * np.ones(sizeX, dtype='float')
            aux[:tt.shape[0]] = tt
            TT = np.vstack([TT, aux])
            del(aux)
            
            aux = tb[-1] * np.ones(sizeX, dtype='float')
            aux[:tb.shape[0]] = tb
            TB = np.vstack([TB, aux])
            del(aux)
            
            aux = dhVec[i] * np.ones(sizeX, dtype='float')
            DH = np.vstack([DH, aux])
            del(aux)

    return TH, BE, RU, RL, TT, TB, DH, spanSize

###################################################################################
def main():
    pass

if (__name__ == '__main__'):
    main()

