import numpy as np
from seisclass import Model

def Xmin(angmax, topDepth):
    return -np.tan(angmax) * topDepth

def Xmax(angmax, topDepth, gamma, dhmax):
    return topDepth * np.tan(angmax) + dhmax / np.tan(gamma) 

def XonWedge(angle_in, topDepth, Xsrc):
    xw = Xsrc + topDepth * np.tan(angle_in)
    if xw >= -0.1 and xw < 0:
        return 0.0
    elif xw >= 0:
        return xw
    else:
        print(xw, np.degrees(angle_in), Xsrc, topDepth)
        raise ValueError('Something went wrong on XonWedge')
        return -1

def DHtop(angle_in, topDepth, gamma, Xsrc):
    return XonWedge(angle_in, topDepth, Xsrc) * np.tan(gamma)

def AlphaTr(angle_in, V1, V2):
    return np.arcsin(np.sin(angle_in) * V2 / V1)

def BetaBase(angle_in, gamma, V1, V2):
    return AlphaTr(angle_in, V1, V2) + gamma

def PsiTr(angle_in, gamma, V1, V2):
    return np.arcsin(np.sin(DeltaUp(angle_in, gamma, V1, V2)) * V1/V2)

def DeltaUp(angle_in, gamma, V1, V2):
    return 2 * BetaBase(angle_in, gamma, V1, V2) - AlphaTr(angle_in, V1, V2)

def P1(angle_in, topDepth):
    return topDepth / np.cos(angle_in)

def P2(angle_in, topDepth, gamma, Xsrc, V1, V2):
    return ( np.cos(gamma) * DHtop(angle_in, topDepth, gamma, Xsrc) / \
            np.cos(BetaBase(angle_in, gamma, V1, V2)) )

def XbaseWedge(angle_in, topDepth, gamma, Xsrc, V1, V2):
    xb = XonWedge(angle_in, topDepth, Xsrc) + np.sin(AlphaTr(angle_in, V1, V2)) * \
            P2(angle_in, gamma, topDepth, Xsrc, V1, V2)
    if xb >= -0.1 and xb < 0:
        return 0.0
    elif xb >= 0:
        return xb
    else:
        raise ValueError('Something went wrong on XbaseWedge')
        return -1

def DHbase(angle_in, topDepth, gamma, Xsrc, V1, V2):
    return XbaseWedge(angle_in, topDepth, gamma, Xsrc, V1, V2) * np.tan(gamma)

def P3(angle_in, topDepth, gamma, Xsrc, V1, V2):
     return np.cos(AlphaTr(angle_in, V1, V2)) * P2(angle_in, topDepth, gamma, Xsrc, V1, V2) \
            / np.cos(2 * gamma + AlphaTr(angle_in, V1, V2))

def P4(angle_in, topDepth, gamma, V1, V2):
    return topDepth / np.cos(PsiTr(angle_in, gamma, V1, V2))

def fullOffset(angle_in, topDepth, gamma, Xsrc, V1, V2):
    return P1(angle_in, topDepth) * np.sin(angle_in) + P3(angle_in, topDepth, gamma, Xsrc, V1, V2)\
            * np.sin(2 * BetaBase(angle_in, gamma, V1, V2)) / np.cos(AlphaTr(angle_in, V1, V2)) + \
            np.tan(PsiTr(angle_in, gamma, V1, V2)) * topDepth

def ThetaEquiv(angle_in, topDepth, gamma, Xsrc, V1, V2):
    return np.arctan(0.5 *  fullOffset(angle_in, topDepth, gamma, Xsrc, V1, V2) / topDepth)

def wedge_shotgather(gamma, radmax_downwards, radmax_upwards, angstep, topDepth, velocities, X):
    v1, v2 = velocities[0], velocities[1] #m/s

    Angles_in = np.zeros(0, dtype='float')
    Angles_top = np.zeros(0, dtype='float')
    Angles_base = np.zeros(0, dtype='float')
    rad_in = -np.arctan(X / topDepth) #ang min
    while True:
        alpha = AlphaTr(rad_in, v1, v2)
        beta = BetaBase(rad_in, gamma, v1, v2)
        delta = DeltaUp(rad_in, gamma, v1, v2)
        theta = ThetaEquiv(rad_in, topDepth, gamma, X, v1, v2)
        
        Angles_in = np.append(Angles_in, rad_in)
        Angles_base = np.append(Angles_base, beta)
        Angles_top = np.append(Angles_top, theta)
        
        if ((theta >= radmax_downwards) or (delta >= radmax_upwards)):
            break
        rad_in += angstep

    print(Angles_in, Angles_top, Angles_base)

    RayPath_top =  2 * topDepth / np.cos(Angles_top)
    RayPath_base1 =  P1(Angles_in, topDepth) + P4(Angles_in, topDepth, gamma, v1, v2)
    RayPath_base2 = P2(Angles_in, topDepth, gamma, X, v1, v2) + \
            P3(Angles_in, topDepth, gamma, X, v1, v2)
    RayPath_base = RayPath_base1 + RayPath_base2
    TopTime = RayPath_top / v1
    BaseTime = ((RayPath_base1 / v1) + (RayPath_base2 / v2))
    CDPtop = XonWedge(Angles_top, topDepth, X)
    CDPbase = XbaseWedge(Angles_in, topDepth, gamma, X, v1, v2)
    TopDH = DHtop(Angles_top, topDepth, gamma, X)
    BaseDH = DHbase(Angles_in, topDepth, gamma, X, v1, v2)
    return Angles_top, Angles_base, RayPath_top, RayPath_base, TopTime, BaseTime, TopDH, BaseDH, \
            CDPtop, CDPbase

def wedge_array_maker(model, wedgeSlope, dhmax, maxAng, topDepth, nsrc=1000):
    import warnings
    warnings.filterwarnings("error")
    velocities = model.vp 
    v1, v2 = velocities[0], velocities[1] #m/s

    gamma = np.radians(wedgeSlope)
    try:
        radmax_downwards = min(np.arcsin(v1/v2), np.radians(maxAng))
    except:
        radmax_downwards = np.radians(maxAng)
    try:
        radmax_upwards = min(np.arcsin(v2/v1), DeltaUp(radmax_downwards, gamma, v1, v2))
    except:
        radmax_upwards = DeltaUp(radmax_downwards, gamma, v1, v2)
    
    srcMin = Xmin(radmax_downwards, topDepth)
    srcMax = Xmax(radmax_downwards, topDepth, gamma, dhmax)
    XsrcVector = np.linspace(srcMin, srcMax, nsrc)
    XsrcStep = XsrcVector[-1] - XsrcVector[-2]
    angStep = np.arctan(XsrcStep / topDepth)
    print(np.degrees(angStep))
    #print('Xsrc Vector:')
    #print(XsrcVector)
    th, be, ru, rl, tt, tb, dhu, dhl, cdpu, cdpl = wedge_shotgather(gamma, radmax_downwards, \
            radmax_upwards, angStep, topDepth, velocities, XsrcVector[0])
    sizeX = th.shape[0]
    TH = th
    BE = be
    RU = ru
    RL = rl
    TT = tt
    TB = tb
    DHU = dhu
    DHL = dhl
    CDPU = cdpu
    CDPL = cdpl
    X = XsrcVector[0] * np.ones(th.shape[0], dtype='float')
    for i in range(1, XsrcVector.size):
        th, be, ru, rl, tt, tb, dhu, dhl, cdpu, cdpl = wedge_shotgather(gamma, radmax_downwards, \
                radmax_upwards, angStep, topDepth, velocities, XsrcVector[i])
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
        aux[:dhu.shape[0]] = dhu
        DHU = np.vstack([DHU, aux])
        aux[:dhl.shape[0]] = dhl
        DHL = np.vstack([DHL, aux])
        aux[:cdpu.shape[0]] = cdpu
        CDPU = np.vstack([CDPU, aux])
        aux[:cdpl.shape[0]] = cdpl
        CDPL = np.vstack([CDPL, aux])
        aux[:th.shape[0]] = XsrcVector[i] * np.ones(th.shape[0], dtype='float')
        X = np.vstack([X, aux])
        del(aux)
    return TH, BE, RU, RL, TT, TB, DHU, DHL, CDPU, CDPL, X 
