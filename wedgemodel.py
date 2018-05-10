import numpy as np
from seisclass import Model

def Xmin(angmax, topDepth):
    return -np.tan(angmax) * topDepth

def Xmax(angmax, topDepth, gamma, dhmax):
    return topDepth * np.tan(angmax) + dhmax / np.tan(gamma) 

def XonWedge(angle_in, topDepth, Xsrc):
    xw = Xsrc + topDepth * np.tan(angle_in)
    if xw.any() < -0.1:
        raise ValueError('Something went wrong on XonWedge')
        return -1
    else:
        return xw

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
            P2(angle_in, topDepth, gamma, Xsrc, V1, V2)
    if xb.any() < -0.1:
        raise ValueError('Something went wrong on XbaseWedge')
        return -1
    else:
        return xb

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
        beta = BetaBase(rad_in, gamma, v1, v2)
        delta = DeltaUp(rad_in, gamma, v1, v2)
        theta = ThetaEquiv(rad_in, topDepth, gamma, X, v1, v2)
        
        Angles_in = np.append(Angles_in, rad_in)
        Angles_base = np.append(Angles_base, beta)
        Angles_top = np.append(Angles_top, theta)
        
        if ((theta >= radmax_downwards) or (delta >= radmax_upwards)):
            break
        rad_in += angstep

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

def wedge_array_maker(model, wedgeSlope, dhmax, maxAng, topDepth, nsrc=500):
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
    print(np.degrees(radmax_downwards), np.degrees(radmax_upwards))
    srcMin = Xmin(radmax_downwards, topDepth)
    srcMax = Xmax(radmax_downwards, topDepth, gamma, dhmax)
    XsrcVector = np.linspace(srcMin, srcMax, nsrc)
    XsrcStep = XsrcVector[-1] - XsrcVector[-2]
    angStep = np.arctan(XsrcStep / topDepth)
    print(XsrcVector)
    sizeX = int(np.ceil(1+2 * radmax_downwards/angStep))
    spanSize = np.zeros(XsrcVector.size, dtype='int')
    for i in range(XsrcVector.size):
        th, be, ru, rl, tt, tb, dhu, dhl, cdpu, cdpl = wedge_shotgather(gamma, radmax_downwards, \
                radmax_upwards, angStep, topDepth, velocities, XsrcVector[i])
        spanSize[i] = th.size
        if i == 0:
            TH = np.nan * np.ones(sizeX, dtype='float')
            TH[:th.size] = th
            BE = np.nan * np.ones(sizeX, dtype='float')
            BE[:be.size] = be
            RU = np.nan * np.ones(sizeX, dtype='float')
            RU[:ru.size] = ru
            RL = np.nan * np.ones(sizeX, dtype='float')
            RL[:rl.size] = rl
            TT = np.nan * np.ones(sizeX, dtype='float')
            TT[:tt.size] = tt
            TB = np.nan * np.ones(sizeX, dtype='float')
            TB[:tb.size] = tb
            DHU = np.nan * np.ones(sizeX, dtype='float')
            DHU[:dhu.size] = dhu
            DHL = np.nan * np.ones(sizeX, dtype='float')
            DHL[:dhl.size] = dhl
            CDPU = np.nan * np.ones(sizeX, dtype='float')
            CDPU[:cdpu.size] = cdpu
            CDPL = np.nan * np.ones(sizeX, dtype='float')
            CDPL[:cdpl.size] = cdpl
            X = np.nan * np.ones(sizeX, dtype='float')
            X[:cdpl.size] = XsrcVector[i] * np.ones(cdpl.size, dtype='float')
        else:   
            aux = np.nan * np.ones(sizeX, dtype='float')
            aux[:th.size] = th
            TH = np.vstack([TH, aux])
            
            aux[:be.size] = be
            BE = np.vstack([BE, aux])
            
            aux[:ru.size] = ru
            RU = np.vstack([RU, aux])
            
            aux[:rl.size] = rl
            RL = np.vstack([RL, aux])
            
            aux[:tt.size] = tt
            TT = np.vstack([TT, aux])
            
            aux[:tb.size] = tb
            TB = np.vstack([TB, aux])

            aux[:dhu.size] = dhu
            DHU = np.vstack([DHU, aux])

            aux[:dhl.size] = dhl
            DHL = np.vstack([DHL, aux])
            
            aux[:cdpu.size] = cdpu
            CDPU = np.vstack([CDPU, aux])
            
            aux[:cdpl.size] = dhl
            CDPL = np.vstack([CDPL, aux])
            
            aux[:cdpl.size] = XsrcVector[i] * np.ones(cdpl.size, dtype='float')
            X = np.vstack([X, aux])
            del(aux)
            
    return TH[~np.isnan(TH)], BE[~np.isnan(BE)], RU[~np.isnan(RU)], RL[~np.isnan(RL)], TT[~np.isnan(TT)], \
            TB[~np.isnan(TB)], DHU[~np.isnan(DHU)], DHL[~np.isnan(DHL)], CDPU[~np.isnan(CDPU)], \
            CDPL[~np.isnan(CDPL)], X[~np.isnan(X)], XsrcStep, XsrcVector 


def CDPgather(srcspacing, cdpMax, CDParray, th, be, ru, rl, tt, tb, dhu, dhl):
    '''
    all the arrays should be reshaped to 1d
    '''
    cdpRanges = np.arange(0.0, cdpMax + 2 * srcspacing, srcspacing)
    zSize = cdpRanges.size-1
    cdpVector = np.zeros([zSize, 2])

    TH = np.zeros([zSize, zSize], dtype='float')
    BE = np.zeros([zSize, zSize], dtype='float')
    RU = np.zeros([zSize, zSize], dtype='float')
    RL = np.zeros([zSize, zSize], dtype='float')
    TT = np.zeros([zSize, zSize], dtype='float')
    TB = np.zeros([zSize, zSize], dtype='float')
    DHU = np.zeros([zSize, zSize], dtype='float')
    DHL = np.zeros([zSize, zSize], dtype='float')
    sps = np.zeros(zSize, dtype='int')

    for i in range(cdpRanges.size-1):
        k = 0
        locix = np.where((CDParray >= cdpRanges[i]) & (CDParray < cdpRanges[i+1])) #1d array w indices
        cdpVector[i] = (cdpRanges[i],cdpRanges[i+1])
        for e in locix[0]:
            TH[i][k] = th[e]
            BE[i][k] = be[e]
            RU[i][k] = ru[e]
            RL[i][k] = rl[e]
            TT[i][k] = tt[e]
            TB[i][k] = tb[e]
            DHU[i][k] = dhu[e]
            DHL[i][k] = dhl[e]
            k += 1
        pos = np.where(TH[i]==0)[0][0]
        sps[i] = pos
        TH[i][pos:] = TH[i][pos-1]
        BE[i][pos:] = BE[i][pos-1]
        RU[i][pos:] = RU[i][pos-1]
        RL[i][pos:] = RL[i][pos-1]
        TT[i][pos:] = TT[i][pos-1]
        TB[i][pos:] = TB[i][pos-1]
        DHU[i][pos:] = DHU[i][pos-1]
        DHL[i][pos:] = DHL[i][pos-1]
    
    forDel=np.array([], dtype='int')
    for i, column in enumerate(RU):
        if column[:sps[i]].sum()<=column[0]:
            forDel = np.hstack([forDel, int(i)])
    TH = np.delete(TH, forDel, 0)            
    BE = np.delete(BE, forDel, 0)            
    RU = np.delete(RU, forDel, 0)            
    RL = np.delete(RL, forDel, 0)            
    TT = np.delete(TT, forDel, 0)            
    TB = np.delete(TB, forDel, 0)            
    DHU = np.delete(DHU, forDel, 0)            
    DHL = np.delete(DHL, forDel, 0)            
    sps = np.delete(sps, forDel, 0)            
    cdpVector = np.delete(cdpVector, forDel, 0)            

    return TH, BE, RU, RL, TT, TB, DHU, DHL, cdpVector, sps

