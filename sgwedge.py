import numpy as np

def Xmin(angmax, topDepth):
    return -np.tan(angmax) * topDepth

def Xmax(angmax, topDepth, gamma, dhmax):
    return topDepth * np.tan(angmax) + dhmax / np.tan(gamma) 

def XonWedge(angle_in, topDepth, Xsrc):
    xw = Xsrc + topDepth * np.tan(angle_in)
    if xw >= -0.1:
        return xw
    else:
        print(xw)
        raise ValueError('Something went wrong on XonWedge')
        return -1

def DHtop(XonW, gamma):
    return XonW * np.tan(gamma)

def AlphaTr(angle_in, V1, V2):
    return np.arcsin(np.sin(angle_in) * V2 / V1)

def BetaBase(angle_in, gamma, V1, V2):
    return AlphaTr(angle_in, V1, V2) + gamma

def PsiTr(angle_in, gamma, V1, V2):
    return np.arcsin(np.sin(DeltaUp) * V1/V2)

def DeltaUp(angle_in, gamma, V1, V2):
    return 2 * BetaBase(angle_in, gamma, V1, V2) - AlphaTr(angle_in, V1, V2)

def P1(angle_in, topDepth):
    return topDepth / np.cos(angles_in)

def P2(angle_in, topDepth, gamma, Xsrc, V1, V2):
    return ( np.cos(gamma) * DHtop(XonWedge(angle_in, topDepth, Xsrc), gamma) / \
            np.cos(BetaBase(angle_in, V1, V2, gamma)) )

def XbaseWedge(angle_in, topDepth, gamma, Xsrc, V1, V2):
    xb = XonWedge(angle_in, topDepth, Xsrc) + np.sin(AlphaTr(angle_in, V1, V2)) * \
            P2(angle_in, gamma, topDepth, Xsrc, V1, V2)
    if xb < -0.1:
        raise ValueError('Something went wrong on XbaseWedge')
        return -1
    else:
        return xb

def DHbaseWedge(angle_in, topDepth, gamma, Xsrc, V1, V2):
    return XbaseWedge(angle_in, topDepth, gamma, Xsrc, V1, V2) * np.tan(gamma)

def P3(angle_in, topDepth, gamma, Xsrc, V1, V2):
     return np.cos(AlphaTr(angle_in, V1, V2)) * P2(angle_in, gamma, topDepth, Xsrc, V1, V2) \
            / np.cos(2 * gamma + AlphaTr(angle_in, V1, V2))

def P4(angle_in, topDepth, gamma, V1, V2):
    return topDepth / np.cos(PsiTr(angle_in, gamma, V1, V2))

def fullOffset(angle_in, topDepth, gamma, Xsrc, V1, V2):
    return P1(angle_in, topDepth) * np.sin(angle_in) + P3(angle_in, topDepth, gamma, V1, V2, Xsrc)\
            * np.sin(2 * BetaBase(angle_in, gamma, V1, V2)) / np.cos(AlphaTr(angle_in, V1, V2)) + \
            np.tan(PsiTr(angle_in, gamma, V1, V2)) * topDepth

def ThetaEquiv(angle_in, topDepth, gamma, Xsrc, V1, V2):
    return np.arctan(0.5 *  fullOffset(angle_in, topDepth, gamma, Xsrc, V1, V2)) / topDepth

def totaLowerPath(angle_in, topDepth, gamma, Xsrc, V1, V2):
    return P1(angle_in, topDepth) + P2(angle_in, gamma, topDepth, Xsrc, V1, V2) + \
            P3(angle_in, topDepth, gamma, V1, V2, Xsrc) + P4(angle_in, topDepth, gamma, V1, V2)

def totaLowerTime(angle_in, topDepth, gamma, Xsrc, V1, V2):
    return ( (P1(angle_in, topDepth) + P4(angle_in, topDepth, gamma, V1, V2)) / V1 + \
            (P2(angle_in, gamma, topDepth, Xsrc, V1, V2) + \
            P3(angle_in, topDepth, gamma, V1, V2, Xsrc)) / V2)

def wedge_shotgather(wedgeSlope, maxAng, angstep, dhmax, dhstep, topDepth, velocities):
    import warnings
    warnings.filterwarnings("error")
    
    v1, v2 = velocities[0], velocities[1] #m/s
    gamma = np.radians(gamma)
    angstep = np.radians(angstep)
    try:
        radmax_downwards = min(np.arcsin(v1/v2), np.radians(maxAng))
    except:
        radmax_downwards = np.radians(maxAng)
    try:
        radmax_upwards = min(np.arcsin(v2/v1), DeltaUp(radmax_downwards, gamma, v1, v2))
    except:
        radmax_upwards = DeltaUp(radmax_downwards, gamma, v1, v2)
    
    Angles_in = np.zeros(0, dtype='float')
    Angles_top = np.zeros(0, dtype='float')
    Angles_base = np.zeros(0, dtype='float')
    srcMin = Xmin(radmax_downwards, topDepth)
    srcMax = Xmax(radmax_downwards, dhmax, topDepth, gamma)
    XsrcStep = int(topDepth * np.tan(angstep))
    XsrcVector = np.arange(srcMin, srcMax + XsrcStep, XsrcStep)
    for x in XsrcVector:
        rad_in = -np.arctan(x / topDepth) #deg min
        while True:
            alpha = AlphaTr(rad_in, v1, v2)
            beta = BetaBase(rad_in, gamma, v1, v2)
            delta = DeltaUp(rad_in, gamma, v1, v2)
            theta = ThetaEquiv(rad_in, topDepth, x, v1, v2)
            
            Angles_in = np.append(Angles_in, rad_in)
            Angles_base = np.append(Angles_base, beta)
            Angles_top = np.append(Angles_top, theta)
            
            if ((theta >= radmax_downwards) or (DeltaUp >= radmax_upwards)):
                break
            rad_in += angstep

    RayPath_top =  2 * topDepth / np.cos(Angles_top)
    RayPath_base1 =  P1(Angles_in, topDepth) + P4(Angles_in, topDepth, gamma, v1, v2)
    RayPath_base2 = P2(Angles_in, topDepth, gamma, XsrcVector, v1, v2) + \
            P3(Angles_in, topDepth, gamma, XsrcVector, v1, v2)
    RayPath_base = RayPath_base1 + RayPath_base2
    TopTime = RayPath_top / v1
    BaseTime = ((RayPath_base1 / v1) + (RayPath_base2 / v2))
    dhupper = -999
    dhlower = -999
    return Angles_top, Angles_base, RayPath_top, RayPath_base, TopTime, BaseTime

