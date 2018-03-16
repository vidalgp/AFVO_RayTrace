import numpy as np

def alpha_tr(angle_in, V1, V2):
    return np.arcsin(np.sin(angle_in) * V2 / V1)

def wedge_offset(angle_in, topDepth, dh, V1, V2):
    pass

def theta_equivalent(angle_in, topDepth, dh, V1, V2):
    return np.arctan(0.5 * wedge_offset() / topDepth)

def P1(angle_in, X):
    xt = topDepth * np.tan(angle_in)
    print(xt.size)
    alo = topDepth / np.cos(angles_in)
    if (X < 0) and (xt.any() < np.abs(X)):
        for i in range(xt.shape[0]):
            if xt[i] < np.abs(X):
                alo[i] = np.nan
    return alo
def P2(angle_in, gamma, velocities, X):
    return ( (topDepth * np.tan(angle_in) + X) * np.sin(gamma) \
            / np.cos(beta(angle_in, gamma, velocities)) )
def P3(angle_in, gamma, velocities, X):
     return np.cos(alpha(angle_in, velocities)) * \
            P2(angle_in, gamma, velocities, X)\
            / np.cos(2 * gamma + alpha(angle_in, velocities))
def P4(angle_in, gamma, velocities):
    return topDepth / np.cos(psi(angle_in, gamma, velocities))
def totaLowerPath(angle_in, gamma, velocities, X):
    return P1(angle_in, X) + P2(angle_in, gamma, velocities, X) + \
            P3(angle_in, gamma, velocities, X) + P4(angle_in, gamma, velocities)
def totaLowerTime(angle_in, gamma, velocities, X):
    return ( (P1(angle_in, X) + P4(angle_in, gamma, velocities)) / Vup + \
            (P2(angle_in, gamma, velocities, X) + \
            P3(angle_in, gamma, velocities, X)) / Vmid)
def alpha(angle_in, velocities):
    return np.arcsin( np.sin(angle_in) * velocities[1]/velocities[0] )
def beta(angle_in, gamma, velocities):
    return alpha(angle_in, velocities) + gamma
def psi(angle_in, gamma, velocities):
    return np.arcsin(np.sin(2 * beta(angle_in, gamma, velocities) - \
            alpha(angle_in, velocities)) * velocities[0]/velocities[1])
def fullOffset(angle_in, gamma, velocities):
    alo = P1(angle_in, X) * np.sin(angle_in) + P3(angle_in, gamma, velocities, X) \
            * np.sin(2*beta(angle_in, gamma, velocities)) \
            / np.cos(alpha(angle_in, velocities)) + \
            np.tan(psi(angle_in, gamma, velocities)) * topDepth
    return alo
def theta(angle_in, gamma, velocities, topDepth):
    return np.arctan((0.5 * fullOffset(angle_in, gamma, velocities)) / topDepth)

def wedge_shotgather(wedgeSlope, maxAng, step, topDepth, velocities, X=0):
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

        Angles_base = np.append(Angles_base, beta(rad_in, gamma, [v1, v2]))
        
        Angles_top = np.append(Angles_top, theta(rad_in, gamma, [v1, v2], topDepth))
        if critical:
            if ((theta >= radmax_downwards) or (alpha >= radmax_upwards)):
                break
        else:    
            if theta >= np.radians(maxAng):
                break
        
        if (np.degrees(theta(, gamma, [Vup, Vmid], topDepth )) <= maXd) and\
                (np.degrees(psi(np.radians(i*step), gamma, [Vup, Vmid])) <= maXu):
                    break
        i += 1

    rayPath_upper_reflection =  2 * topDepth / np.cos(angles_upper)
    rayPath_lower_reflection = totaLowerPath(angles_in, gamma, velocities, X)
    upperTime = 1e3 * rayPath_upper_reflection / Vup
    lowerTime = 1e3 * totaLowerTime(angles_in, gamma, velocities, X)
    dhupper = -999
    dhlower = -999

    return angles_upper, angles_lower, rayPath_upper_reflection, \
            rayPath_lower_reflection, upperTime, lowerTime, dhupper, dhlower


