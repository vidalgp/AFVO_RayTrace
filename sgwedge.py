def simple_shotgather(dh, maxAng, step, topDepth, velocities):
    Vup, Vmid = velocities[0], velocities[1]
    import warnings
    warnings.filterwarnings("error")
    try:
        critic_downw= np.degrees(np.arcsin(Vup/Vmid))
    except:
        critic_downw = 0
        print('Warning: No critical angle found for incident ray')
    try:
        critic_upw = np.degrees(np.arcsin(Vmid/Vup))
    except:
        critic_upw = 0 
        print('Warning: No critical angle found for upgoing reflected ray')

    angles_in = np.zeros(0, dtype='float')
    angles_upper = np.zeros(0, dtype='float')
    angles_transmitted = np.zeros(0, dtype='float')

    alphA = lambda incident: np.arcsin( np.sin(incident) * Vmid/Vup )
    midOffseT = lambda inci: topDepth * np.tan(inci) + \
                                dh * np.tan(alphA(inci))
    thetA = lambda inci: np.arctan(midOffseT(inci) / topDepth)
    
    if critic_downw:
        maXd = min(critic_downw, maxAng)
    else:
        maXd = maxAng

    if critic_upw:
        maXu = min(critic_upw, np.degrees(alphA(np.radians(maXd))))
    else:
        maXu = np.degrees(alphA(np.radians(maXd)))
    
    i = 0
    while (np.degrees(thetA(np.radians(step*i))) <= maXd) and \
            (np.degrees(alphA(np.radians(step*i))) <= maXu):
        angles_in = np.append(angles_in, np.radians(step*i))
        angles_transmitted = np.append(angles_transmitted, alphA(np.radians(step*i)))
        angles_upper = np.append(angles_upper, thetA(np.radians(step*i)))
        i += 1

    rayPath_upper_reflection =  2 * topDepth / np.cos(angles_upper)
    rayPath_lower_reflection2 = 2 * dh / np.cos(angles_transmitted)
    rayPath_lower_reflection1 = 2 * topDepth / np.cos(angles_in)
    rayPath_lower_reflection =  rayPath_lower_reflection1 + rayPath_lower_reflection2   
    upperTime = 1e3 * rayPath_upper_reflection / Vup
    lowerTime = 1e3*(rayPath_lower_reflection1 / Vup + rayPath_lower_reflection2 / Vmid)
    return angles_upper, angles_transmitted, rayPath_upper_reflection, \
            rayPath_lower_reflection, upperTime, lowerTime

def wedge_shotgather(wedgeSlope, maxAng, step, topDepth, velocities, X=0):
    import warnings
    warnings.filterwarnings("error")
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

    Vup, Vmid = velocities[:-1]
    gamma = np.arctan(wedgeSlope) #radians
    try:
        critic_downw= np.degrees(np.arcsin(Vup/Vmid))
    except:
        critic_downw = 0
        print('Warning: No critical angle found for incident ray')
    try:
        critic_upw = np.degrees(np.arcsin(Vmid/Vup))
    except:
        critic_upw = 0 
        print('Warning: No critical angle found for upgoing reflected ray')

    angles_in = np.zeros(0, dtype='float')
    angles_upper = np.zeros(0, dtype='float')
    angles_lower = np.zeros(0, dtype='float')
    
    if critic_downw:
        maXd = min(critic_downw, maxAng)
    else:
        maXd = maxAng

    if critic_upw:
        maXu = min(critic_upw, np.degrees(psi(np.radians(maxAng), gamma, [Vup, Vmid])))
    else:
        maXu = np.degrees(psi(np.radians(maxAng), gamma, [Vup, Vmid]))
    
    i = 0
    while (np.degrees(theta(np.radians(step*i), gamma, [Vup, Vmid], topDepth )) <= \
            maXd) and (np.degrees(psi(np.radians(i*step), gamma, [Vup, Vmid])) <= maXu):
        angles_in = np.append(angles_in, np.radians(step*i))
        angles_lower = np.append(angles_lower, beta(np.radians(step*i), gamma, \
                [Vup, Vmid]))
        angles_upper = np.append(angles_upper, theta(np.radians(step*i), gamma, \
                [Vup, Vmid], topDepth))
        i += 1

    rayPath_upper_reflection =  2 * topDepth / np.cos(angles_upper)
    rayPath_lower_reflection = totaLowerPath(angles_in, gamma, velocities, X)
    upperTime = 1e3 * rayPath_upper_reflection / Vup
    lowerTime = 1e3 * totaLowerTime(angles_in, gamma, velocities, X)
    dhupper = -999
    dhlower = -999

    return angles_upper, angles_lower, rayPath_upper_reflection, \
            rayPath_lower_reflection, upperTime, lowerTime, dhupper, dhlower


