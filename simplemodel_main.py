import sys
import numpy as np
from seisclass import*
from AFVOplots import*
from utils import*
from simplemodel import simple_array_maker

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
    dhmin = 1
    dhmax = 51
    dhstep = 10
    global TH, B, RU, RL, TT, TB, DH, sps
    TH,B,RU,RL,TT,TB,DH, sps = simple_array_maker(mod, dhmin, dhmax, dhstep, angmax, angstep, \
            topdepth)

    dimX = TH.shape[1]
    dimY = int(TB[TB!=0].max()/dt * (1.05))
    dimZ = TH.shape[0]

    global seismik, ymin, ymax
    ymax = dimY*dt
    ymin = TT[0].min()* 0.95
    seismik = Seismic(dt, dimX, dimY, dimZ)
    create_timeModel(seismik, mod, dt, np.degrees(TH), TB, TT, Aq)
    
    global Tmin, Tmax, Bmin, Bmax
    Tmin = TT-0.1
    Tmin[Tmin<0] = 0.0
    Tmax = 0.5 * (TT + TB)
    Bmin = Tmax

    Bmax = TB
    Bmax[Bmax>0] += 0.1
    
    print('\nStarting AFVO single computations\n')
    for dh in range(0, seismik.zLen, 5):
        print(('AFVO for dh = {}m').format(dh * dhstep + dhmin))
        plot_AFVO(seismik.get_amplitude[dh],np.degrees(TH[dh]),Tmin[dh],Tmax[dh],Bmin[dh],\
                Bmax[dh], seismik.dt, sps[dh],('TopBase_{}').format(dh * dhstep + dhmin))
        seismik.plot_seismogram(ymin=ymin, ymax=ymax, maxtrace=sps[dh], excursion=3, z=dh, \
                angleVec=np.degrees(TH[dh]))
        plt.close('all') 
    
    dh = seismik.zLen - 1
    seismik.plot_seismogram(ymin=ymin, ymax=ymax, maxtrace=sps[dh], excursion=3, z=dh, angleVec=np.degrees(TH[dh]))
    plot_AFVO(seismik.get_amplitude[dh], np.degrees(TH[dh]), Tmin[dh], Tmax[dh], Bmin[dh], \
            Bmax[dh], seismik.dt,sps[dh],('TopBase_{}').format(dh*dhstep))

    global fullArray, tminT, tmaxT
    totalTraces = seismik.zLen * seismik.xTraces
    fullArray = np.zeros([totalTraces, 4], dtype='float') 
    tt = TT.reshape(totalTraces)
    tb = TB.reshape(totalTraces)
    theta = np.degrees(TH.reshape(totalTraces))
    dh = DH.reshape(totalTraces) 
  
    print('\nStarting AFVO map computations: Top reflector')
    tminT = tt - 0.1
    tmaxT = 0.5 * (tt + tb)

    fullArray.T[0] = theta
    fullArray.T[1] = dh
    fullArray.T[2] = AVO(seismik.get_amplitude, tminT, tmaxT, seismik.dt)
    fullArray.T[3] = FVO(seismik.get_amplitude, tminT, tmaxT, seismik.dt)

    fullArray = fullArray[~np.isnan(fullArray).any(axis=1)]

    xmin = np.floor(fullArray.T[1].min())
    xmax = np.ceil(fullArray.T[1].max())
    ymin = np.floor(fullArray.T[0].min())
    ymax = np.ceil(fullArray.T[0].max())
    
    plot_map(fullArray.T[1], fullArray.T[0], fullArray.T[2], xmin, xmax, ymin, ymax,\
            ['dhT','angleT'], 'amp', 'BsimpleTop')
    plot_map(fullArray.T[1], fullArray.T[0], fullArray.T[3], xmin, xmax, ymin, ymax,\
            ['dhT','angleT'], 'freq', 'BsimpleTop')

    print('\nStarting AFVO map computations: Base reflector')
    tminB = 0.5 * (tb + tt)
    tmaxB = tb + 0.1 
    
    fullArray.T[2] = AVO(seismik.get_amplitude, tminB, tmaxB, seismik.dt)
    fullArray.T[3] = FVO(seismik.get_amplitude, tminB, tmaxB, seismik.dt)

    fullArray = fullArray[~np.isnan(fullArray).any(axis=1)]

    plot_map(fullArray.T[1], fullArray.T[0], fullArray.T[2], xmin, xmax, ymin, ymax,\
            ['dhT','angleT'], 'amp', 'BsimpleBase')
    plot_map(fullArray.T[1], fullArray.T[0], fullArray.T[3], xmin, xmax, ymin, ymax,\
            ['dhT','angleT'], 'freq', 'BsimpleBase')
 
    end = time.time()
    print(("\nElapsed time {}s\n").format(end - start))

    plt.close('all') 
    return 0

if (__name__ == '__main__'):
    main()

