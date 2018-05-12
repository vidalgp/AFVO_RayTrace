import sys
import numpy as np
from seisclass import*
from AFVOplots import*
from utils import*
from wedgemodel import wedge_array_maker, CDPgather

def main():
    import time
    start = time.time()
    if len(sys.argv) > 1:
        mtype, Qbool, nsrc = sys.argv[1], sys.argv[2], int(sys.argv[3])
    
    mod = Model(mtype)
    dt = 0.0001 #ms
    topdepth = 2005
    angmax = 40
    angstep = 1
    dhmax = 51
    wedgeSlope = 5
    global TH, B, RU, RL, TT, TB, DHU, DHL, CDPU, CDPL, X, srcSpacing, srcVector
    TH, B, RU, RL, TT, TB, DHU, DHL, CDPU, CDPL, X, srcSpacing, srcVector = wedge_array_maker(mod, wedgeSlope, \
            dhmax, angmax, topdepth, nsrc)

    global cdpVector,sps
    TH, B, RU, RL, TT, TB, DHU, DHL, cdpVector, sps = CDPgather(srcSpacing, CDPU.max(), CDPU, TH, B, RU, RL, \
            TT, TB, DHU, DHL)

    dimX = TH.shape[1]
    dimY = int(TB[TB!=0].max()/dt * (1.05))
    dimZ = TH.shape[0]

    global seismik, ymin, ymax
    ymax = dimY*dt
    ymin = TT[TT>0].min()* 0.95
    seismik = Seismic(dt, dimX, dimY, dimZ)

    create_timewedgeModel(seismik, mod, dt, np.degrees(TH), np.degrees(B), TB, TT, wedgeSlope, Qbool)
    
    global Tmin, Tmax, Bmin, Bmax
    Tmin = TT-0.1
    Tmin[Tmin<0] = 0.0
    Tmax = 0.5 * (TT + TB)
    Bmin = Tmax

    Bmax = TB
    Bmax[Bmax>0] += 0.1
    
    print('\nStarting AFVO single computations\n')
    for cdp in range(0, seismik.zLen, 5):
        print(('AFVO for cdp = {}m').format(cdpVector[cdp]))
        plot_AFVO(seismik.get_amplitude[cdp], np.degrees(TH[cdp]), Tmin[cdp], Tmax[cdp], Bmin[cdp], \
            Bmax[cdp], seismik.dt,sps[cdp],('TopBase_{}').format(cdpVector[cdp]))
        seismik.plot_seismogram(ymin=ymin, ymax=ymax, maxtrace=sps[cdp], excursion=5, z=cdp, \
                angleVec=np.degrees(TH[cdp]))
        plt.close('all') 
    
    cdp = seismik.zLen - 1
    seismik.plot_seismogram(ymin=ymin, ymax=ymax, maxtrace=sps[cdp], excursion=5, z=cdp, \
            angleVec=np.degrees(TH[cdp]))
    plot_AFVO(seismik.get_amplitude[cdp], np.degrees(TH[cdp]), Tmin[cdp], Tmax[cdp], Bmin[cdp], \
            Bmax[cdp], seismik.dt,sps[cdp],('TopBase_{}').format(cdpVector[cdp]))

    global fullArray, tminT, tmaxT
    totalTraces = seismik.zLen * seismik.xTraces
    fullArray = np.zeros([totalTraces, 5], dtype='float') 
    tt = TT.reshape(totalTraces)
    tb = TB.reshape(totalTraces)
    theta = np.degrees(TH.reshape(totalTraces))
    dhu = DHU.reshape(totalTraces) 
    dhl = DHL.reshape(totalTraces) 
  
    print('\nStarting AFVO map computations: Top reflector')
    tminT = tt - 0.1
    tmaxT = 0.5 * (tt + tb)

    fullArray.T[0] = theta
    fullArray.T[1] = dhu
    fullArray.T[2] = AVO(seismik.get_amplitude, tminT, tmaxT, seismik.dt)
    fullArray.T[3] = FVO(seismik.get_amplitude, tminT, tmaxT, seismik.dt)
    fullArray.T[4] = dhl

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
    xmin = np.floor(fullArray.T[4].min())
    xmax = np.ceil(fullArray.T[4].max())
    
    fullArray.T[2] = AVO(seismik.get_amplitude, tminB, tmaxB, seismik.dt)
    fullArray.T[3] = FVO(seismik.get_amplitude, tminB, tmaxB, seismik.dt)

    plot_map(fullArray.T[4], fullArray.T[0], fullArray.T[2], xmin, xmax, ymin, ymax,\
            ['dhT','angleT'], 'amp', 'BsimpleBase')
    plot_map(fullArray.T[4], fullArray.T[0], fullArray.T[3], xmin, xmax, ymin, ymax,\
            ['dhT','angleT'], 'freq', 'BsimpleBase')
 
    end = time.time()
    print(("\nElapsed time {}s\n").format(end - start))

    plt.close('all') 
    return 0

if (__name__ == '__main__'):
    main()

        
