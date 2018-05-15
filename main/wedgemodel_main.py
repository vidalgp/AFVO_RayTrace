import sys
import numpy as np
from seisclass import*
from AFVOplots import*
from wedgemodel import*

def main():
    print('\n######## GeoComp: Trazado de Rayos y analisis AFVO ########\n\n')
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
    #global TH, B, RU, RL, TT, TB, DHU, DHL, CDPU, CDPL, X, srcSpacing, srcVector
    TH, B, RU, RL, TT, TB, DHU, DHL, CDPU, CDPL, X, srcSpacing, srcVector = wedge_array_maker(mod, wedgeSlope, \
            dhmax, angmax, topdepth, nsrc)

    #global cdpVector,sps
    TH, B, RU, RL, TT, TB, DHU, DHL, cdpVector, sps = CDPgather(srcSpacing, CDPU.max(), CDPU, TH, B, RU, RL, \
            TT, TB, DHU, DHL)

    dimX = TH.shape[1]
    dimY = int(TB.max()/dt * (1.1))
    dimZ = TH.shape[0]

    print(mod)
    print('\n\tInformacion general de la simulacion:\n')
    print(('Tasa de muestreo dt={}s\nProfundidad del tope = {} m\nAngulo Maximo = {}\nNumero de fuentes = {}\nEspesor maximo = {} m\nAtenuacion Q50 = {}').format(dt, topdepth, angmax, nsrc, dhmax, Qbool))
    print(Wavelet(wtype='bp', wf=[5, 10, 40, 80], duration=0.28, wdt=dt))
    print(('\n\tDimensiones del cubo sintetico DH-Gather a generar:\n(angulos x muestras x espesores)\n{} x {} x {}').format(dimX, dimY, dimZ))
    #global seismik, ymin, ymax
    ymax = dimY*dt
    ymin = TT[TT>0].min()* 0.95
    seismik = Seismic(dt, dimX, dimY, dimZ)

    create_timewedgeModel(seismik, mod, dt, np.degrees(TH), np.degrees(B), TB, TT, wedgeSlope, Qbool)
    
    #global Tmin, Tmax, Bmin, Bmax
    Tmin = TT-0.1
    Tmin[Tmin<0] = 0.0
    Tmax = 0.5 * (TT + TB)
    Bmin = Tmax

    Bmax = TB
    Bmax[Bmax>0] += 0.1
    
    print('\n\tIniciando calculos individuales de AFVO\n')
    for cdp in range(0, seismik.zLen, 5):
        print(('AFVO para CDP = {}m').format(cdpVector[cdp]))
        plot_AFVO(seismik.get_amplitude[cdp], np.degrees(TH[cdp]), Tmin[cdp], Tmax[cdp], Bmin[cdp], \
            Bmax[cdp], seismik.dt,sps[cdp],('TopBase_{}').format(cdpVector[cdp]))
        seismik.plot_seismogram(ymin=ymin, ymax=ymax, maxtrace=sps[cdp], excursion=6, z=cdp, \
                angleVec=np.degrees(TH[cdp]))
        plt.close('all') 
    
    cdp = seismik.zLen - 1
    print(('AFVO para CDP = {}m').format(cdpVector[cdp]))
    seismik.plot_seismogram(ymin=ymin, ymax=ymax, maxtrace=sps[cdp], excursion=6, z=cdp, \
            angleVec=np.degrees(TH[cdp]))
    plot_AFVO(seismik.get_amplitude[cdp], np.degrees(TH[cdp]), Tmin[cdp], Tmax[cdp], Bmin[cdp], \
            Bmax[cdp], seismik.dt,sps[cdp],('TopBase_{}').format(cdpVector[cdp]))

    #global fullArray, tminT, tmaxT
    totalTraces = seismik.zLen * seismik.xTraces
    fullArray = np.zeros([totalTraces, 5], dtype='float') 
    tt = TT.reshape(totalTraces)
    tb = TB.reshape(totalTraces)
    theta = np.degrees(TH.reshape(totalTraces))
    dhu = DHU.reshape(totalTraces) 
    dhl = DHL.reshape(totalTraces) 
  
    print('\n\tIniciando calculos de mapas AFVO: Reflector Tope')
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

    print('\n\tIniciando calculos de mapas AFVO: Reflector Base')
    tminB = 0.5 * (tb + tt)
    tmaxB = tb + 0.1 
    xmin = np.floor(fullArray.T[4].min())
    xmax = np.ceil(fullArray.T[4].max())
    
    fullArray.T[2] = AVO(seismik.get_amplitude, tminB, tmaxB, seismik.dt)
    fullArray.T[3] = FVO(seismik.get_amplitude, tminB, tmaxB, seismik.dt)

    plot_map(fullArray.T[4], fullArray.T[0], fullArray.T[2], xmin, xmax, ymin, ymax,\
            ['dhB','angleB'], 'amp', 'BwedgeBase')
    plot_map(fullArray.T[4], fullArray.T[0], fullArray.T[3], xmin, xmax, ymin, ymax,\
            ['dhB','angleB'], 'freq', 'BwedgeBase')
 
    end = time.time()
    print(("\nTiempo de la simulacion {}s\n").format(end - start))

    plt.close('all') 
    return 0

if (__name__ == '__main__'):
    main()

        
