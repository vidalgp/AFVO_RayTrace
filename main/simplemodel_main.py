import sys
import numpy as np
from seisclass import*
from AFVOplots import*
from utils import*
from simplemodel import simple_array_maker

def main():
    print('\n######## GeoComp: Trazado de Rayos y analisis AFVO ########\n\n')
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
    dhstep = 1
    #global TH, B, RU, RL, TT, TB, DH, sps
    TH,B,RU,RL,TT,TB,DH, sps = simple_array_maker(mod, dhmin, dhmax, dhstep, angmax, angstep, \
            topdepth)

    dimX = TH.shape[1]
    dimY = int(TB.max()/dt * (1.02))
    dimZ = TH.shape[0]
    
    print(mod)
    print('\n\tInformacion general de la simulacion:\n')
    print(('Tasa de muestreo dt={}s\nProfundidad del tope = {} m\nAngulo Maximo = {}\nEspesor maximo = {} m\nAtenuacion Q50 ={}').format(dt, topdepth, angmax, dhmax, Aq))
    print(Wavelet(wtype='bp', wf=[5, 10, 40, 80], duration=0.28, wdt=dt))
    print(('\n\tDimensiones del cubo sintetico DH-Gather a generar:\n(angulos x muestras x espesores)\n{} x {} x {}').format(dimX, dimY, dimZ))


    #global seismik, ymin, ymax
    ymax = dimY*dt
    ymin = TT[0].min()* 0.95
    seismik = Seismic(dt, dimX, dimY, dimZ)
    create_timeModel(seismik, mod, dt, np.degrees(TH), TB, TT, Aq)
    
    #global Tmin, Tmax, Bmin, Bmax
    Tmin = TT-0.1
    Tmin[Tmin<0] = 0.0
    Tmax = 0.5 * (TT + TB)
    Bmin = Tmax

    Bmax = TB
    Bmax[Bmax>0] += 0.1
    
    print('\n\tIniciando calculos individuales de AFVO\n')
    for dh in range(0, seismik.zLen, 5):
        print(('AFVO para espesor dh = {}m').format(dh * dhstep + dhmin))
        plot_AFVO(seismik.get_amplitude[dh],np.degrees(TH[dh]),Tmin[dh],Tmax[dh],Bmin[dh],\
                Bmax[dh], seismik.dt, sps[dh],('TopBase_{}').format(dh * dhstep + dhmin))
        seismik.plot_seismogram(ymin=ymin, ymax=ymax, maxtrace=sps[dh], excursion=6, z=dh, \
                angleVec=np.degrees(TH[dh]))
        plt.close('all') 
    
    dh = seismik.zLen - 1
    print(('AFVO para espesor dh = {}m').format(dh * dhstep + dhmin))
    seismik.plot_seismogram(ymin=ymin, ymax=ymax, maxtrace=sps[dh], excursion=6, z=dh, angleVec=np.degrees(TH[dh]))
    plot_AFVO(seismik.get_amplitude[dh], np.degrees(TH[dh]), Tmin[dh], Tmax[dh], Bmin[dh], \
            Bmax[dh], seismik.dt,sps[dh],('TopBase_{}').format(dh*dhstep+dhmin))

    #global fullArray, tminT, tmaxT
    totalTraces = seismik.zLen * seismik.xTraces
    fullArray = np.zeros([totalTraces, 4], dtype='float') 
    tt = TT.reshape(totalTraces)
    tb = TB.reshape(totalTraces)
    theta = np.degrees(TH.reshape(totalTraces))
    dh = DH.reshape(totalTraces) 
  
    print('\n\tIniciando calculos de mapas AFVO: Reflector Tope')
    tminT = tt - 0.1
    tmaxT = 0.5 * (tt + tb)

    fullArray.T[0] = theta
    fullArray.T[1] = dh
    fullArray.T[2] = AVO(seismik.get_amplitude, tminT, tmaxT, seismik.dt)
    fullArray.T[3] = FVO(seismik.get_amplitude, tminT, tmaxT, seismik.dt)

    #fullArray = fullArray[~np.isnan(fullArray).any(axis=1)]

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
    
    fullArray.T[2] = AVO(seismik.get_amplitude, tminB, tmaxB, seismik.dt)
    fullArray.T[3] = FVO(seismik.get_amplitude, tminB, tmaxB, seismik.dt)

    #fullArray = fullArray[~np.isnan(fullArray).any(axis=1)]

    plot_map(fullArray.T[1], fullArray.T[0], fullArray.T[2], xmin, xmax, ymin, ymax,\
            ['dhT','angleT'], 'amp', 'BsimpleBase')
    plot_map(fullArray.T[1], fullArray.T[0], fullArray.T[3], xmin, xmax, ymin, ymax,\
            ['dhT','angleT'], 'freq', 'BsimpleBase')
 
    end = time.time()
    print(("\nTiempo de la simulacion {}s\n").format(end - start))

    plt.close('all') 
    return 0

if (__name__ == '__main__'):
    main()

