#!/usr/bin/nv python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: seisclass.py
#  Purpose: Seismic Signal Analysis
#   Author: Vidal Gonzalez P
#    Email: vidalgonz8@gmail.com
# --------------------------------------------------------------------
"""
    Object oriented code to create and analyze seismic data.

"""

__author__ = "Vidal Gonzalez P"
__email__ = "vidalgonz8@gmail.com"
__status__ = "Completed"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

class Model(object):
    def __init__(self, m=None, vp=None, vs=None, rho=None, poisson=None):
        '''
        vs[m/s]; vp[m/s]; rho[kg/m^3]
        angles: deg
        time: seconds
        '''
        def poisson(vp, vs):
            return ( 0.5 * ( (vp/vs)**2 - 2 ) / ( (vp/vs)**2 - 1) )

        if m is 'bright' or m is 'b':
            self._vp = [2735.0, 2844.0, 2735.0]
            self._vs = [1294.0, 1335.0, 1294.0]
            self._rho= [2140.0, 2400.0, 2140.0]
            self._poisson = poisson(np.asarray(self._vp), np.asarray(self._vs))
        elif m is 'dimmed' or m is 'd':
            self._vp = [3580.0, 3500.0, 3580.0]
            self._vs = [1991.0, 1853.0, 1991.0]
            self._rho= [2330.0, 2540.0, 2330.0]
            self._poisson = poisson(np.asarray(self._vp), np.asarray(self._vs))
        elif m is 'ostrander' or m is 'o':
            self._vp = [3048.0, 2438.0, 3048.0]
            self._vs = [1244.0, 1626.0, 1244.0]
            self._rho= [2400.0, 2140.0, 2400.0]
            self._poisson = poisson(np.asarray(self._vp), np.asarray(self._vs))
        elif m is 'mazzottiA' or m is 'a':
            self._vp = [2260.0, 2770.0, 2345.0]
            self._vs = [1075.0, 1385.0, 1170.0]
            self._rho= [2200.0, 2300.0, 2170.0]
            self._poisson = poisson(np.asarray(self._vp), np.asarray(self._vs))
        elif m is 'mazzottiB' or m is 'm':
            self._vp = [2260.0, 2000.0, 2345.0]
            self._vs = [1075.0, 1330.0, 1170.0]
            self._rho= [2200.0, 2300.0, 2170.0]
            self._poisson = poisson(np.asarray(self._vp), np.asarray(self._vs))
        else:
            vp1, vs1, rho1 = list(map(float, \
                    input('Vp Vs Rho for under/overlying bed:\n').split()))
            vp2, vs2, rho2 = list(map(float, \
                    input('Vp Vs Rho for middle bed:\n').split()))
            self._vp = [vp1, vp2, vp1]
            self._vs = [vs1, vs2, vs1]
            self._rho = [rho1, rho2, rho1]
            self._poisson = poisson(np.asarray(self._vp), np.asarray(self._vs))

    @property
    def vp(self):
        return self._vp

    @property
    def vs(self):
        return self._vs

    @property
    def rho(self):
        return self._rho

    @property
    def poisson(self):
        return self._poisson

    def __str__(self):
        ref = '\n\t\t<Vp\tVs\tRho\tPoisson\t>\n'
        over = ('<{}\t{}\t{}\t{:.2f}\t>\n').format(self.vp[0], self.vs[0], \
                self.rho[0], self.poisson[0])
        sandy = ('<{}\t{}\t{}\t{:.2f}\t>\n').format(self.vp[1], self.vs[1], \
                self.rho[1], self.poisson[1])
        under = ('<{}\t{}\t{}\t{:.2f}\t>\n').format(self.vp[2], self.vs[2], \
                self.rho[2], self.poisson[2])
        return(("{}Overlying bed\t{}Middle Bed\t{}Underlying bed\t{}"\
                ).format(ref, over, sandy, under))


class Signal(object):
    def __init__(self, dt, signal=[], tvec=[], tmin=0, duration=None):
        '''
        dt = [ms]
        tmin = [ms]
        duration = [ms]
        '''
        if (len(signal)):
            if (len(tvec)):
                self._signal = np.array(signal, dtype = 'float64')
                self._dt = dt
                self._ns = self._signal.shape[0]
                self._time = self._dt * (self._ns - 1)
                self._tvec = np.array(tvec, dtype='float')
                self._CWT_flag = None
                self.FFT_analysis()
            else:
                self._signal = np.array(signal, dtype = 'float64')
                self._dt = dt
                self._ns = self._signal.shape[0]
                self._time = self._dt * (self._ns - 1)
                self._tvec = np.arange(tmin, tmin + self._time + dt , dt)
                self._CWT_flag = None
                self.FFT_analysis()
        elif (duration):
            self._dt = dt
            time = duration
            self._time = time
            self._ns = int(time / dt) + 1
            self._signal = np.zeros(self._ns, dtype = 'float64')
            self._tvec = np.arange(tmin, tmin + duration + dt, dt)
            self._CWT_flag = None
            self.FFT_analysis()
        else:
            print ("Incorrect input parameters")

    @property
    def signal(self):
        return self._signal

    @property
    def dt(self):
        return self._dt

    @property
    def time(self):
        return self._time

    @property
    def ns(self):
        return self._ns

    @property
    def tvec(self):
        return self._tvec

    def normalize(self):
        self._signal /= np.abs(self._signal).max()
        return 'Signal Normalized'

    def scale(self, scalar=1):
        self._signal *= scalar
        return 'Signal Scaled'

    def FFT_analysis(self):
        from numpy.fft import fft, fftfreq, fftshift

        self._yf = np.abs(fftshift(fft(self._signal)))
        self._xf = fftshift(fftfreq(self._ns, self._dt))

    def __iadd__(self, other):
        try:
            if ( self._signal.shape[0] != other._signal.shape[0] ):
                raise IndexError('Incompatible signals. Check time vectors')
            elif any(self._tvec != other._tvec):
                raise ValueError('Incompatible signals. Check time vectors')
            else:
                self._signal += other._signal
                self._CWT_flag = None
                self.FFT_analysis()
                return self
        except:
            raise

    def to_file(self):
        fname = ('dt{}_ns{}.dat').format(self._dt, self._ns)
        with open(fname, 'x') as fi:
            for e in self._signal:
                fi.write(('{:.6f}\n').format(e))

    def apply_cwt(self, fmin=1, fmax=90, w0=8):
        '''
        w0: tradeoff between time and frequency resolution
        '''
        from obspy.signal.tf_misfit import cwt

        self._fmin = fmin
        self._fmax = fmax
        self._scalogram = cwt(self._signal, self._dt, w0, self._fmin, self._fmax)
        self._nfgrid = np.logspace(np.log10(self._fmin), np.log10(self._fmax), \
                self._scalogram.shape[0])
        self._CWT_flag = True

    def plot(self, fmax=120):
        from obspy.imaging.cm import obspy_sequential
        import warnings
        warnings.filterwarnings('ignore')
        if (self._CWT_flag):
            end = 3
        else:
            end = 2

        fig = plt.figure( figsize=(8, 6) )
        gs= gridspec.GridSpec(end, 1)

        ax0 = plt.subplot( gs[ end-1,0 ] )

        ax0.plot( self._xf, self._yf, color='black', linewidth=1)
        ax0.set_xlabel('Frecuencia [Hz]')
        ax0.set_ylabel('Energía')
        ax0.set_xlim([0, fmax])
        ax0.grid()

        ax1 = plt.subplot( gs[ end-2,0 ] )
        ax1.plot(self._tvec, self._signal, color='black', linewidth=1)
        ax1.set_xlabel('Tiempo [s]')
        ax1.set_ylabel('Amplitud')
        ax1.autoscale(enable=True, axis='x', tight=True)
        ax1.grid()

        if (self._CWT_flag):
            ax2 = plt.subplot(gs[ end-3,0 ])
            x, y = np.meshgrid(self._tvec, self._nfgrid)
            f=ax2.pcolormesh(x, y, np.abs(self._scalogram), cmap=obspy_sequential)
            ax2.set_xlabel("Tiempo [s]")
            ax2.set_ylabel("Frecuencia [Hz]")
            ax2.set_yscale('log')
            ax2.set_ylim(self._fmin, self._fmax)
            ax2.set_title('Frecuencia CWT')
            cbaxes = fig.add_axes([0.7, 0.98, 0.25, 0.01])
            cbar = fig.colorbar(f, orientation='horizontal', cax = cbaxes)
            tck = cbar.ax.get_xticklabels()
            tck = list(map(lambda x: x.get_text(), tck))
            tck[1:-1] = len(tck[1:-1])*' '
            cbar.ax.set_xticklabels(tck)
            cbar.ax.tick_params(labelsize=8)
        plt.tight_layout()


class Wavelet(Signal):
    def __init__(self, wf, wtype, wdt=0.0001, duration=0.28):
        '''
        wdt = [s]
        duration = [s]
        wtype: tipo de ondicula (Ricker, Ormsby)
        wduration: tamaño de la ondicula en segundos.
        wfc: frecuencia central (caso Ricker) y frecuencias [f1, f2, f3, f4] ormsby
        '''
        from wavelets import ricker, ormsby
        self._wtype = wtype
        if (1e3*duration % 2 == 0):
            atvec = np.arange(-duration/2, duration/2 + wdt, wdt)
        else:
            atvec = np.arange(-(duration - wdt) / 2, (duration - wdt) / 2 + wdt, wdt)

        if wtype is 'ricker' or wtype is 'r':
            self._wf = float(wf)
            Wa = ricker(self._wf, atvec)
        elif wtype is 'ormsby' or wtype is 'bp':
            self._wf = list(map(float, wf))
            Wa = ormsby(self._wf, atvec)
        else:
            raise ValueError('Undefined wavelet')

        super().__init__(dt = wdt, signal = Wa, tvec = atvec)
        self.normalize()

    def apply_Q(self, pathlength, V, Q=50):
        '''
        Apply attenuation Q according to the path length in time
        '''
        alpha = (np.pi*np.mean(self._wf))/(Q*V)
        self._signal *= np.exp(-alpha*pathlength)
        self.FFT_analysis()

    def __str__(self):
        return ('\n\tWavelet information\nType: {:s}\nLength = {:.6f} ms\
                \nFrequency: {} Hz\nSampling rate: {:.6f} ms\n# of samples: {:d}\n'\
                ).format(self._wtype, self._time, str(self._wf), self._dt, self._ns)


class ReflectivityS(object):
    def __init__(self, ns=None, rs=[]):
        if (len(rs)):
            self._rserie = np.array(rs, dtype = 'float')
        elif(ns):
            self._rserie = np.zeros(ns, dtype = 'float')

    @property
    def rserie(self):
        return self._rserie

    def add_pulse(self, ix, rval):
        self._rserie[ix] += rval

    def add_layer_reflection(self, vp, vs, rho, iAngle, ix, reflector):
        '''
        type: 'A' for Acoustic Impedance / 'B' for Elastic Impedance
        iAngle: angle of incidence degrees
        ix = digitizing index [0, ns-1]
        vp, vs and rho are tuples defining the distincts parameters of adjacent layers
        '''
        from bruges.reflection import zoeppritz_element
        if reflector is 'top':
            self._rserie[ix] += np.real(zoeppritz_element(vp[0], vs[0], rho[0], \
                    vp[1], vs[1], rho[1], iAngle, 'PdPu'))
        elif reflector is 'base':
            transm = np.real(zoeppritz_element(vp[0], vs[0], rho[0], vp[1], vs[1], \
                    rho[1], iAngle, 'PdPd'))
            beta = np.degrees(np.arcsin( (vp[1]/vp[0]) * np.sin(np.radians(iAngle))))
            transmref = transm * np.real(zoeppritz_element(vp[1], vs[1], rho[1], \
                    vp[0], vs[0], rho[0], beta, 'PdPu'))
            final = transmref * np.real(zoeppritz_element(vp[1], vs[1], rho[1], \
                    vp[0], vs[0], rho[0], beta, 'PdPd'))
            self._rserie[ix] += final
    
    def add_wedge_reflection(self, vp, vs, rho, iAngle, ix, reflector, wedgeAngle=0):
        from bruges.reflection import zoeppritz_element
        if reflector is 'top':
            self._rserie[ix] += np.real(zoeppritz_element(vp[0], vs[0], rho[0], \
                    vp[1], vs[1], rho[1], iAngle, 'PdPu'))
        elif reflector is 'base':
            transm = np.real(zoeppritz_element(vp[0], vs[0], rho[0], vp[1], vs[1], \
                    rho[1], iAngle, 'PdPd'))
            alpha = np.degrees(np.arcsin( (vp[1]/vp[0]) * np.sin(np.radians(iAngle))))
            beta = alpha + wedgeAngle
            delta = beta + wedgeAngle
            if delta >=90:
                raise ValueError("Ray won't reach surface")
            transmref = transm * np.real(zoeppritz_element(vp[1], vs[1], rho[1], \
                    vp[0], vs[0], rho[0], beta, 'PdPu'))
            final = transmref * np.real(zoeppritz_element(vp[1], vs[1], rho[1], \
                    vp[0], vs[0], rho[0], delta, 'PdPd'))
            self._rserie[ix] += final 

    def __iadd__(self, other):
        try:
            if ( self._rserie.shape[0] != other._rserie.shape[0]):
                raise IndexError('Incompatible series. Check number of samples')
            else:
                self._rserie += other._rserie
                return self
        except:
            raise


class Trace(Signal, ReflectivityS):
    def __init__(self, dt=None, duration=None, asignal=[], rseries=[], wavelet=[]):
        '''
        dt = [ms]
        duration = [ms]
        '''
        if (len(asignal) and dt):
            ns = len(asignal)
            duration = dt * ns
            tvector = np.arange(0, duration, dt)
            Signal.__init__(self, dt = dt, signal = asignal, tvec = tvector)
            if len(rseries):
                ReflectivityS.__init__(self, rs = rseries)
            else:
                ReflectivityS.__init__(self, ns=ns)
        else:
            ReflectivityS.__init__(self, rs = rseries)
            rawsignal = np.convolve(self._rserie, wavelet._signal, mode = 'same')
            duration = wavelet._dt * len(rawsignal)
            tvector = np.arange(0, duration, wavelet._dt)
            Signal.__init__(self, dt = wavelet._dt, signal = rawsignal, tvec = tvector)

    def make_copy(self):
        return Trace(dt = self._dt, asignal = self._signal, rseries = self._rserie)

    def __iadd__(self, other):
        Signal.__iadd__(self, other)
        ReflectivityS.__iadd__(self, other)
        return self

    def plot(self, ymax=None, ymin=0, fmax=120, RS=True, tmark=[]):
        '''
        Plots the signal
        '''
        from obspy.imaging.cm import obspy_sequential
        if ymax:
            pass
        else:
            ymax=np.max(self._tvec)

        if (self._CWT_flag) and RS:
            end = 4
            rsp = 3
            cwp = 2
        elif RS or self._CWT_flag:
            end = 3
            rsp = cwp = 2
        else:
            end = 2

        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, end)

        ax2 = plt.subplot(gs[:,0])
        ax2.plot( self._yf, self._xf, color='black', linewidth=0.75)
        ax2.set_ylabel('Frecuencia [Hz]')
        ax2.set_xlabel('Energía')
        ax2.set_ylim([0, fmax])
        ax2.set_title('Análisis FFT')
        ax2.grid()

        ax1 = plt.subplot(gs[:,1])
        ax1.plot(self._signal, self._tvec, color='black', linewidth=0.65)
        ax1.set_ylabel('Tiempo [s]')
        ax1.set_xticklabels([])
        labels = [item.get_text() for item in ax1.get_xticklabels()]
        labels[1] = '-'
        labels[3] = '+'
        ax1.set_xticklabels(labels)
        plt.fill_betweenx(self._tvec, 0, self._signal, \
                where=self._signal > 0, facecolor = [0.6,0.6,1.0], linewidth = 0)
        plt.fill_betweenx(self._tvec, 0, self._signal, \
                where=self._signal < 0, facecolor = [1.0,0.7,0.7], linewidth = 0)
        ax1.invert_yaxis()
        tp = np.abs(self._signal).max() * 1.1
        ax1.set_xlim(-tp, tp)
        ax1.set_ylim([ymax, ymin])
        if len(tmark):
            for value in list(tmark):
                ax1.axhline(y=value, lw=0.3, c='g')
        ax1.set_title('Traza Sísmica')
        ax1.grid()


        if RS and any(self._rserie != 0):
            ax0 = plt.subplot(gs[:,rsp])
            ax0.plot(self._rserie, self._tvec, color='black', linewidth=1)
            ax0.set_xticklabels([])
            labels = [item.get_text() for item in ax0.get_xticklabels()]
            labels[1] = '-'
            labels[3] = '+'
            ax0.set_xticklabels(labels)
            ax0.invert_yaxis()
            th = np.abs(self._rserie).max() * 1.1
            ax0.set_xlim(-th, th)
            ax0.set_ylim([ymax, ymin])
            ax0.set_title('Serie de Reflectividad')
            ax0.grid()

        if self._CWT_flag is True:
            ax3 = plt.subplot(gs[:,cwp])
            x, y = np.meshgrid(self._nfgrid, self._tvec)
            f=ax3.pcolormesh(x, y, np.abs(self._scalogram).T, cmap=obspy_sequential)
            ax3.set_ylabel("Tiempo [s]")
            ax3.set_xlabel("Frecuencia [Hz]")
            ax3.set_xscale('log')
            ax3.set_xlim(self._fmin, self._fmax)
            ax3.set_title('Frecuencia CWT')
            ax3.invert_yaxis()
            ax3.set_ylim([ymax, ymin])
            cbar = fig.colorbar(f, pad=0.04, fraction=0.046)
            tck = cbar.ax.get_yticklabels()
            tck = list(map(lambda y: y.get_text(), tck))
            tck[1:-1] = len(tck[1:-1])*' '
            cbar.ax.set_yticklabels(tck)
        plt.tight_layout()


class Seismic(object):
    def __init__(self, dt, xTraces, ySamples, zLen = 1):
        '''
        initialization of seismic given its dimensions
        '''
        self._xTraces = xTraces
        self._ySamples = ySamples
        self._zLen = zLen
        self._SEIS = np.zeros ( [zLen, xTraces, ySamples] , dtype='float64')
        self._RS = np.zeros ( [zLen, xTraces, ySamples] , dtype='float64')
        self._dt = dt
        self._time = ySamples * dt
        self._tvec = np.arange(0, self._time, dt)

    @property
    def xTraces(self):
        return self._xTraces

    @property
    def ySamples(self):
        return self._ySamples

    @property
    def zLen(self):
        return self._zLen

    @property
    def dt(self):
        return self._dt
    
    @property
    def get_amplitude(self):
        return self._SEIS
    
    @property
    def get_rs(self):
        return self._RS 

    def get_trace(self, xTrace, zLen = 0):
        return self._SEIS[zLen][xTrace], self._RS[zLen][xTrace]
    
    def add_trace(self, trace, xTrace, zLen):
        if trace._ns == self._ySamples:
            self._SEIS [zLen][xTrace] = trace.signal
            self._RS [zLen][xTrace] = trace.rserie
        else:
            print(('TraceNS = {} != Seismic ySamples ={}').format(trace._ns, \
                    self._ySamples))
            raise ValueError("Incompatible number of samples")

    def from_file(self, fname, zLen=0):
        '''
        fname: seismogram file name
        idt = traces sampling rate
        '''
        self._SEIS[z] = np.loadtxt(fname, dtype='float64')

    def apply_NMO(self):
        pass

    def normalize(self):
        self._SEIS /= np.abs(self._SEIS).max()
        return('Seismic cube Amplitudes normalized')
    
    def plot_density(self, ymax=None, ymin = 0, z = 1, depth = False):
        import warnings
        warnings.filterwarnings('ignore')
        if not ymax:
            ymax = self._time
        yLabel = "Tiempo Doble de Viaje [$s^{-4}$]"
        xTitle = 'Sismograma en Tiempo'
        tvec = self._tvec
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0,0])
        f = ax1.imshow(self._SEIS[z].T, aspect='auto')
        ax1.set_title(xTitle)
        ax1.set_ylabel(yLabel)
        ax1.set_ylim([ymax, ymin])
        cbaxes = fig.add_axes([0.7, 0.98, 0.25, 0.01])
        cbar = fig.colorbar(f, orientation='horizontal', \
                ticks=[self._SEIS[z-1].min(), self._SEIS[z-1].max()], cax = cbaxes)
        cbar.ax.set_xticklabels(['-', '+'])
        cbar.ax.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(('fig/seismo_den{}.png').format(z), bbox_inches='tight')

    
    def plot_seismogram(self, ymax=None, ymin = 0, maxtrace=0, z = 0, depth = False, excursion=1):
        '''
        Created by: Wes Hamlyn, 2014
        Modified by: Vidal Gonzalez P, 2018
        '''
        if not maxtrace:
            maxtrace = self._xTraces
        if not ymax:
            ymax = self._time
        yLabel = "Tiempo Doble de Viaje [$s^{-4}$]"
        xTitle = 'Sismograma en Tiempo'
        tvec = self._tvec
        excursion = excursion
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0,0])
        self.plot_vawig(ax1, self._SEIS[z], tvec, excursion, maxtrace)
        ax1.set_ylim([ymin,ymax])
        ax1.set_xlim((-.999,maxtrace-.001))
        ax1.invert_yaxis()
        ax1.set_xlabel(xTitle, fontsize=14)
        ax1.set_ylabel(yLabel)
        plt.tight_layout()
        plt.savefig(('fig/seismo_wig{}.png').format(z), bbox_inches='tight')

    def plot_vawig(self, axhdl, data, t, excursion, maxtrace):
        '''
        Created by: Wes Hamlyn, 2014
        Modified by: Vidal Gonzalez P, 2018
        '''
        ntrc = self._xTraces
        nsamp = self._ySamples
        t = np.hstack([0, t, t.max()])
        i = 0
        for trace in data[:maxtrace]:
            tbuf = excursion * np.array(trace) + i
            tbuf = np.hstack([i, tbuf, i])
            axhdl.plot(tbuf, t, color='black', linewidth=0.2)
            plt.fill_betweenx(t, tbuf, i, where=tbuf>i, facecolor=[0.6,0.6,1.0], \
                    linewidth=0)
            plt.fill_betweenx(t, tbuf, i, where=tbuf<i, facecolor=[1.0,0.7,0.7], \
                    linewidth=0)
            axhdl.set_xlim((-excursion, ntrc+excursion))
            axhdl.xaxis.tick_top()
            axhdl.xaxis.set_label_position('top')
            axhdl.invert_yaxis()
            i += 1

###################################################################################

def main():
    global ss
    ss = Seismic(1, 100, 5000) 
    w = Wavelet(wtype='bp', wf=[15,35,55,75], wdt=0.0001)
    rs = ReflectivityS(ns=5000) 
    rs.add_pulse(700, 0.5)
    rs.add_pulse(2500, -.1)
    rs.add_pulse(2000, .71)
    rs.add_pulse(4000, -.29)
    tr = Trace(dt=0.001, wavelet=w, rseries=rs.rserie)
    for i in range(ss.xTraces):
        ss.add_trace(tr, i, 0)
    
    ss.plot_density()
    ss.plot_seismogram(); plt.show()
    
    ss.normalize()
    ss.plot_seismogram(); plt.show()

    return 0


if (__name__ == '__main__'):
    main()
