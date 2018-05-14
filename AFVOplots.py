import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import CubicSpline, interp1d
from utils import*
import seaborn as sns

def plot_AFVO(gather, angles, tmin1, tmax1, tmin2, tmax2, dt, sps=0, name=''):
    sns.set()
    if not sps:
        sps = angles.size

    topRi = AVO(gather, tmin1, tmax1, dt)
    topFi = FVO(gather, tmin1, tmax1, dt)
    
    baseRi = AVO(gather, tmin2, tmax2, dt)
    baseFi = FVO(gather, tmin2, tmax2, dt)
    
    orderedA = np.transpose(np.vstack([angles[:sps], topRi[:sps], topFi[:sps], baseRi[:sps], baseFi[:sps]]))
    orderedA = orderedA[orderedA[:,0].argsort()]
    angles = np.transpose(orderedA)[0]
    topRi = np.transpose(orderedA)[1]
    topFi = np.transpose(orderedA)[2]
    baseRi = np.transpose(orderedA)[3]
    baseFi = np.transpose(orderedA)[4]

    if angles.size>2:
        ftopRi = CubicSpline(angles, topRi)
        ftopFi = CubicSpline(angles, topFi)
    
        fbaseRi = CubicSpline(angles, baseRi)
        fbaseFi = CubicSpline(angles, baseFi)
    else:
        ftopRi = interp1d(angles, topRi)
        ftopFi = interp1d(angles, topFi)
    
        fbaseRi = interp1d(angles, baseRi)
        fbaseFi = interp1d(angles, baseFi)

    angles_new = np.linspace(angles.min(), angles.max(), 100, endpoint=True)

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2,1)

    ax0 = plt.subplot(gs[0,0])
    ax0.plot(angles_new, ftopRi(angles_new), label='Tope', lw=1)
    ax0.plot(angles, topRi, 'o', label='Data', markersize=1.5, color='k')
    ax0.plot(angles_new, fbaseRi(angles_new), '--', color='y', label='Base', lw=1)
    ax0.plot(angles, baseRi, 'o', markersize=1.5, color='k')
    ax0.set_xlabel(r'Ángulo de incidencia $\theta$ [$\circ$]', fontsize=8)
    ax0.set_ylabel(r'Coeficiente de Reflexión R($\theta$)', fontsize=8)
    ax0.axhline(color='black', linewidth=0.5)
    ax0.axvline(color='black', linewidth=0.5)
    ax0.set_title('Gráfico AVO')
    ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)

    ax1 = plt.subplot(gs[1,0])
    ax1.plot(angles_new, ftopFi(angles_new), label='Tope', lw=1)
    ax1.plot(angles, topFi, 'o', label='Data', markersize=1.5, color='k')
    ax1.plot(angles_new, fbaseFi(angles_new), '--', color='y', label='Base', lw=1)
    ax1.plot(angles, baseFi, 'o', markersize=1.5, color='k')
    ax1.set_xlabel(r'Ángulo de incidencia $\theta$ [$\circ$]', fontsize=8)
    ax1.set_ylabel(r'Frecuencia CWT pico F($\theta$) [Hz]', fontsize=8)
    ax1.set_title('Gráfico FVO')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)

    plt.tight_layout()
    plt.savefig( 'fig/' + name + 'AFVOplot.png', bbox_inches='tight')
    
    sns.reset_orig()

def plot_map(pointsx, pointsy, peak_vals, xmin, xmax, ymin, ymax, axis, type=None, \
        name=''):
    from scipy.interpolate import griddata
    if axis[0] is 'angleT':
        xxaxis = r'Ángulo de incidencia en tope $\theta$ [$\circ$]'
    elif axis[0] is 'angleB':
        xxaxis = r'Ángulo de incidencia en base $\theta$ [$\circ$]'
    elif axis[0] is 'dhT':
        xxaxis = 'Espesor de capa [m]'
    elif axis[0] is 'dhB':
        xxaxis = 'Espesor de capa (inferior) [m]'
    else:
        xxaxis = ''

    if axis[1] is 'angleT':
        yyaxis = r'Ángulo de incidencia en tope $\theta$ [$\circ$]'
    elif axis[1] is 'angleB':
        yyaxis = r'Ángulo de incidencia en base $\theta$ [$\circ$]'
    elif axis[1] is 'dhT':
        yyaxis = 'Espesor de capa [m]'
    elif axis[1] is 'dhB':
        yyaxis = 'Espesor de capa (inferior) [m]'
    else:
        yyaxis = ''

    if type is 'freq':
        ttl = 'Mapa de Frecuencia CWT Pico'
        marker = 'Frecuencia Pico'
    elif type is 'amp':
        ttl = 'Mapa de Amplitud Pico'
        marker = 'Amplitud Pico'
    else:
        ttl=marker=''

    gx, gy = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
    points = np.vstack([pointsx, pointsy]).T
    PAM = griddata(points, peak_vals, (gx, gy), method='cubic')

    gs = gridspec.GridSpec(1,1)
    fig =  plt.figure(figsize=(8,6))
    ax0 = plt.subplot( gs[0,0] )
    im = ax0.imshow(PAM.T, origin='lower', extent=(xmin, xmax, ymin, ymax))
    ax0.plot(pointsx, pointsy, 'k.', ms=0.5)
    cbar = plt.colorbar(im)
    cbar.set_label(marker)
    ax0.set_ylabel(yyaxis)
    ax0.set_xlabel(xxaxis)
    plt.axis('tight')
    ax0.set_title(ttl)
    plt.tight_layout()
    name = 'fig/' + type + '_' + name + '.png'
    plt.savefig(name, bbox_inches='tight')

