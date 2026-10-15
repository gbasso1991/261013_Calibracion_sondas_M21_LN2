#%% Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
pendiente_HvsI = 3716.3 # 1/m
ordenada_HvsI = 1297.0 # A/m
# from clase_resultados import ResultadosESAR
#%% Lector Templog
def lector_templog(path):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt).
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura
    '''
    data = pd.read_csv(path,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python')
    temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']])

    time = np.array([(t-timestamp[0]).total_seconds() for t in timestamp])
    return timestamp,time,temp_CH1, temp_CH2
#%% Curvatura
def curvatura(T):
    ancho_de_ventana=11
    dT=savgol_filter(T,window_length=ancho_de_ventana,polyorder=3,deriv=1,delta=1.0)
    dTT=savgol_filter(T,window_length=ancho_de_ventana,polyorder=3,deriv=2,delta=1.0)
    curv=np.abs(dTT)/(1+dT**2)**1.5
    return dT,dTT,curv
def expo(t,A,B,tau):
    return A + B*np.exp(-t/tau)
def biexpo(t,A,B1,tau1,B2,tau2):
    return A + B1*np.exp(-t/tau1) + B2*np.exp(-t/tau2)

#%% Imports 
nombre = 'calibracion  sondas 261013\nEG100% --> LN2 --> RT'
print('-'*50,nombre)

temps_antes = glob("calibracion/*test1.csv",recursive=True)
temps_dps = glob("calibracion/*test2.csv",recursive=True)

for t in [temps_antes,temps_dps]:
    t.sort()
for p in temps_antes:
    print('  -',os.path.basename(p))
for p in temps_dps:
    print('  -',os.path.basename(p))

#%% Antes de calibrar  
info = '''
Calibracion sondas Rugged a temperatura de LN2
Sondas juntas dentro de criovial con etilenglicol puro
Sumergidas en LN2 durante 200 s aproximadamente
Veo diferencia de temperatura entre canales 

Abro software Rugged Connect
Leo puertos y selecciono el correspondiente
Connect
Unblock 
Importante: las celdas deben llenarse en orden creciente de temperaturas, de lo contrario la calibracion no funciona!
Guardar la intensidad para la temperatura T= -195.8 C
'''
print(info)

_,time_1,temp_CH1_1, temp_CH2_1 = lector_templog(temps_antes[0])
delta_1 = temp_CH2_1 - temp_CH1_1
#%%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
axins_on=1
fig1,(ax,ax2) = plt.subplots(2,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.plot(time_1,temp_CH1_1,'C0-',label='CH1')
ax.plot(time_1,temp_CH2_1,'C1-',label='CH2')#
ax2.plot(time_1,delta_1,'C2-',label='ΔT = CH2 - CH1')#

if axins_on:
    axin = ax.inset_axes([0.4, 0.4, 0.59, 0.57])
    axin.plot(time_1[-50:],temp_CH1_1[-50:],'C0.-')
    axin.plot(time_1[-50:],temp_CH2_1[-50:],'C1.-')#
    mark_inset(ax, axin, loc1=3, loc2=4)
    
    axin2 = ax2.inset_axes([0.43, 0.08, 0.56, 0.36])
    axin2.plot(time_1[-50:],delta_1[-50:],'C2.-')
    mark_inset(ax2, axin2, loc1=1, loc2=2)
    for b in [axin,axin2]:
        b.grid()

for a in [ax,ax2]:
    a.grid()
    a.legend(loc='lower left',shadow=True,frameon=True,ncol=1)
    a.set_ylabel('T (°C)')

ax2.set_xlabel('t (s)')
plt.suptitle('Antes de calibrar')
#%% Despues de calibrar
info2='''
Luego de setear el valor correspondiente de intensidad en cada canal para la temperatura del LN2, es decir -195.8 C
Registramos desde LN2 hasta RT primero exponiendo a atmosfera y despues con bano termico
'''
print(info2)
axin_on=1
_,time_2,temp_CH1_2, temp_CH2_2 = lector_templog(temps_dps[0])
delta_2 = temp_CH2_2 - temp_CH1_2

fig2,(ax,ax2) = plt.subplots(2,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.plot(time_2,temp_CH1_2,'C0-',label='CH1')
ax.plot(time_2,temp_CH2_2,'C1-',label='CH2')#

if axin_on:
    axin = ax.inset_axes([1/3, 0.08, 0.64, 0.40])
    axin.plot(time_2[1120:1200],temp_CH1_2[1120:1200],'C0.-')
    axin.plot(time_2[1120:1200],temp_CH2_2[1120:1200],'C1.-')#
    mark_inset(ax, axin, loc1=3, loc2=4)
    # axin2 = ax2.inset_axes([0.4, 0.06, 0.59, 0.4])
    # axin2.plot(time_2[-50:],delta_2[-50:],'C2.-')
    # mark_inset(ax2, axin2, loc1=1, loc2=2)
    for b in [axin,axin2]:
        b.grid()

max_delta = np.nonzero(delta_2==max(delta_2))[0][0]
ax2.plot(time_2,delta_2,'C2-',label=f'ΔT = CH2 - CH1\nmax ΔT = {delta_2[max_delta]} °C')#

for a in [ax,ax2]:
    a.grid()
    a.legend(loc='best',shadow=True,frameon=True,ncol=1)
    a.set_ylabel('T (°C)')

ax2.set_xlabel('t (s)')
plt.suptitle('Despues de calibrar')


# %% Salvo figuras

fig1.savefig('precalibracion_T_dT.png', dpi=300)
fig2.savefig('postcalibracion_T_dT.png', dpi=300)

# %%
