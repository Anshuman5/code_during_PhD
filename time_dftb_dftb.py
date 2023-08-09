from __future__ import print_function
#import os
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook

wb = load_workbook('time_dft_dftb_2.xlsx')
ws = wb['Sheet2']
sheet_name = 'Sheet2'
dft_factor = 1#5

df = pd.read_excel('time_dft_dftb_2.xlsx', sheet_name=sheet_name, usecols=range(0,26))
#df = df.drop(0, axis=0)
time_dft =  df['total_time_dft']
total_time_dft =  sum(df['total_time_dft'])
time_dft_zno =  df['total_time_dft_zno_tol']
total_time_dft_zno =  sum(df['total_time_dft_zno_tol'])

time_dftb_zno =  df['total_time_dftb_zno']
total_time_dftb_zno =  sum(df['total_time_dft_zno_tol'])


time_dftb_pbc =  df['total_time_dftb_pbc']
total_time_dftb_pbc = sum(df['total_time_dftb_pbc'])
time_dftb_pbc16 =  df['total_time_dftb_pbc_16_lv']
#total_time_dftb_pbc_16cpus
total_time_dftb_pbc16 = sum(df['total_time_dftb_pbc_16_lv']) 
time_dftb_matsci =  df['total_time_dftb_matsci']
total_time_dftb_matsci = sum(df['total_time_dftb_matsci'])
time_dftb_matsci16 =  df['total_time_dftb_matsci_16cpus_lv']
#total_time_dftb_matsci_16cpus
total_time_dftb_matsci16 = sum(df['total_time_dftb_matsci_16cpus_rlv']) 
time_dftb_skfiv =  df['total_time_dftb_skfiv_16_lv']
#total_time_dftb_skfiv_16
total_time_dftb_skfiv =  sum(df['total_time_dftb_skfiv_16_lv']) 
comp = df['comp']

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig, axs = plt.subplots(2, 2, figsize=(15,12))
axs[0, 0].scatter(comp,time_dft, color = 'b', label='DFT')

axs[0, 1].scatter(comp,time_dft/time_dftb_skfiv, color = 'b', label='DFT/DFTB (SKfIV)')
axs[1, 0].scatter(comp,time_dft/time_dftb_pbc, color = 'b', label='DFT/DFTB (pbc)')
axs[1, 1].scatter(comp,time_dft/time_dftb_matsci, color = 'b', label='DFT/DFTB (matsci)')

axs[0, 0].set(ylabel='Total Time (sec)')
axs[0, 0].yaxis.get_label().set_fontsize(16)

axs[0, 1].set(ylabel='DFT Time / DFTB Time')
axs[0, 1].yaxis.get_label().set_fontsize(16)

axs[1, 0].set(ylabel='DFT Time / DFTB Time')
axs[1, 0].yaxis.get_label().set_fontsize(16)

axs[1, 1].set(ylabel='DFT Time / DFTB Time')
axs[1, 1].yaxis.get_label().set_fontsize(16)

axs[1, 0].set(xlabel ='$x$ in Si$_{2}$C$_{2(1-x)}$')
axs[1, 0].xaxis.get_label().set_fontsize(16)

axs[1, 1].set(xlabel ='$x$ in Si$_{2}$C$_{2(1-x)}$' )
axs[1, 1].xaxis.get_label().set_fontsize(16)

axs[0, 0].legend(loc=0, frameon= True, prop={'size': 12})
axs[0, 1].legend(loc=0, frameon= True, prop={'size': 12})
axs[1, 0].legend(loc=0, frameon= True, prop={'size': 12})
axs[1, 1].legend(loc=0, frameon= True, prop={'size': 12})
print ('total dft time for SiC (16 cpus): {0:.2f} days'.format(total_time_dft/(3600*24)))
print ('total dftb time (matsci) for SiC (4 cpus): {0:.2f} days'.format(total_time_dftb_matsci/(3600*24)))
print ('total dftb time pbc for SiC (4 cpus): {0:.2f} days'.format(total_time_dftb_pbc/(3600*24)))
#print ('total dftb time pbc for SiC (12 cpus): {0:.2f} days'.format(total_time_dftb_skfiv/(3600*24)))
print ('total dftb time (pbc) for SiC (16 cpus): {0:.2f} days'.format(total_time_dftb_pbc16/(3600*24)))
print ('total dft time for ZnO (8 cpus): {0:.2f} days'.format(total_time_dft_zno/(3600*24)))
print ('total dftb time for ZnO (8 cpus): {0:.2f} days'.format(total_time_dftb_zno/(3600*24)))

plt.figure(figsize=(12, 8))
N = 2
dft_time1 = (total_time_dft/(3600*24), total_time_dft_zno/(3600*24))
dftb_time1 = (total_time_dftb_pbc16/(3600*24), total_time_dftb_zno/(3600*24))

ind = np.arange(N) 
width = 0.35       
plt.bar(ind, dft_time1, width, label='DFT')
plt.bar(ind + width, dftb_time1, width,
    label='DFTB')

plt.ylabel('Time (days)', fontsize=24)


plt.xticks(ind + width / 2, ('SiC', 'ZnO'))
plt.legend(loc='best', frameon= True, prop={'size': 18})
plt.show()

plt.figure(figsize=(12, 8))

N = 1
dftb_time1 = (total_time_dftb_pbc16/(3600*24))
dftb_time2 = (total_time_dftb_matsci/(3600*24))
dftb_time3 = (total_time_dftb_skfiv/(3600*24))

ind1 = np.arange(N) 
width = 0.15       
plt.bar(ind1, dftb_time1, width, label='DFTB_pbc')
plt.bar(ind1+width, dftb_time2, width, label='DFTB_matsci')
plt.bar(ind1+2*width, dftb_time3, width, label='DFTB_skfiv')


plt.ylabel('Time (days)', fontsize=24)
plt.xticks([])
plt.xlabel('SiC',fontsize=24)
#plt.title('')

#plt.xticks(ind1+width /3, ('SiC'))
plt.legend(loc='best', frameon= True, prop={'size': 18})

plt.show()
