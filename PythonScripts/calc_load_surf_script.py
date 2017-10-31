#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:56:35 2017

@author: robertcarson
"""

import numpy as np
import FePX_Data_and_Mesh as fepxDM
import FiniteElement as fe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

fileLoc = '/Users/robertcarson/Research_Local_Code/Output/LOFEM_STUDY/n456_cent/low/'
fileLoc = '/Volumes/My Passport for Mac/Simulations/LOFEM_Study/n456_cent_c10/low_txt/'
fileLoc = '/media/robert/My Passport for Mac/Simulations/LOFEM_Study/n456_cent_c10/low_txt/'
fileLoc = '/media/robert/DataDrives/LOFEM_Study/n456_NF/mono/low_txt/'
fileName = 'n456-cent-rcl05'
fileName = 'n456_nf_raster_L2_r1_v2_rcl075'

nproc = 64
nsteps = 42

mesh = fepxDM.readMesh(fileLoc, fileName)

mstrain = np.genfromtxt(fileLoc+'mstrain.txt', comments = '%')

#%%

data = fepxDM.readData(fileLoc, nproc, None, ['adx','stress'], False) #,'gammadot', 'crss'

#%%

qp2d, wt2d, sf, sfgd = fe.surface_quad_tet()

sfgdt = np.swapaxes(sfgd, 0, 1)
#
##%%

scrds = mesh['crd']
sconn = mesh['surfaceNodes']
surf = 'z2'

gconn, lconn = fe.surfaceConn(scrds, sconn, surf)

#%%
load = np.zeros((3, nsteps))
area = np.zeros(nsteps) 

for i in range(nsteps):
    vec = np.unique(gconn[1:7, :])
    scrds = data['coord'][:, vec, i]
    sig = data['stress'][:, gconn[0, :], i]
    load[:, i], area[i] = fe.surfaceLoadArea(scrds, lconn[1:7,:], sig, wt2d, sfgdt)
    
# %%

ldata = fepxDM.readLOFEMData(fileLoc, nproc, 15, None, ['stress']) # 'stress',,'gammadot','crss'

#%%
loload = np.zeros((3, nsteps))
loarea = np.zeros(nsteps) 

for i in range(nsteps):
    vec = np.unique(gconn[1:7, :])
    scrds = data['coord'][:, vec, i]
    sig = ldata['stress'][:, gconn[0, :], i]
    loload[:, i], loarea[i] = fe.surfaceLoadArea(scrds, lconn[1:7,:], sig, wt2d, sfgdt)
    
#%%

#mstrain = mstrain[0:nsteps]
#mstrain[nsteps-1] = 0.128

estress = loload[2,:]/area[0]
estress2 = load[2,:]/area[0]

tstress = loload[2,:]/area[:]
tstress2 = load[2,:]/area[:]

#%%

fig, ax = plt.subplots(1)

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width*1.4, box.height*1.4])

ax.plot(mstrain, estress2, label='Discrete Xtal Lattice Orientation Update')

ax.plot(mstrain, estress, label='LOFEM Xtal Lattice Orientation Update')

ax.set_ylabel('Macroscopic engineering stress [MPa]')
ax.set_xlabel('Macroscopic engineering strain [-]')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, ncol=1)

fig.show()
plt.show()

picLoc = 'lofem_ss_nf_mono_curve.png'
fig.savefig(picLoc, dpi = 300, bbox_inches='tight')

#%%

fig, ax = plt.subplots(1)

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width*1.4, box.height*1.4])

tstrain = np.log(mstrain + 1)

ax.plot(tstrain, tstress2, label='Discrete Xtal Lattice Orientation Update')

ax.plot(tstrain, tstress, label='LOFEM Xtal Lattice Orientation Update')

ax.set_ylabel('Macroscopic true stress [MPa]')
ax.set_xlabel('Macroscopic true strain [-]')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, ncol=1)

fig.show()
plt.show()

picLoc = 'lofem_true_ss_nf_mono_curve.png'
fig.savefig(picLoc, dpi = 300, bbox_inches='tight')

#%%

#fig, ax = plt.subplots(1)
#
#box = ax.get_position()
#ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                 box.width*1.4, box.height*1.4])
#
#loadDiff = (estress - estress2)/estress2
#
#ax.plot(mstrain, loadDiff)
#
#ax.set_ylabel('Difference in DISC and LOFEM macro stress [%]')
#ax.set_xlabel('Macroscopic engineering strain [-]')
#
#fig.show()
#plt.show()
#
#picLoc = 'lofem_diff_percent_curve.png'
#fig.savefig(picLoc, dpi = 300, bbox_inches='tight')
#
##%%
#
#
#
##%%
#gt = np.zeros((12, 44))
#gprob = np.ones((12, 44))
#
#zeros = np.zeros(data['gammadot'].shape[1])
#
#dgammadot = data['gammadot'] - ldata['gammadot']
#
#nelems = data['crss'].shape[1]
#for i in range(nsteps-3):
#    j = i + 3
#    for k in range(12):
#        gt[k, j], gprob[k, j] = stats.mannwhitneyu(np.squeeze(dgammadot[k,:,j]), zeros, alternative='two-sided')
#
##t, prob = stats.ttest_ind(np.abs(data['gammadot']), np.abs(ldata['gammadot']), axis=1)
##
##%%
#fig, ax = plt.subplots(1)
#ind = gprob > 3e-7
#ax.imshow(ind, interpolation='none', cmap='viridis')
#ax.set_xlabel('Load Step number')
#ax.set_ylabel('Gammadot Slip System #')
#ax.yaxis.set_ticks(np.arange(0, 12, 1))
#ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2d'))
#labels = [str(i+1) for i in range(12)]
#ax.yaxis.set_ticklabels(labels)
##ax.grid()
#
#
#fig.show()
#plt.show()
#
#picLoc = 'gammadot_mwu.png'
#fig.savefig(picLoc, dpi = 300, bbox_inches='tight')
#
##%%
#ct = np.zeros(44)
#cprob = np.ones(44)
#nelems = data['crss'].shape[1]
#dcrss = data['crss'] - ldata['crss']
#for i in range(nsteps-3):
#    j = i + 3
#    ct[j], cprob[j] = stats.mannwhitneyu(np.squeeze(dcrss[:,:,j]), zeros, alternative='two-sided')
#
##%%
#fig, ax = plt.subplots(1)
#ind = cprob > 3e-7
#ax.imshow(np.atleast_2d(ind), interpolation='none', cmap='viridis')
#ax.set_xlabel('Load Step number')
#ax.set_ylabel('CRSS')
#ax.yaxis.set_ticks(np.arange(0, 1, 1))
#labels = [str(0)]
#ax.yaxis.set_ticklabels(labels)
#ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1d'))
#
#
##fig.show()
#plt.show()
#
#picLoc = 'crss_mwu.png'
#fig.savefig(picLoc, dpi = 300, bbox_inches='tight')




