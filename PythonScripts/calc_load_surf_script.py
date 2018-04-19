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

#The file locations that we'll be pulling from
fileLoc = '/Users/robertcarson/Research_Local_Code/Output/LOFEM_STUDY/n456_cent/low/'
fileLoc = '/Volumes/My Passport for Mac/Simulations/LOFEM_Study/n456_cent_c10/low_txt/'
fileLoc = '/media/robert/My Passport for Mac/Simulations/LOFEM_Study/n456_cent_c10/low_txt/'
fileLoc = '/media/robert/DataDrives/LOFEM_Study/n456_NF/mono/low_txt/'
fileName = 'n456-cent-rcl05'
fileName = 'n456_nf_raster_L2_r1_v2_rcl075'

#Getting info about the number of processors and steps in the simulation
nproc = 64
nsteps = 42

mesh = fepxDM.readMesh(fileLoc, fileName)
#Reading in where the macroscopic strain spots should be. This is assumming that the simulation
#was conducted in displacement control. If it wasn't then one would need to calculate that from the post.force
#series of files where the time step is provided. 
mstrain = np.genfromtxt(fileLoc+'mstrain.txt', comments = '%')

#%%
#Here we need to read in our stress and nodal coordinate data.
data = fepxDM.readData(fileLoc, nproc, None, ['adx','stress'], False) #,'gammadot', 'crss'

#%%
#Getting our 2D element quadrature point data in order to find surface info from our elemental data
qp2d, wt2d, sf, sfgd = fe.surface_quad_tet()
#Creating the transpose of the shape function gradient 
sfgdt = np.swapaxes(sfgd, 0, 1)
#
##%%
#Getting out what the coords for our mesh and the surface connectivity
scrds = mesh['crd']
sconn = mesh['surfaceNodes']
#Telling it what surface that we want to be dealing with
surf = 'z2'
#Getting the connectivity array of our sample surface in terms of our global coords and then a local version
#where the global connectivity array is renumbered such that our first index is now 0.
#See the function to see how things are laid out in the arrays
gconn, lconn = fe.surfaceConn(scrds, sconn, surf)

#%%
#Initializing a load and surface arrays
load = np.zeros((3, nsteps))
area = np.zeros(nsteps) 
#Going through all of the steps and finding our surface elements
for i in range(nsteps):
    vec = np.unique(gconn[1:7, :])
    #Getting all of the coords that we need in our current frame
    scrds = data['coord'][:, vec, i]
    #Grabbing the stress state from the elements that are along that surface
    sig = data['stress'][:, gconn[0, :], i]
    #We calculate the load and area of the surface here
    load[:, i], area[i] = fe.surfaceLoadArea(scrds, lconn[1:7,:], sig, wt2d, sfgdt)
    
# %%
#This is now doing the same as the above but just calculating it for our LOFEM method
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
#Calculating our engineering strain. The area should really be the initial area, but
#I've decided to use my first step instead since I know it's pretty early on in the elastic regime where
#I'm at less than 0.01% strain so we should see very small differences in the two areas.
estress = loload[2,:]/area[0]
estress2 = load[2,:]/area[0]
#Here we're calculating the true stress
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
#Calculating the true strain here
tstrain = np.log(mstrain + 1)

ax.plot(tstrain, tstress2, label='Discrete Xtal Lattice Orientation Update')

ax.plot(tstrain, tstress, label='LOFEM Xtal Lattice Orientation Update')

ax.set_ylabel('Macroscopic true stress [MPa]')
ax.set_xlabel('Macroscopic true strain [-]')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, ncol=1)

fig.show()
plt.show()

#We can save off our stress-strain curve if we'd like
picLoc = 'lofem_true_ss_nf_mono_curve.png'
fig.savefig(picLoc, dpi = 300, bbox_inches='tight')




