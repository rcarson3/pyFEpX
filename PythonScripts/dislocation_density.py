#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:49:20 2017

@author: robertcarson
"""

import numpy as np
import FePX_Data_and_Mesh as fepxDM
import FiniteElement as fe
from latorifem import mainlatprogram as latfem

fileLoc = '/Users/robertcarson/Research_Local_Code/Output/LOFEM_STUDY/n456_cent/low/'
#fileLoc = '/Volumes/My Passport for Mac/Simulations//LOFEM_Study/n456_cent_c10/low_txt/'
fileLoc = '/media/robert/My Passport for Mac/Simulations/LOFEM_Study/n456_cent_c03/low_txt/'
fileLoc = '/media/robert/DataDrives/LOFEM_Study/n456_NF/mono/low_txt/'
fileName = 'n456-cent-rcl05'
fileName = 'n456_nf_raster_L2_r1_v2_rcl075'
fBname = 'gr_dd'

nproc = 64
nsteps = 42

frames = np.arange(0,nsteps)

mesh = fepxDM.readMesh(fileLoc,fileName)

ngrains = 456

grains = np.r_[1:(ngrains+1)]

#%%

print('About to start processing data')
kor = 'rod'
print('Starting to read DISC data')
data = fepxDM.readData(fileLoc, nproc, fepxData=['adx'])
print('Finished Reading DISC data')

#%%

for i in grains:
    print('###### Starting Grain Number '+str(i)+' ######')
    
    gdata = fepxDM.readGrainData(fileLoc, i, frames=None, grData=['ang'])
    
    lcon, lcrd, ucon, uelem = fe.localConnectCrd(mesh, i)
    
    nel = lcon.shape[1]
    
    indlog = mesh['grains'] == i
    strgrnum = np.char.mod('%4.4d', np.atleast_1d(i))[0]

    ncrd = lcrd.shape[1]
    ngdot = 12
    ncvec = ncrd*3
    dim = 3
    nnpe = 9
    kdim1 = 29
    
    gdot = np.zeros((nel,12))
    vel = np.zeros((ncrd, 3))
    strain = np.zeros((nel,3,3))
    gdot = np.zeros((nel,12))
    density = np.zeros((12, nel))
    grod0 = np.zeros((ncvec, 1))
    ang = np.zeros((ncrd, 3))
    crd = np.zeros((ncrd, 3))
    
    latfem.initializeall(nel, ngdot, ncrd)
    
    for j in range(nsteps):
        
        crd[:,:] = np.squeeze(data['coord'][:,ucon, j]).T
        ang[:,:] = np.squeeze(gdata['angs'][:,:,j]).T
        
        latfem.setdata(strain, gdot, vel, lcon.T, crd, grod0, nel1=nel-1, dim1=dim-1, ngd1=ngdot-1, ncr1=ncrd-1, nnp=nnpe, ncvc1=ncvec-1)
        
        density = latfem.get_disc_dens(nel-1, ang, nc1=ncrd-1, dim1=dim-1)
        
        with open(fileLoc+fBname+strgrnum+'.data','ab') as f_handle:
            f_handle.write(bytes('%Grain step'+str(j)+'\n','UTF-8'))
            for k in range(nel):
                np.savetxt(f_handle,density[:, k], newline=' ')
                f_handle.write(bytes('\n','UTF-8'))
                
        print('Grain #'+str(i)+'% done:  {:.3f}'.format(((j+1)/nsteps)))
        
        
    latfem.deallocate_vars()   
