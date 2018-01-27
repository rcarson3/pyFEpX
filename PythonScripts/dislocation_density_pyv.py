#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:49:20 2017

@author: robertcarson
"""

import numpy as np
import Rotations as rot
import FePX_Data_and_Mesh as fepxDM
import FiniteElement as fe

#%%
#fileLoc = '/Users/robertcarson/Research_Local_Code/Output/LOFEM_STUDY/n456_cent/low/'
#fileLoc = '/media/robert/My Passport for Mac/Simulations/LOFEM_Study/n456_cent_m15/mid_txt/'
#fileLoc = '/home/rac428/Outputs/LOFEM_Study/n456_cent_uori_m15/low_txt/'
#fileLoc = '/media/robert/DataDrives/LOFEM_Study/n456_NF/mono/low_txt/'
#fileLoc = '/Users/robertcarson/Research_Local_Code/fepx_robert/Examples/ControlMode/LOFEM_REFACTOR2/data/'
fileLoc = '/Volumes/My Passport for Mac/Simulations/LOFEM_Study/n456_cent_m15/mid_txt/'
fileName = 'n456-cent-rcl04'
#fileName = 'n456_nf_raster_L2_r1_v2_rcl075'
#fileName = 'n6'
fBname = 'gr_dd'

#fileLoc = '/media/robert/DataDrives/n1k_pois_iso_reg_pt2/'
#fileName = 'n1k-id6k-rcl05'


nproc = 64
#nsteps = 19
#nsteps = 46
#nsteps = 19
#nsteps = 43
nsteps = 44
#nsteps = 52
#nsteps = 64
#nsteps = 86

frames = np.arange(0,nsteps)
#Uncomment the below line for the LOFEM Refactored data sets
#then make sure to comment out the line below it.
#mesh = fepxDM.readMesh(fileLoc, fileName, LOFEM = True)
mesh = fepxDM.readMesh(fileLoc, fileName)

#ngrains = 6
ngrains = 456
#ngrains = 1000

grains = np.r_[1:(ngrains+1)]

#%%
#From here on uncomment commented lines to run code on the LOFEM Refactored
#data
print('About to start processing data')
kor = 'rod'
#ldata = fepxDM.readLOFEMData(fileLoc, nproc, lofemData=['strain', 'ang'])
print('Starting to read DISC data')
data = fepxDM.readData(fileLoc, nproc, fepxData=['adx'])
print('Finished Reading DISC data')

#%%
#
#gconn = np.asarray([], dtype='float64')
#gconn = np.atleast_2d(gconn)
#gupts = np.asarray([], dtype=np.int32)
#guelem = np.asarray([], dtype=np.int32)
#
#se_bnds = np.zeros((ngrains*2), dtype='int32')
#se_el_bnds = np.zeros((ngrains*2), dtype='int32')
#
#st_bnd = 0
#en_bnd = 0
#
#st_bnd2 = 0
#en_bnd2 = 0
#
#for i in grains:
#    
#    lcon, lcrd, lupts, luelem = fe.localConnectCrd(mesh, i)
#    st_bnd = en_bnd
#    en_bnd = st_bnd + lupts.shape[0]
#    
#    j = (i - 1) * 2
#    
#    se_bnds[j] = st_bnd
#    se_bnds[j+1] = en_bnd
#    
#    st_bnd2 = en_bnd2
#    en_bnd2 = st_bnd2 + luelem.shape[0]
#    
#    j = (i - 1) * 2
#    
#    se_el_bnds[j] = st_bnd2
#    se_el_bnds[j+1] = en_bnd2
#    
#    gconn, gupts, guelem = fe.concatConnArray(gconn, lcon, gupts, lupts, guelem, luelem) 
#
#npts = gupts.shape[0]
#nelem = guelem.shape[0]
#
##%%
#
#gconn2 = np.asarray([], dtype='float64')
#gconn2 = np.atleast_2d(gconn2)
#gupts2 = np.asarray([], dtype=np.int32)
#guelem2 = np.asarray([], dtype=np.int32)
#
#se_bnds2 = np.zeros((ngrains*2), dtype='int32')
#se_el_bnds2 = np.zeros((ngrains*2), dtype='int32')
#
#st_bnd = 0
#en_bnd = 0
#
#st_bnd2 = 0
#en_bnd2 = 0
#
#for i in grains:
#    
#    lcon, lupts, luelem = fe.localGrainConnectCrd(mesh, i)
#    st_bnd = en_bnd
#    en_bnd = st_bnd + lupts.shape[0]
#    
#    j = (i - 1) * 2
#    
#    se_bnds2[j] = st_bnd
#    se_bnds2[j+1] = en_bnd
#    
#    st_bnd2 = en_bnd2
#    en_bnd2 = st_bnd2 + luelem.shape[0]
#    
#    j = (i - 1) * 2
#    
#    se_el_bnds2[j] = st_bnd2
#    se_el_bnds2[j+1] = en_bnd2
#    
#    gconn2, gupts2, guelem2 = fe.concatConnArray(gconn2, lcon, gupts2, lupts, guelem2, luelem) 
#
#npts2 = gupts2.shape[0]
#nelem2 = guelem2.shape[0]
#
##%%
##  
#gr_angs = np.zeros((1, npts,  nsteps), dtype='float64')
#lofem_angs = np.zeros((1, nelem,  nsteps), dtype='float64')
#disc_angs = np.zeros((1, nelem,  nsteps), dtype='float64')
##
#origin = np.zeros((3,1), dtype='float64')

#%%

iso_dndx = fe.iso_dndx()
lmat = fe.get_l2_matrix()
nnpe = 10
dim = 3
ngdot = 12

for i in grains:
    print('###### Starting Grain Number '+str(i)+' ######')
    
    gdata = fepxDM.readGrainData(fileLoc, i, frames=None, grData=['ang'])
    
    lcon, lcrd, ucon, uelem = fe.localConnectCrd(mesh, i)
    
    nel = lcon.shape[1]
    ncrd = ucon.shape[0]
    
    indlog = mesh['grains'] == i
    
#    lcon, lcrd, ucon, uelem = fe.localConnectCrd(mesh, i)
#    lcon2, ucon2, uelem2 = fe.localGrainConnectCrd(mesh, i)
#    
#    nel = lcon.shape[1]
#    npts = ucon.shape[0]
#    
#    indlog = mesh['grains'] == i
#    indlog2 = mesh['crd_grains'] == i
    
    
    strgrnum = np.char.mod('%4.4d', np.atleast_1d(i))[0]
    
    elem_crd = np.zeros((nnpe, dim, nel), dtype='float64', order='F')
    crd = np.zeros((ncrd, dim), dtype='float64', order='F')
    
    el_vec = np.zeros((dim, nnpe, nel), dtype='float64', order='F')
    vec_grad = np.zeros((dim, dim, nel), dtype='float64', order='F')
    nye_ten = np.zeros((dim, dim, nel), dtype='float64', order='F')
    density = np.zeros((ngdot, nel), dtype='float64', order='F')
    loc_dndx = np.zeros((dim, nnpe, nel), dtype='float64', order='F')
    det_qpt = np.zeros((nel), dtype='float64', order='F')
    
    
    ang_axis = np.zeros((dim, ncrd, nsteps), dtype='float64', order='F')
    
    #Realized the Nye Tensor is based upon the axial vector and not
    #a Rod vec so need to update the code to take note of this being the case
    for j in range(nsteps):
        ang_axis[:,:,j] = rot.AngleAxisOfRod(gdata['angs'][:,:,j])
    
    for j in range(nsteps):
        
        crd[:,:] = np.squeeze(data['coord'][:,ucon, j]).T 
        
        for k in range(nel):
            elem_crd[:, :, k] = crd[lcon[:, k], :]
            el_vec[:, :, k] = ang_axis[:, lcon[:, k], j]
            
        loc_dndx[:,:,:], det_qpt[:] = fe.local_gradient_shape_func(iso_dndx, elem_crd, 4)
        vec_grad = fe.get_vec_grad(el_vec, loc_dndx)
        nye_ten = fe.get_nye_tensor(vec_grad)
        
        density = fe.get_l2_norm_dd(nye_ten, lmat)
        
        
        with open(fileLoc+fBname+strgrnum+'.data','ab') as f_handle:
            f_handle.write(bytes('%Grain step'+str(j)+'\n','UTF-8'))
            for k in range(nel):
                np.savetxt(f_handle,density[:, k], newline=' ')
                f_handle.write(bytes('\n','UTF-8'))
                
#        print('Grain #'+str(i)+'% done:  {:.3f}'.format(((j+1)/nsteps)))
        
           