#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 13:30:15 2017

@author: robertcarson
"""

import FePX_Data_and_Mesh as fepxDM
import numpy as np
import FiniteElement as fe
from latorifem import mainlatprogram as latfem


floc = '/Users/robertcarson/Research_Local_Code/fepx_master/Examples/ControlMode/'
fname = 'n2_small'
#fBname = 'gr_agamma'

mesh = fepxDM.readMesh(floc,fname)
#
nproc = 2
nframe = 14
ngrains = 1

#%%

data = fepxDM.readData(floc, nproc, None, ['stress_q', 'adx'])
#
##%%
#ncrds = mesh['crd'].shape[1]
nslip = data['stress_q'].shape[1]


Nsf = fe.sfmat()

NTNsf, NT = fe.sftsfmat()

#ag_nodal = np.zeros((nslip, ncrds, nframe))
#residuals = np.zeros((nslip, nframe))
#
for grnum in range(ngrains):
    print("Starting grain number: "+str(grnum+1))
    lcon, lcrd,  upts, uelem = fe.localConnectCrd(mesh, grnum+1)
    indlog = mesh['grains'] == grnum+1
    agamma = data['stress_q'][:,:,indlog,:]
    ncrds = lcrd.shape[1]
    
    nel = lcon.shape[1]
    
    amat2 = fe.gr_lstq_amat(lcon, Nsf, ncrds)
    
    ag_nodal = np.zeros((nslip, ncrds, nframe))
    ag_nodal2 = np.zeros((nslip, ncrds, nframe))
    residuals = np.zeros((nslip, nframe))
    
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
    
    for j in range(nframe):
        
        print("Starting frame: "+str(j)+" out of:" + str(nframe))
        
        crd[:,:] = np.squeeze(data['coord'][:,upts, j]).T
        latfem.setdata(strain, gdot, vel, lcon.T, crd, grod0, nel1=nel-1, dim1=dim-1, ngd1=ngdot-1, ncr1=ncrd-1, nnp=nnpe, ncvc1=ncvec-1)
        qpt_det = latfem.getjacobiandet(nel1=nel-1, nqp1=14)
        amat = fe.superconvergence_mat(NTNsf, qpt_det, lcon.T, ncrds)
        bvec = fe.superconvergence_vec(NT, qpt_det, lcon.T, agamma[:, :, :, j], ncrds)
        ag_nodal[:,:,j] = fe.superconvergnce_solve(amat, bvec)
        ag_nodal2[:, :, j], residuals[:, j] = fe.gr_lstq_solver(amat2, agamma[:, :, :, j], ncrds)
##    
## 
    latfem.deallocate_vars()   
#    strgrnum = np.char.mod('%4.4d', np.atleast_1d(grnum+1))[0]
#    with open(floc+fBname+strgrnum+'.data','ab') as f_handle:
#        for j in range(nframe):
#            f_handle.write(bytes('%Frame Number'+str(j)+'\n','UTF-8'))
#            for k in range(ncrds):
#                np.savetxt(f_handle,np.squeeze(ag_nodal[:, k, j]), newline = ' ')
#                f_handle.write(bytes('\n','UTF-8'))
#                
#    print("Finished grain number: "+str(grnum+1))