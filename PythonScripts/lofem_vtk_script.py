#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:52:05 2017

@author: robertcarson
"""

import numpy as np
import fepx_vtk as fvtk
#import graph_cc_dfs as gcdfs
import FePX_Data_and_Mesh as fepxDM
import textadapter as ta
import FiniteElement as fe
#import sklearn.preprocessing as sklpp

#fileLoc = '/Users/robertcarson/Research_Local_Code/Output/n1k_pois_iso_reg/'
#fileLoc = '/Volumes/My Passport for Mac/Simulations/hires/n500_pois_iso/'
#fileName = 'n1k-id6k-rcl05'
#fileName = 'n500-id6k'

#nproc = 64
#nsteps = 44
#fileLoc = '/Users/robertcarson/Research_Local_Code/Output/LOFEM_STUDY/n456_cent/low/'
#fileLoc = '/Volumes/My Passport for Mac/Simulations/LOFEM_Study/n456_cent_c10/low_txt/'
fileLoc = '/media/robert/My Passport for Mac/Simulations/LOFEM_Study/n456_cent_m15/low_txt/'
#fileLoc = '/media/robert/DataDrives/LOFEM_Study/n456_NF/mono/low_txt/'
fileName = 'n456-cent-rcl05'
#fileName = 'n456_nf_raster_L2_r1_v2_rcl075'
fBname = 'grainData'

#fileVTK = '/Users/robertcarson/Research_Local_Code/Output/LOFEM_STUDY/n456_cent/'
fileVTK = '/media/robert/Data/SimulationData/LOFEM_Data/n456_cent/mono/'
#fileVTK = '/Users/robertcarson/Research_Local_Code/Output/'
#fileVTK = '/media/robert/Data/SimulationData/LOFEM_Data/n456_NF/mono/'
fVTKLOFEM = fileVTK + 'lofem_mono'
fVTKDISC = fileVTK + 'disc_mono'

fVTKLGROUP = fVTKLOFEM + '_group'
fVTKDGROUP = fVTKDISC + '_group'
#
nproc = 64
nsteps = 44

frames = np.arange(0,nsteps)

mesh = fepxDM.readMesh(fileLoc,fileName)

ngrains = 456
nangs = 3
nss = 12
#ngrains = 1000
#ngrains = 500

grains = np.r_[1:(ngrains+1)]

nels = mesh['grains'].shape[0]


#%%
r2d = 180/np.pi
print('About to start processing data')
kor = 'rod'
ldata = fepxDM.readLOFEMData(fileLoc, nproc, lofemData=['strain','stress','crss'])
#print('Finished Reading LOFEM data')
print('Starting to read DISC data')
data = fepxDM.readData(fileLoc, nproc, fepxData=['adx', 'strain','stress','crss'], restart=False)#, 'ang'])
print('Finished Reading DISC data')
#%%
misori = ta.genfromtxt(fileLoc+fBname+'diff.emisori', comments='%')
dmisori = ta.genfromtxt(fileLoc+fBname+'_DISC.cmisori', comments='%')
gr_cmisori = ta.genfromtxt(fileLoc+fBname+'.cmisori', comments='%')
#alpha = ta.genfromtxt(fileLoc+fBname+'.alpha', comments='%')

dmisori = dmisori.reshape((nsteps, nels)).T*r2d
misori = misori.reshape((nsteps, nels)).T*r2d
#gr_cmisori = gr_misori.reshape((nsteps, nels)).T*r2d
#alpha = alpha.reshape((nsteps, nels)).T
print('Finished Reading in Misori, Gr Misori, and Alpha')

#%%

gconn = np.asarray([], dtype='float64')
gconn = np.atleast_2d(gconn)
gupts = np.asarray([], dtype=np.int32)
guelem = np.asarray([], dtype=np.int32)

grains_elem = np.asarray([], dtype=np.int32)
grains_pts = np.asarray([], dtype=np.int32)

se_bnds = np.zeros((ngrains*2), dtype='int32')
se_el_bnds = np.zeros((ngrains*2), dtype='int32')

st_bnd = 0
en_bnd = 0

st_bnd2 = 0
en_bnd2 = 0

for i in grains:
    
    lcon, lcrd, lupts, luelem = fe.localConnectCrd(mesh, i)
    st_bnd = en_bnd
    en_bnd = st_bnd + lupts.shape[0]
    
    j = (i - 1) * 2
    
    se_bnds[j] = st_bnd
    se_bnds[j+1] = en_bnd
    
    st_bnd2 = en_bnd2
    en_bnd2 = st_bnd2 + luelem.shape[0]
    
    j = (i - 1) * 2
    
    se_el_bnds[j] = st_bnd2
    se_el_bnds[j+1] = en_bnd2
    
    lcon = fvtk.fepxconn_2_vtkconn(lcon)
    gconn, gupts, guelem = fe.concatConnArray(gconn, lcon, gupts, lupts, guelem, luelem) 

npts = gupts.shape[0]
nelem = guelem.shape[0]

  
gr_angs = np.zeros((nangs, npts,  nsteps), dtype='float64')
gr_gdot = np.zeros((nss, npts, nsteps), dtype='float64')
gr_gamma = np.zeros((nss, npts, nsteps), dtype='float64')
gr_dd = np.zeros((nss, nelem, nsteps), dtype='float64')
gr_cmisori = gr_cmisori.reshape((nsteps, npts)).T*r2d
   
#%%    

for i in grains:
    print('###### Starting Grain Number '+str(i)+' ######')
    
    gdata = fepxDM.readGrainData(fileLoc, i, frames=None, grData=['ang','gdot','gamma','dd'])
    j = (i - 1) * 2
    k = j + 2
    ind = se_bnds[j:k]
    ind2 = se_el_bnds[j:k]
    gr_gamma[:, ind[0]:ind[1], :] = gdata['gamma']
    gr_gdot[:, ind[0]:ind[1], :] = gdata['gdot']
    gr_angs[:, ind[0]:ind[1], :] = gdata['angs']
    gr_dd[:, ind2[0]:ind2[1], :] = gdata['dd']
#%%    
evtk_conn, evtk_offset, evtk_type = fvtk.evtk_conn_offset_type_creation(gconn)
#
nelems = guelem.shape[0]
#
grains_elem = np.atleast_2d(mesh['grains'])
grains_elem = np.atleast_2d(grains_elem)
#nelems = grains_elem.shape[1]
#grains_pts = np.atleast_2d(grains_pts)


#%%

lofem_file = []
disc_file = []
#%%
#
sgr_gamma = np.zeros((4, gr_gamma.shape[1], nsteps))

sgr_gamma[0, :, :] = np.sum(gr_gamma[0:3, :, :], axis = 0)
sgr_gamma[1, :, :] = np.sum(gr_gamma[3:6, :, :], axis = 0)
sgr_gamma[2, :, :] = np.sum(gr_gamma[6:9, :, :], axis = 0)
sgr_gamma[3, :, :] = np.sum(gr_gamma[9:12, :, :], axis = 0)

sgr_gdot = np.zeros((4, gr_gamma.shape[1], nsteps))

sgr_gdot[0, :, :] = np.sum(gr_gdot[0:3, :, :], axis = 0)
sgr_gdot[1, :, :] = np.sum(gr_gdot[3:6, :, :], axis = 0)
sgr_gdot[2, :, :] = np.sum(gr_gdot[6:9, :, :], axis = 0)
sgr_gdot[3, :, :] = np.sum(gr_gdot[9:12, :, :], axis = 0)

ss_rho = np.zeros((4, gr_dd.shape[1], nsteps))

ss_rho[0, :, :] = np.sum(np.abs(gr_dd[0:3, :, :]), axis = 0)
ss_rho[1, :, :] = np.sum(np.abs(gr_dd[3:6, :, :]), axis = 0)
ss_rho[2, :, :] = np.sum(np.abs(gr_dd[6:9, :, :]), axis = 0)
ss_rho[3, :, :] = np.sum(np.abs(gr_dd[9:12, :, :]), axis = 0)

#%%
mStress = np.zeros((nels,nsteps), dtype="float64")
effStress = np.zeros((nels,nsteps), dtype="float64")
triax = np.zeros((nels,nsteps), dtype="float64")

for j in range(nsteps):
    stress = fepxDM.fixStrain(np.squeeze(ldata['stress'][:,:,j]).T)
    for i in range(nels):
        mStress[i,j] = 1/3*np.trace(stress[i,:,:])
        devStress = stress[i,:,:] - mStress[i,j]*np.eye(3)
        effStress[i,j] = np.sqrt(3/2*np.trace(np.dot(devStress,devStress.T)))
        triax[i,j] = mStress[i,j]/effStress[i,j]
#%%
#ngr_angs = np.zeros((gr_angs.shape))
#for i in range(nsteps):
#    ngr_angs[:,:,i] = sklpp.normalize(gr_angs[:,:,i], axis=0)

#%%
#npts = data['coord'].shape[1]
#guelem = np.r_[0:nelems]
#gupts = np.r_[0:npts]
#nsteps = 44

#cDict = {'grain_elem':'Scalars', 'misorientation':'Scalars', 'alpha':'Scalars', 'Grain_misorientation':'Scalars'}

loDict = {'stress':'Tensors', 'strain':'Tensors', 'grain_elem':'Scalars', 'crss':'Scalars', 'misorientation':'Scalars',
          'rho_n1':'Scalars', 'rho_n2':'Scalars', 'rho_n3':'Scalars', 'rho_n4':'Scalars', 'disc_misori':'Scalars',
          'mean_stress':'Scalars','deff_stress':'Scalars', 'triaxiality':'Scalars'}
cDict = {'stress':'Tensors', 'strain':'Tensors', 'grain_elem':'Scalars', 'crss':'Scalars'}
pDict = {'grain_rod':'Vectors', 'gr_gamma_n1':'Scalars', 'gr_gamma_n2':'Scalars', 'gr_gamma_n3':'Scalars', 'gr_gamma_n4':'Scalars',
         'gr_gdot_n1':'Scalars', 'gr_gdot_n2':'Scalars', 'gr_gdot_n3':'Scalars', 'gr_gdot_n4':'Scalars', 'gr_misori':'Scalars'}



for i in range(nsteps):
    print('##########Starting Step # '+str(i)+'##########')
    ldict = {}
    ddict = {}
    ldict2 = {}
    ddict2 = {}
    
    ldict2['grain_rod'] = np.ascontiguousarray(gr_angs[:,:,i])
    ldict2['gr_misori'] = np.ascontiguousarray(gr_cmisori[:,i])
    ldict2['gr_gamma_n1'] = np.ascontiguousarray(sgr_gamma[0,:,i])
    ldict2['gr_gamma_n2'] = np.ascontiguousarray(sgr_gamma[1,:,i])
    ldict2['gr_gamma_n3'] = np.ascontiguousarray(sgr_gamma[2,:,i])
    ldict2['gr_gamma_n4'] = np.ascontiguousarray(sgr_gamma[3,:,i])
    ldict2['gr_gdot_n1'] = np.ascontiguousarray(sgr_gdot[0,:,i])
    ldict2['gr_gdot_n2'] = np.ascontiguousarray(sgr_gdot[1,:,i])
    ldict2['gr_gdot_n3'] = np.ascontiguousarray(sgr_gdot[2,:,i])
    ldict2['gr_gdot_n4'] = np.ascontiguousarray(sgr_gdot[3,:,i])
    
    ldict['disc_misori'] = np.ascontiguousarray(dmisori[:,i])
    ldict['rho_n1'] = np.ascontiguousarray(ss_rho[0,:,i])
    ldict['rho_n2'] = np.ascontiguousarray(ss_rho[1,:,i])
    ldict['rho_n3'] = np.ascontiguousarray(ss_rho[2,:,i])
    ldict['rho_n4'] = np.ascontiguousarray(ss_rho[3,:,i])
    ddict['grain_elem'] = fvtk.evtk_elem_data_creation(grains_elem, guelem, nelems, 0)
    ldict['misorientation'] = fvtk.evtk_elem_data_creation(misori[:,i], guelem, nelems, 0)
#    ddict['alpha'] = fvtk.evtk_elem_data_creation(alpha[:,i], guelem, nelems, 0)
#    ddict['Grain_misorientation'] = fvtk.evtk_elem_data_creation(gr_misori[:,i], guelem, nelems, 0)
#    lkeys = {}
#    ddict['grain_elem'] = fvtk.evtk_elem_data_creation(grains_elem, guelem, nelems, 0)
#    lkeys['grain_elem'] = 'Scalars' 
    ldict['grain_elem'] = fvtk.evtk_elem_data_creation(grains_elem, guelem, nelems, 0)
#    ddict2['grain_pts'] = fvtk.evtk_pts_data_creation(grains_pts, gupts)
#    ldict2['grain_pts'] = fvtk.evtk_pts_data_creation(grains_pts, gupts)
    xcrd, ycrd, zcrd = fvtk.evtk_xyz_crd_creation(data['coord'][:,:,i], gupts)
    ddict['stress'] = fvtk.evtk_elem_data_creation(data['stress'][:,:,i], guelem, nelems, 0)
    ddict['strain'] = fvtk.evtk_elem_data_creation(data['strain'][:,:,i], guelem, nelems, 0)
    ddict['crss'] = fvtk.evtk_elem_data_creation(data['crss'][:,:,i], guelem, nelems, 0)
#    ddict['dpeff'] = fvtk.evtk_elem_data_creation(data['dpeff'][:,:,i], guelem, nelems, 0)
    ldict['stress'] = fvtk.evtk_elem_data_creation(ldata['stress'][:,:,i], guelem, nelems, 0)
    ldict['strain'] = fvtk.evtk_elem_data_creation(ldata['strain'][:,:,i], guelem, nelems, 0)
    ldict['crss'] = fvtk.evtk_elem_data_creation(ldata['crss'][:,:,i], guelem, nelems, 0)
#   ldict['dpeff'] = fvtk.evtk_elem_data_creation(ldata['dpeff'][:,:,i], guelem, nelems, 0)
 
    ldict['mean_stress'] = fvtk.evtk_elem_data_creation(mStress[:,i], guelem, nelems, 0)
    ldict['deff_stress'] = fvtk.evtk_elem_data_creation(effStress[:,i], guelem, nelems, 0)
    ldict['triaxiality'] = fvtk.evtk_elem_data_creation(triax[:,i], guelem, nelems, 0)
    temp = fvtk.evtk_fileCreation(fVTKLOFEM+str(i), xcrd, ycrd, zcrd, evtk_conn, evtk_offset, evtk_type, cellData=ldict, cellKeys = loDict, ptsData=ldict2, ptsKeys=pDict)
    lofem_file.append(temp)
    print('######## Printed LOFEM File #########')
    temp = fvtk.evtk_fileCreation(fVTKDISC+str(i), xcrd, ycrd, zcrd, evtk_conn, evtk_offset, evtk_type, cellData=ddict, cellKeys = cDict)#, ptsData=ddict2)
    disc_file.append(temp)
    print('######## Printed DISC File #########')

#%%

simTimes = np.r_[0:nsteps]
#
fvtk.evtk_groupVTKData(fVTKLGROUP, lofem_file, simTimes)
fvtk.evtk_groupVTKData(fVTKDGROUP, disc_file, simTimes)

#%%

smstress = np.mean(np.abs(mStress), axis=0)
seffstress = np.mean(np.abs(effStress), axis=0)
striax = np.mean(np.abs(triax), axis=0)

with open(fileLoc+fBname+'mean_mstress.text','wb') as f_handle:
#        f_handle.write(bytes('%Grain number'+str(grnum)+'\n','UTF-8'))
    np.savetxt(f_handle, smstress)
    
with open(fileLoc+fBname+'mean_deffstress.text','wb') as f_handle:
#        f_handle.write(bytes('%Grain number'+str(grnum)+'\n','UTF-8'))
    np.savetxt(f_handle, seffstress)
    
with open(fileLoc+fBname+'mean_triax.text','wb') as f_handle:
#        f_handle.write(bytes('%Grain number'+str(grnum)+'\n','UTF-8'))
    np.savetxt(f_handle, striax)
