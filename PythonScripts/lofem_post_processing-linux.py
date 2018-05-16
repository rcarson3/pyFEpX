#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:20:49 2017

@author: robertcarson
"""

import numpy as np
import FePX_Data_and_Mesh as fepxDM
import FiniteElement as fe
#from latorifem import mainlatprogram as latfem
import Rotations as rot
import Misori as mis
#%%
#Getting the location of all of our simulation data and then the mesh file name
#fileLoc = '/Users/robertcarson/Research_Local_Code/Output/LOFEM_STUDY/n456_cent/low/'
#fileLoc = '/media/robert/My Passport for Mac/Simulations/LOFEM_Study/n456_cent_m15/mid_txt/'
fileLoc = '/home/rac428/Outputs/LOFEM_Study/n456_cent_uori_m15/low_txt/'
#fileLoc = '/media/robert/DataDrives/LOFEM_Study/n456_NF/mono/low_txt/'
#fileLoc = '/Users/robertcarson/Research_Local_Code/fepx_robert/Examples/ControlMode/LOFEM_REFACTOR2/data/'
fileName = 'n456-cent-rcl05'
#fileName = 'n456_nf_raster_L2_r1_v2_rcl075'
#fileName = 'n6'
#What we want the basename of the file where we save our kinematic metrics saved along with a few other variables.
fBname = 'grainData'

#fileLoc = '/media/robert/DataDrives/n1k_pois_iso_reg_pt2/'
#fileName = 'n1k-id6k-rcl05'

#The number of processors and steps within the simulation.
nproc = 64
#nsteps = 16
nsteps = 46
#nsteps = 19
#nsteps = 43
#nsteps = 44
#nsteps = 52
#nsteps = 64
#nsteps = 86

frames = np.arange(0,nsteps)
#Reading in our mesh data
mesh = fepxDM.readMesh(fileLoc, fileName, LOFEM = True)
#How many grains that our polycrystal had
#ngrains = 6
ngrains = 456
#ngrains = 1000

grains = np.r_[1:(ngrains+1)]
#Misorientation difference variable that shows the relative angle of rotation between the discrete and smooth lattice methods
#from element to element
misoriD = np.zeros((mesh['grains'].shape[0], nsteps))

#%%

print('About to start processing data')
#Tells us what our angle file data is whether its a rod vec or kocks angles
kor = 'rod'
#Reading in our LOFEM data
ldata = fepxDM.readLOFEMData(fileLoc, nproc, lofemData=['strain', 'ang'])
print('Finished Reading LOFEM data')
print('Starting to read DISC data')
data = fepxDM.readData(fileLoc, nproc, fepxData=['ang', 'adx', 'strain'], restart=False)
print('Finished Reading DISC data')

#%%
#Global connectivity array reordered such that it goes grain by grain
gconn = np.asarray([], dtype='float64')
gconn = np.atleast_2d(gconn)
#The unique pts and elements that correspond to the above
gupts = np.asarray([], dtype=np.int32)
guelem = np.asarray([], dtype=np.int32)
#Finding the nodal points and elements upper and lowere bounds for all of the grain data
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
    
    gconn, gupts, guelem = fe.concatConnArray(gconn, lcon, gupts, lupts, guelem, luelem) 

npts = gupts.shape[0]
nelem = guelem.shape[0]

#%%
#The below is the same as the above but here we just use the LOFEM connectivity array
gconn2 = np.asarray([], dtype='float64')
gconn2 = np.atleast_2d(gconn2)
gupts2 = np.asarray([], dtype=np.int32)
guelem2 = np.asarray([], dtype=np.int32)

se_bnds2 = np.zeros((ngrains*2), dtype='int32')
se_el_bnds2 = np.zeros((ngrains*2), dtype='int32')

st_bnd = 0
en_bnd = 0

st_bnd2 = 0
en_bnd2 = 0

for i in grains:
    
    lcon, lupts, luelem = fe.localGrainConnectCrd(mesh, i)
    st_bnd = en_bnd
    en_bnd = st_bnd + lupts.shape[0]
    
    j = (i - 1) * 2
    
    se_bnds2[j] = st_bnd
    se_bnds2[j+1] = en_bnd
    
    st_bnd2 = en_bnd2
    en_bnd2 = st_bnd2 + luelem.shape[0]
    
    j = (i - 1) * 2
    
    se_el_bnds2[j] = st_bnd2
    se_el_bnds2[j+1] = en_bnd2
    
    gconn2, gupts2, guelem2 = fe.concatConnArray(gconn2, lcon, gupts2, lupts, guelem2, luelem) 

npts2 = gupts2.shape[0]
nelem2 = guelem2.shape[0]

#%%
#
#These are variables telling us the relative rotation away from the current grain average orientation for either
#nodal or elemental data  
gr_angs = np.zeros((1, npts,  nsteps), dtype='float64')
lofem_angs = np.zeros((1, nelem,  nsteps), dtype='float64')
disc_angs = np.zeros((1, nelem,  nsteps), dtype='float64')
#Telling us the origin in 3D space
origin = np.zeros((3,1), dtype='float64')
#%%
#
for i in grains:
    print('###### Starting Grain Number '+str(i)+' ######')
    
    #Reading in our local connectivity arrays in terms of our regular connectivity array and the one generated for the LOFEM simulations
    lcon, lcrd, ucon, uelem = fe.localConnectCrd(mesh, i)
    lcon2, ucon2, uelem2 = fe.localGrainConnectCrd(mesh, i)
    # # of elements and nodes in a grain
    nel = lcon.shape[1]
    npts = ucon.shape[0]
    #Tells us globally what points correspond to the grain we're examing
    indlog = mesh['grains'] == i
    indlog2 = mesh['crd_grains'] == i
    #Here we're getting the misorientation angle and quaternion for our angles when taken with respect the original orientation
    #for the discrete method
    misAngs, misQuats = mis.misorientationGrain(mesh['kocks'][:,i-1], data['angs'][:,indlog,:], frames, 'kocks')
    #Legacy code but just setting our deformation gradient to the identity array
    defgrad = np.tile(np.atleast_3d(np.identity(3)), (1,1,nel))
    #A list holding our deformation stats for the discrete and lofem methods
    deflist = []
    ldeflist = []
    #el_angs is a temporary variable that will hold the grain values that go into misoriD
    el_angs = np.zeros((3,nel,nsteps))
    #Our difference quats, lofem quaternion at nodes, lofem quaternion at the centroid of the element, and discrete method quats
    diff_misQuats = np.zeros((4,nel,nsteps))
    lQuats = np.zeros((4, npts, nsteps))
    leQuats = np.zeros((4, nel, nsteps))
    dQuats = np.zeros((4, nel, nsteps))
    #Just converting from our inputted orientation data to quaternions
    for j in range(nsteps):      
        el_angs[:,:,j] = fe.elem_fe_cen_val(ldata['angs'][:,indlog2,j], lcon2)
        lQuats[:,:,j] = rot.QuatOfRod(np.squeeze(ldata['angs'][:,indlog2,j]))
        leQuats[:,:,j] = rot.QuatOfRod(np.squeeze(el_angs[:,:,j]))
        dQuats[:,:,j] = rot.OrientConvert(np.squeeze(data['angs'][:,indlog,j]), 'kocks', 'quat', 'degrees', 'radians')
    #Here we're getting the misorientation angle and quaternion for our angles when taken with respect the original orientation
    #for the lofem method
    lemisAngs, lemisQuats = mis.misorientationGrain(mesh['kocks'][:,i-1], el_angs, frames, kor)
        
    for j in range(nsteps):
        #Getting misorientation between the lofem and disc elements
        temp2, tempQ = mis.misorientationGrain(data['angs'][:,indlog, j], el_angs[:,:,j], [0], kor)
        diff_misQuats[:,:,j] = np.squeeze(tempQ)
        misoriD[indlog, j] = np.squeeze(temp2)
        
        crd = np.squeeze(data['coord'][:,ucon, j])
        #Getting strain data
        epsVec = np.squeeze(ldata['strain'][:, indlog, j])
        #Taking the strain data and putting it into the tensorial view
        #FEpX saves strain data off as 11, 21, 31, 22, 32, 33 so we also have to do some other
        #fanagling of the data
        strain = fepxDM.fixStrain(epsVec)
        #Calculating the volume and wts of the element assumming no curvature to the element
        #The wts are used in all of the calculations and these are relative wts where each element wts is based on
        #vol_elem/vol_grain
        vol, wts = fe.calcVol(crd, lcon)
        #Getting our deformation data but this method is old so we can actually update it a bit
        ldefdata = fe.deformationStats(defgrad, wts, crd, lcon, lemisQuats[:, :, j], el_angs[:,:,j], strain, kor)
        ldeflist.append(ldefdata)
        #Doing the same as the above but now for the discrete data case
        epsVec = np.squeeze(data['strain'][:, indlog, j])
        strain = fepxDM.fixStrain(epsVec)
        
        defdata = fe.deformationStats(defgrad, wts, crd, lcon, misQuats[:, :, j], data['angs'][:, indlog, j], strain, 'kocks')
        deflist.append(defdata)
        
        print('Grain #'+str(i)+'% done:  {:.3f}'.format(((j+1)/nsteps)))
    #Saving off all of the data now
    with open(fileLoc+fBname+'LOFEM'+'.vespread','ab') as f_handle:
        f_handle.write(bytes('%Grain number'+str(i)+'\n','UTF-8'))
        for j in range(nsteps):
            np.savetxt(f_handle,ldeflist[j]['veSpread'])
    
    with open(fileLoc+fBname+'DISC'+'.vespread','ab') as f_handle:
        f_handle.write(bytes('%Grain number'+str(i)+'\n','UTF-8'))
        for j in range(nsteps):
            np.savetxt(f_handle,deflist[j]['veSpread'])
            
    with open(fileLoc+fBname+'LOFEM'+'.fespread','ab') as f_handle:
        f_handle.write(bytes('%Grain number'+str(i)+'\n','UTF-8'))
        for j in range(nsteps):
            np.savetxt(f_handle,ldeflist[j]['feSpread'])
    
    with open(fileLoc+fBname+'DISC'+'.fespread','ab') as f_handle:
        f_handle.write(bytes('%Grain number'+str(i)+'\n','UTF-8'))
        for j in range(nsteps):
            np.savetxt(f_handle,deflist[j]['feSpread'])
    #Calculating all of our misorientation data now
    stats = mis.misorientationTensor(lQuats, lcrd, lcon, data['coord'][:, ucon, :], i, True)
    lmisAngs, lmisQuats = mis.misorientationGrain(origin, stats['angaxis'], frames, 'axis', True)
    
    with open(fileLoc+fBname+'LOFEM'+'.misori','ab') as f_handle:
        f_handle.write(bytes('%Grain number '+str(i)+'\n','UTF-8'))
        np.savetxt(f_handle,stats['gSpread'])

    stats = mis.misorientationTensor(dQuats, lcrd, lcon, data['coord'][:, ucon, :], i, False)
    misAngs, misQuats = mis.misorientationGrain(origin, stats['angaxis'], frames, 'axis', True)
    
    with open(fileLoc+fBname+'DISC'+'.misori','ab') as f_handle:
        f_handle.write(bytes('%Grain number '+str(i)+'\n','UTF-8'))
        np.savetxt(f_handle,stats['gSpread'])

    stats = mis.misorientationTensor(leQuats, lcrd, lcon, data['coord'][:, ucon, :], i, False)
    lemisAngs, lemisQuats = mis.misorientationGrain(origin, stats['angaxis'], frames, 'axis', True)
    with open(fileLoc+fBname+'LOFEM_ELEM'+'.misori','ab') as f_handle:
        f_handle.write(bytes('%Grain number '+str(i)+'\n','UTF-8'))
        np.savetxt(f_handle,stats['gSpread'])

    stats = mis.misorientationTensor(diff_misQuats, lcrd, lcon, data['coord'][:, ucon, :], i, False)
    with open(fileLoc+fBname+'DIFF_LOFEM'+'.misori','ab') as f_handle:
        f_handle.write(bytes('%Grain number '+str(i)+'\n','UTF-8'))
        np.savetxt(f_handle,stats['gSpread'])

    l = (i - 1) * 2
    k = l + 2
    
    ind = se_bnds[l:k]
    ind2 = se_el_bnds[l:k]
    #Saving off the relative misori data mentioned earlier
    gr_angs[:, ind[0]:ind[1], :] = lmisAngs
    disc_angs[:, ind2[0]:ind2[1], :] = misAngs
    lofem_angs[:, ind2[0]:ind2[1], :] = lemisAngs
        
#%%
#Writing those misori data off to a file
with open(fileLoc+fBname+'diff'+'.emisori','ab') as f_handle:
    for i in range(nsteps):
        f_handle.write(bytes('%Step number '+str(i)+'\n','UTF-8'))
        np.savetxt(f_handle, np.squeeze(misoriD[:, i]))


#%%
with open(fileLoc+fBname+'.cmisori','ab') as f_handle:
    for i in range(nsteps):
        f_handle.write(bytes('%Step number '+str(i)+'\n','UTF-8'))
        np.savetxt(f_handle, np.squeeze(gr_angs[:, :, i]))
        
#%%
with open(fileLoc+fBname+'_DISC'+'.cmisori','ab') as f_handle:
    for i in range(nsteps):
        f_handle.write(bytes('%Step number '+str(i)+'\n','UTF-8'))
        np.savetxt(f_handle, np.squeeze(disc_angs[:, :, i]))
#%%
with open(fileLoc+fBname+'_LOFEM_ELEM'+'.cmisori','ab') as f_handle:
    for i in range(nsteps):
        f_handle.write(bytes('%Step number '+str(i)+'\n','UTF-8'))
        np.savetxt(f_handle, np.squeeze(lofem_angs[:, :, i]))
