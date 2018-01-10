#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:59:18 2018

@author: robertcarson
"""

import numpy as np
import FePX_Data_and_Mesh as fepxDM

#Where the mesh, grain, and phase info is located

fileLoc = '/Users/robertcarson/OneDrive/LOFEM_Study/n456_cent/mid/'
fileName = 'n456-cent-rcl04'
#fileName = 'n456_nf_raster_L2_r1_v2_rcl05'
#Mesh is read
mesh = fepxDM.readMesh(fileLoc,fileName)
#Getting the initial mesh
conn = mesh['con']
#Finding out how many crds, nodes, and nelems there are
ncrds = np.unique(np.ravel(conn)).size
ncvec = ncrds * 3
nelems = conn.shape[1]
nnode = conn.shape[0]
#Getting the grain and phases for each element
#The phase information should be changed later onto to be either
#1 == FCC, 2 == BCC, and 3 == HCP
grains = mesh['grains']
phases = mesh['phases']
#We now get the nodal connectivity array
ndconn = fepxDM.mesh_node_conn(conn, ncrds)
#We now fix the connectivity such that all of the grain boundary nodes have
#a unique index for every grain that it is shared between.
#If we didn't do this all of the grains would be connected. We don't want this.
#We want each grain to be its own mesh for the LOFEM method. 
conn2 = fepxDM.grain_conn_mesh(ndconn, conn, grains, ncrds)
#We now find the new number of coords
ncrds = np.unique(np.ravel(conn2)).size
#We find the final nodal connectivity array which will be used for the
#coord phase and grain info
ndconn2 = fepxDM.mesh_node_conn(conn2, ncrds)
#We offset this by one just for LOFEM method since it assumes a 1 base for
#its connectivity arrays.
conn2 = conn2 + 1
#Initiallizing the grain and phase arrays
grains2 = np.zeros((ncrds,1), dtype='int32') - 1
phase2 = np.zeros((ncrds,1), dtype='int32') - 1
#We now loop through all of the nodes
for i in range(len(ndconn2)):
    #We find all of the elements that a node is connected to
    ind = np.array(list(ndconn2[i]), dtype='int32')
    #All elements that a node are connected to should have the same
    #grain and phase info, so we just set that nodes grain and phase
    #to the same value as the first elements value
    grains2[i,0] = grains[ind[0]]
    phase2[i,0] = phases[ind[0]]

#Finally we can write out our new connectivity array for the global
#LOFEM mesh. The first thing for that files requirements is the number of
#coords in the entire system.
with open(fileLoc+fileName+'2.cmesh','wb') as f_handle:
    f_handle.write(bytes(str(ncrds)+'\n','UTF-8'))
    np.savetxt(f_handle, np.squeeze(conn2).T, fmt='%d')
#We now write out the coord grain file that follows the same format as the
#traditional one from FEpX. The one difference is we don't write out how
#many grains or phases there are since this is just repeated information
#from the elemental grain file.    
tmp = np.concatenate((grains2, phase2), axis=1)    
with open(fileLoc+fileName+'2.cgrain','wb') as f_handle:
    np.savetxt(f_handle, np.squeeze(tmp), fmt='%d')
    
