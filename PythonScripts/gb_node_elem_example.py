#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:54:56 2018

@author: robertcarson
"""

import numpy as np
import FePX_Data_and_Mesh as fepxDM
import FiniteElement as fe

#The location of our mesh/data
fileLoc = '/Users/robertcarson/Research_Local_Code/fepx_robert/Source/LOFEM/'
#The name of our mesh
fileName = 'n6'

mesh = fepxDM.readMesh(fileLoc,fileName)
#How many grains we have
ngrains = 6

mesh = fepxDM.readMesh(fileLoc,fileName)

conn = mesh['con']

ncrds = np.unique(np.ravel(conn)).size
nelems = conn.shape[1]

grains = mesh['grains']
#The list of all of our grains
ugrains = np.unique(grains)
#Here we're creating our nodal connectivity array
ndconn = fepxDM.mesh_node_conn(conn, ncrds)
#This gets all of the grain boundary nodes
gbnodes, nincr = fepxDM.grain_boundary_nodes(ndconn, grains, ncrds)

grain_gbnode_list = list()
grain_neigh_list = list()
#%%
for igrain in ugrains:
    print('###### Starting Grain Number '+str(igrain)+' ######')
    tmp = set()
    tmp2 = set()
    for inode in gbnodes:
        if igrain in gbnodes[inode]:
            tmp.add(inode)
            tmp2.update(list(gbnodes[inode].keys()))
    #Once we've iterated through all of the grain boundary nodes we append
    #our temporary set to the grain gb node list 
    tmp2.remove(igrain)          
    grain_gbnode_list.append(tmp)
    grain_neigh_list.append(tmp2)
   
for i in ugrains:
    print('###### Starting Grain Number '+str(i)+' ######')
    #Create a set that will hold all of the elements that have a GB node
    grain_set = set()
    #We might be doing work where we need the local connectivity and etc.
    #If we don't we could create an index array and from there use a logical
    #array to get the unique elements belonging to the array
    #so something along the lines of:
    #   ind = np.r_[0:nelems]
    #   uelem = ind[mesh['grains'] == i]
    lcon, lcrd, ucon, uelem = fe.localConnectCrd(mesh, i)
    #We're going to need to perform the intersection between two nodal
    #connectivity set and the unique element set for the grain
    #This will allow us a quick way to obtain the elements that are on the
    #grain boundary
    uelem_set = set(uelem.tolist())
    
    for inode in grain_gbnode_list[i-1]:
        #The intersection of our two sets
        tmp = ndconn[inode].intersection(uelem_set)
        #Adding the values from this set to our grain set
        #We used a set here because we don't want any duplicate values
        grain_set.update(tmp)
    #Append our set for this grain to our list
    grain_gbelem_list.append(grain_set)
#Now that we have all of our grain boundary elements and nodes we split up
#we can do what ever analysis we need to on our data located there.        