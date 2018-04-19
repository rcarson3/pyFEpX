#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:06:47 2018

@author: robertcarson
"""

import numpy as np
#import graph_cc_dfs as gccd
import pickle
from itertools import product

'''
The gb_slip_transferal module will contain all of the necessary functions to find
what grains can allow slip transferal. Next, it will contain all of the necessary
functions that allow us to create our grain boundary element interaction matrix.

'''

def gb_interaction_rss_list(gr_angs, gr_neigh_list, xtal_sn, fLoc):
    '''
    It attempts to find grains that have slip plane normals within 20 degs
    of their neighboring grains. If they don't then we don't have to worry
    about slip transferal between those two grains.
    
    It also saves all of the grain boundary interactions off to a Python pickle
    file. These calculations should not need to change between different mesh
    resolutions or from different loading conditions ideally. 
    
    Input:
        gr_angs - The orientations for each grain represented as rotation matrix.
                    It should be a numpy array of 3x3xngrains
        gr_neigh_list - A list of all of the neighbors for each grain. It should
                        be a list of sets that has ngrain elements in it.
        xtal_sn - The xtal slip plane normals for the specified xtal type.
                  This is currently limited to single phase materials, but it
                  should be extendable to multiple phase materials.
                  It is a numpy array of 3xnslip_normals.
        fLoc - The location for where one wants all of the grain boundary
               interactions saved off to. It should be a valid path string.
    Output:
        grain_inter_rss - A dictionary that contains all of the possible GB
                        interactions. The keys are a tuple of (GA#, GC#) where
                        GA# and GC# are the sorted grain numbers that describe a specific
                        GB. The contents of the dictionary at a specific key are a list
                        with two numpy arrays. The first array contains the permutations
                        of all of the xtal_sn indices. The second array is a numpy
                        boolean array that tells us if slip transferal is even possible there.
    '''
    
    
    grain_inter_rss = dict()
    nsn = xtal_sn.shape[1]
    ngrains = len(gr_neigh_list)
    
    tmp = np.r_[0:nsn]
    
    p=list(product(tmp,repeat=2))
    #This is going to be a constant numpy array that goes into the list of
    #which resides in p.
    arr1 = np.array([p[i:i+nsn] for i in range(0,len(p),nsn)])

    nperms = nsn*nsn    
    arr1 = arr1.reshape((nperms, 2))
    #We are preallocating our boolean array for when it's used in our inner loop
    bool_arr = np.full((nperms), False, dtype=bool)
    
    mindeg = 20 * np.pi/180
    
    
    #Looping through all of the 
    for i in range(ngrains):
        #The current grain that we are on
        pr_grain = i + 1
        for gr in gr_neigh_list[i]:
            #We need our dictionary keys to be placed in order 
            #ganum and gbnum are our keys
            ganum = np.min([gr, pr_grain])
            gbnum = np.max([gr, pr_grain])
            #Figuring out which way we need to do our multiplications for our
            #permutation matrices
            if ganum == pr_grain:
                pr_loc = 0
                ngh_loc = 1
            else:
                pr_loc = 1
                ngh_loc = 0
            
            dict_key = tuple([ganum, gbnum])
            #We only need to go through the calculations steps if this grain
            #boundary interaction has not already been seen
            if dict_key not in grain_inter_rss:
                bool_arr[:] = False
                #Getting the rotated slip plane normals
                pr_gr_sn = np.squeeze(gr_angs[:,:,i]).dot(xtal_sn)
                #Python is a little dumb so this is now and we can't just
                #subtract the number inside the index.
                ngh_gr_sn = np.squeeze(gr_angs[:,:,(gr - 1)]).dot(xtal_sn)
                for j in range(nperms):
                    pr_perm = arr1[j, pr_loc]
                    ngh_perm = arr1[j, ngh_loc]
                    #Calculating the dot product and angle
                    dp = np.squeeze(pr_gr_sn[:,pr_perm]).dot(np.squeeze(ngh_gr_sn[:, ngh_perm]))
                    if np.abs(dp) > 1.0:
                        dp = 0.99999 * np.sign(dp)
                    ang = np.arccos(dp)
                    #Checking to see if our degree is below the minimum 20 degs
                    if ang <= mindeg:
                        bool_arr[j] = True
                
                #We are now assigning our data to our dictionary with the provided key
                #The copy is required due to numpy wanting to just have shallow copies everywhere
                grain_inter_rss[dict_key] = [arr1.copy(), bool_arr.copy()]
                        
                
    #We now pickle all of our grain interactions so we don't need to constantly
    #recalculate this on subsequent simulations 
    #Later on we can just reread all of this in by doing something like
    #with open(fileLoc + 'gb_inter_rss_dict.pickle', 'rb') as f_handle:
    #    grain_neigh_list2 = pickle.load(f_handle)       
    with open(fLoc + 'grain_inter_rss_dict.pickle', 'wb') as f_handle:
        pickle.dump(grain_inter_rss, f_handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return grain_inter_rss

def gb_inter_rss_selection(gr_angs, gr_inter_list, grains, gb_elem_set, xtal_ss, stress, xtal_type, step, fLoc):
    '''
    It goes through all of the allowable grain boundary interactions in order
    to find the slip system that has the largest allowable resolved shear
    stress. The final structure is saved off for future post processing incase
    one wants to try and look at various different grain 
    
    Input:
        gr_angs - The orientations elemental orientations for all elements. It should
                  be a numpy array of 3x3xnelems.
                  
        gr_inter_list - A dictionary that contains all of the possible GB
                        interactions. The keys are a tuple of (GA#, GC#) where
                        GA# and GC# are the sorted grain numbers that describe a specific
                        GB. The contents of the dictionary at a specific key are a list
                        with two numpy arrays. The first array contains the permutations
                        of all of the xtal_sn indices. The second array is a numpy
                        boolean array that tells us if slip transferal is even possible there.
        grains - The grain number that each element corresponds to. It is represented as 
                 a 1D numpy int array.
        gb_elem_set - A set of frozen sets that contains all of grain boundary
                      element pairs. 
        xtal_ss - The xtal slip systems schmid tensors for the specified xtal type.
                  This is currently limited to single phase materials, but it
                  should be extendable to multiple phase materials.
                  It is a numpy array of 3x3xnslip_systems.
        stress - The Caucgy stress for every element. It should be a numpy array
                 with dimensions 3x3xnelems.
        xtal_type - It tells us what crystal type we are dealing with to allow
                    for easier post processing. Once again this is currently single
                    phase. However, it could be easily extended to multiple
                    phases. The possible values are "FCC", "BCC", or "HCP"
        step - The load step you're on. It's used in pickle file for indexing purposes.
        fLoc - The location for where one wants all of the grain element
               interactions saved off to. It should be a valid path string.
    Output:
        gb_inter_rss -  A similar structure to gr_inter_list. 
                        A dictionary that contains all of the possible GB element
                        interactions. The keys are a tuple of (GB_e1, GB_e2) where
                        GB_e1 and GB_e2 are the sorted grain elements. 
                        The contents of the dictionary at a specific key are a list
                        with two numpy arrays. The first array contains the permutations
                        of all of the xtal_sn indices. It also contains which slip systems have the
                        highest resolved shear stress for that slip normal. The order goes perms and then
                        corresponds ss num for GB_e1 and GB_e2 respectively. The second array is a numpy
                        boolean array that tells us if slip transferal is even possible there.
                        This dictionary will have to be recreated at each simulation step due to there
                        being new stress values. The nice thing it also will tell us what the 
                        structure of our global connected component list will look like.
    '''
    #A list of lists that tells us for each slip normal what slip systems to examine    
    if xtal_type == 'FCC':
        ss_list = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        nsn = 4
    elif xtal_type == 'BCC':
        ss_list = [[0, 9], [1, 7], [2, 5], [3, 6], [4, 10], [8, 11]]
        nsn = 6
    elif xtal_type == 'HCP':
        ss_list = [[0, 1, 2], [3], [4], [5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17]]
        nsn = 10
    else:
        raise ValueError('Provided xtal type is not a valid type. FCC, BCC, and HCP are only accepted values')
    
    gb_inter_rss = dict()
    
    tmp = np.r_[0:nsn]
    
    p=list(product(tmp,repeat=2))
    #This is going to be a constant numpy array that goes into the list of
    #which resides in p.
    arr1 = np.array([p[i:i+nsn] for i in range(0,len(p),nsn)])
    nperms = nsn*nsn    
    arr1 = arr1.reshape((nperms, 2))
    
    nind = np.r_[0:nperms]
    #This is going to be the first item going into our list
    #It contains all of the permuations of slip normals and the rss with the highest
    #value only for those systems with a true value. If isn't true its value is
    #zero.
    arr3 = np.zeros((nperms, 4), dtype='int32', order='F')
    arr3[:,0:2] = arr1.copy() 
    
    #We are going to loop through all of the elements of the gb_elem_set
    for felems in gb_elem_set:
        arr3[:,2:4] = 0
        #We need to convert from a frozenset to a list
        elems = list(felems)
        #Now find the min and max values we will use this as our dict_key
        min_elem = min(elems)
        max_elem = max(elems)
        dict_key = tuple([min_elem, max_elem])
        #We want to find the grains associatted with the min and max elements
        min_gr = grains[min_elem]
        max_gr = grains[max_elem]
        #Now we want to sort our min_gr and max_gr for our dict_key
        in_dict_key = tuple(sorted([min_gr, max_gr]))
        bool_arr = gr_inter_list[in_dict_key][1]
        #The normals of interest if any
        if np.any(bool_arr):
            #Go ahead and retrieve our stress and oritation values
            min_rot = np.squeeze(gr_angs[:,:,min_elem])
            max_rot = np.squeeze(gr_angs[:,:,max_elem])
            min_stress = np.squeeze(stress[:,:,min_elem])
            max_stress = np.squeeze(stress[:,:,max_elem])
            #Get the indices of all of those of interest
            ind = nind[bool_arr]
            for i in ind:
                #Get the necessary permutation we're looking at currently
                perm = arr1[i, :]
                #See if the smallest grain number is the minimum angle
                #We then assign the correct permutation number to it
                if in_dict_key[0] == min_gr:
                    min_ss = perm[0]
                    max_ss = perm[1]
                else:
                    min_ss = perm[1]
                    max_ss = perm[0]
                #Find the index corresponding to the minimum resolved shear stress  
                min_rss_ind = -1
                max_rss = 0
                for ss in ss_list[min_ss]:
                    #Rotating from crystal to sample frame for the xtal schmid tensor
                    rxtal_ss = min_rot.dot(np.squeeze(xtal_ss[:,:,ss]).dot(min_rot.T))
                    #We now are finding the resolved shear stress on the system.
                    #We want the absolute maximum value.
                    rss = np.abs(np.trace(min_stress.dot(rxtal_ss.T)))
                    if rss > max_rss:
                        max_rss = rss
                        min_rss_ind = ss
                #Find the index corresponding to the maximum resolved shear stress
                max_rss_ind = -1
                max_rss = 0
                for ss in ss_list[max_ss]:
                    #Rotating from crystal to sample frame for the xtal schmid tensor
                    rxtal_ss = max_rot.dot(np.squeeze(xtal_ss[:,:,ss]).dot(max_rot.T))
                    #We now are finding the resolved shear stress on the system.
                    #We want the absolute maximum value.
                    rss = np.abs(np.trace(max_stress.dot(rxtal_ss.T)))
                    if rss > max_rss:
                        max_rss = rss
                        max_rss_ind = ss   
                #Now save off the min and max rss ind for that permutation
                arr3[i, 2] = min_rss_ind
                arr3[i, 3] = max_rss_ind
             
        gb_inter_rss[dict_key] = [arr3.copy(), bool_arr.copy()]
        
        
    
    #We are now going to pickle all of this data to be used later on possibly.
    with open(fLoc + 'gb_inter_rss_s'+str(step)+'dict.pickle', 'wb') as f_handle:
        pickle.dump(gb_inter_rss, f_handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return gb_inter_rss
    
    