#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:43:32 2017

@author: robertcarson
"""

import numpy as np
from scipy import optimize as sciop

def sfmat():
    '''
    Outputs the shape function matrix for a 10 node tetrahedral element
    '''
    NDIM = 3
    
    qp3d_ptr = np.zeros((NDIM*15))
    
    qp3d_ptr[0] =  0.333333333333333333e0
    qp3d_ptr[1 * NDIM] =  0.333333333333333333e0
    qp3d_ptr[2 * NDIM] =  0.333333333333333333e0
    qp3d_ptr[3 * NDIM] =  0.0e0
    qp3d_ptr[4 * NDIM] =  0.25e0
    qp3d_ptr[5 * NDIM] =  0.909090909090909091e-1
    qp3d_ptr[6 * NDIM] =  0.909090909090909091e-1
    qp3d_ptr[7 * NDIM] =  0.909090909090909091e-1
    qp3d_ptr[8 * NDIM] =  0.727272727272727273e0
    qp3d_ptr[9 * NDIM] =  0.665501535736642813e-1
    qp3d_ptr[10 * NDIM] = 0.665501535736642813e-1
    qp3d_ptr[11 * NDIM] = 0.665501535736642813e-1
    qp3d_ptr[12 * NDIM] = 0.433449846426335728e0
    qp3d_ptr[13 * NDIM] = 0.433449846426335728e0
    qp3d_ptr[14 * NDIM] = 0.433449846426335728e0
    
    qp3d_ptr[1] =  0.333333333333333333e0
    qp3d_ptr[1 + 1 * NDIM] =  0.333333333333333333e0
    qp3d_ptr[1 + 2 * NDIM] =  0.0e0
    qp3d_ptr[1 + 3 * NDIM] =  0.333333333333333333e0
    qp3d_ptr[1 + 4 * NDIM] =  0.25e0
    qp3d_ptr[1 + 5 * NDIM] =  0.909090909090909091e-1
    qp3d_ptr[1 + 6 * NDIM] =  0.909090909090909091e-1
    qp3d_ptr[1 + 7 * NDIM] =  0.727272727272727273e0
    qp3d_ptr[1 + 8 * NDIM] =  0.909090909090909091e-1
    qp3d_ptr[1 + 9 * NDIM] =  0.665501535736642813e-1
    qp3d_ptr[1 + 10 * NDIM] = 0.433449846426335728e0
    qp3d_ptr[1 + 11 * NDIM] = 0.433449846426335728e0
    qp3d_ptr[1 + 12 * NDIM] = 0.665501535736642813e-1
    qp3d_ptr[1 + 13 * NDIM] = 0.665501535736642813e-1
    qp3d_ptr[1 + 14 * NDIM] = 0.433449846426335728e0
    
    qp3d_ptr[2] =  0.333333333333333333e0
    qp3d_ptr[2 + 1 * NDIM] =  0.0e0
    qp3d_ptr[2 + 2 * NDIM] =  0.333333333333333333e0
    qp3d_ptr[2 + 3 * NDIM] =  0.333333333333333333e0
    qp3d_ptr[2 + 4 * NDIM] =  0.25e0
    qp3d_ptr[2 + 5 * NDIM] =  0.909090909090909091e-1
    qp3d_ptr[2 + 6 * NDIM] =  0.727272727272727273e0
    qp3d_ptr[2 + 7 * NDIM] =  0.909090909090909091e-1
    qp3d_ptr[2 + 8 * NDIM] =  0.909090909090909091e-1
    qp3d_ptr[2 + 9 * NDIM] =  0.433449846426335728e0
    qp3d_ptr[2 + 10 * NDIM] = 0.665501535736642813e-1
    qp3d_ptr[2 + 11 * NDIM] = 0.433449846426335728e0
    qp3d_ptr[2 + 12 * NDIM] = 0.665501535736642813e-1
    qp3d_ptr[2 + 13 * NDIM] = 0.433449846426335728e0
    qp3d_ptr[2 + 14 * NDIM] = 0.665501535736642813e-1
            
    sfvec_ptr = np.zeros((10))
    N = np.zeros((15,10))
            
    for i in range(15):
        loc_ptr = qp3d_ptr[i*3:(i+1)*3]
        sfvec_ptr[0] = 2.0e0 * (loc_ptr[0] + loc_ptr[1] + loc_ptr[2] - 1.0e0) * (loc_ptr[0] + loc_ptr[1] +loc_ptr[2] - 0.5e0)
        sfvec_ptr[1] = -4.0e0 * (loc_ptr[0] + loc_ptr[1] + loc_ptr[2] - 1.0e0) * loc_ptr[0]
        sfvec_ptr[2] = 2.0e0 * loc_ptr[0] * (loc_ptr[0] - 0.5e0)
        sfvec_ptr[3] = 4.0e0 * loc_ptr[1] * loc_ptr[0]
        sfvec_ptr[4] = 2.0e0 * loc_ptr[1] * (loc_ptr[1] - 0.5e0)
        sfvec_ptr[5] = -4.0e0 * (loc_ptr[0] + loc_ptr[1] + loc_ptr[2] - 1.0e0) * loc_ptr[1]
        sfvec_ptr[6] = -4.0e0 * (loc_ptr[0] + loc_ptr[1] + loc_ptr[2] - 1.0e0) * loc_ptr[2]
        sfvec_ptr[7] = 4.0e0 * loc_ptr[0] * loc_ptr[2]
        sfvec_ptr[8] = 4.0e0 * loc_ptr[1] * loc_ptr[2]
        sfvec_ptr[9] = 2.0e0 * loc_ptr[2] * (loc_ptr[2] - 0.5e0)
        N[i, :] = sfvec_ptr[:]
    
    return N

def lofem_nnlstq(conn, abs_gamma, nsf, ncrds):
    '''
        Inputs:
                conn - the local connectivity array a nelem x 10 size array
                abs_gamma - the absolute gamma values at each quad point
                        size = nelem x nqpts x nslip
                nsf - the shape function matrix
                ncrds - number of coordinates/nodal points in the grain
        Output:
                nod_agamma - the nodal values of a grain for the abs_gamma
                residual - the residual from the least squares
                
        A nonlinear least squares optimization routine is used to solve for
        the solution. It'll find the nodal values of the absolute gamma for
        a grain.
    '''
    
    nelems = conn.shape[1]
    nslip = abs_gamma.shape[1]
    amat = np.zeros((nelems*15, ncrds))
    #Build up our A matrix to be used in the nonnegative lstsqs solution
    j = 0
    k = 0
    for i in range(nelems):
        j = i * 15
        k = (i + 1) * 15
        ecrds = np.squeeze(conn[:, i])
        amat[j:k,  ecrds] = nsf
    
    
    nod_agamma = np.zeros((nslip, ncrds))
    residual = np.zeros((nslip))
    
    for i in range(nslip):
        agamma = np.ravel(abs_gamma[:, i, :], order = 'F')
        nod_agamma[i, :], residual[i] = sciop.nnls(amat, agamma)
        
    return (nod_agamma, residual)
        
        
        
        
        
        
        
    