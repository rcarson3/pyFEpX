import numpy as np
import scipy as sci
import Utility as utl
import Sphere as sph
import Rotations as rot
import FiniteElement as fe
from sklearn.preprocessing import normalize

'''
This program was written by Robert Carson on June 10th, 2015.
It is based upon the OdfPf library that the Deformation Processing Lab has written in MATLAB.

The following functions are available in this module:
misorientationStats
misorientationGrain
misorientationBartonTensor


'''

def bartonStats(misorient, locations, *wts):
    '''
    MisorientationStats - Misorientation correlation statistics.

      USAGE:

      stats = MisorientationStats(misorient, locations)
      stats = MisorientationStats(misorient, locations, wts)

      INPUT:

      misorient is 4 x n, 
                a list of misorientation quaternions,
                assumed to have been derived from properly clustered 
                orientation data
      locations is d x n, (d <= 3) 
                a list of spatial locations corresponding to the 
                misorientations
      wts       is 1 x n, (optional)
                a list of weights; if not specified, uniform weights are used

      OUTPUT:

      stats is a structure with five components:

            W     is a 3 x 3 matrix (A in Barton paper)
            X     is a d x d matrix (M in Barton paper)
            WX    is a 3 x d matrix (cross-correlation
                       of normalized variables; X in
                       Barton paper)
            wi    is 3 x n, the unnormalized axial vectors
            xi    is d x n, the unnormalized spatial directions
                            from the centroid

      REFERENCE:  

      "A Methodology for Determining Average Lattice Orientation and 
      Its Application to the  Characterization of Grain Substructure",

      Nathan R. Barton and Paul R. Dawson,

      Metallurgical and Materials Transactions A,
      Volume 32A, August 2001, pp. 1967--1975
  

    '''
    locations = utl.mat2d_row_order(locations)
    misorient = utl.mat2d_row_order(misorient)
    d, n = misorient.shape  
    if len(wts) == 0:
        wts = np.tile(1.0/n, (3, n))
    else:
        wts = np.tile(wts, (3, 1))
        
    wts1 = np.tile(wts[0, :], (4, 1))
    misOriCen = sph.SphereAverage(misorient, **{'wts':wts1})
    misorient = misorient - np.tile(misOriCen, (1, n))
    
    limit = (ang < np.finfo(float).eps)
    nlimit = (ang > np.finfo(float).eps)
    
    angn = ang[nlimit]
    
    wsc[nlimit]= angn/np.sin(angn/2)
    wsc[limit] = 2
    
    wi = misorient[1:4, :]*np.tile(wsc.T, (3, 1))
    
    wi = wi*np.tile(ang.T, (3,1))

    
    cen = utl.mat2d_row_order(np.sum(locations*wts, axis=1))
    
    xi = locations - np.tile(cen, (1, n))
    
    Winv = np.sum(utl.RankOneMatrix(wi*wts, wi), axis=2)
    Xinv = np.sum(utl.RankOneMatrix(xi*wts, xi), axis=2)
    #We needed to scale this up if it was close to being ill-conditioned
    if(np.abs(np.linalg.det(Winv)) < 1e-6):
        if(np.abs(np.linalg.det(Winv)) < 1e-16):
            W = np.zeros((3,3))
        else:
            Wtemp = np.multiply(1e9, Winv)
            W = np.multiply(1e9, np.linalg.inv(Wtemp))
    else:
        W = np.linalg.inv(Winv)
    Whalf = sci.linalg.sqrtm(W)
    
    if(np.abs(np.linalg.det(Xinv)) < 1e-6):
        Xtemp = np.multiply(1e9, Xinv)
        X = np.multiply(1e9, np.linalg.inv(Xtemp))
    else:
        X = np.linalg.inv(Xinv)
    Xhalf = sci.linalg.sqrtm(X)
    
    wibar = np.dot(Whalf, wi)
    xibar = np.dot(Xhalf, xi)
    
    WX = np.sum(utl.RankOneMatrix(wibar*wts, xibar), axis=2)
    
    stat = {'W':W, 'Winv':Winv, 'Xinv':Xinv, 'X':X, 'WX':WX, 'wi':wi, 'xi':xi}
    
    return stat
    
    
def misorientationGrain(kocks, angs, frames, kor, gr_mis=False):
    '''
    It takes in the mesh, the grain number of interest, the angles output from
    FePX and then the number of frames.
    
    It outputs the misorientations angles calculated for that specific grain
    by using the built in misorientation function within the ODFPF library.
    
    Input: kocks - the kocks angle of the the grain being examined
           grNum - an integer of the grain that you want to find the 
                   misorientation for each element
           angs - a numpy array 3xnxnframes of the angles output from FePX
           frames - list of frames your interested in examining
           
   Output: misAngs - a numpy array of nxframes that contains the angle of
                     misorientation for each element with respect to the
                     original orientation of the grain
           misQuat - a numpy array of 4xnxnframes that contains the
                     misorientation quaternion for each element with respect to
                     the original orientation of the grain
    '''
    angs = np.atleast_3d(angs)
    if angs.shape[0] == 1:
        angs = angs.T
    lenQuat = angs.shape[1]
    deg = 'degrees'
    misAngs = np.zeros((lenQuat,len(frames)))
    misQuat = np.zeros((4,lenQuat, len(frames)))
    misQuat[0, :, :] = 1
    if(gr_mis):
        origQuat = rot.OrientConvert(kocks, 'rod', 'quat', deg, deg)
    else:
        origQuat = rot.OrientConvert(kocks, 'kocks', 'quat', deg, deg)
    csym = rot.CubSymmetries()
    j = 0
    for i in frames:
        if kor == 'axis' or kor == 'axisangle':
            tQuat = rot.QuatOfAngleAxis(np.squeeze(angs[0, :, i]), np.squeeze(angs[1:4, :, i]))    
        else:
            tQuat = rot.OrientConvert(np.squeeze(angs[:, :, i]), kor, 'quat', deg, deg)
        misAngs[:, j], misQuat[:, :, j] = rot.Misorientation(origQuat, tQuat, csym)
        j +=1
        
    return (misAngs, misQuat)
    

def misorientationBartonTensor(misori, lcrd, lcon, crd , grnum, crdOpt = False):
    '''
        misOrientationBartonTensor takes in the misorientation quaternions
    for the grain of interest and finds the various quantities located in
    Barton's paper "A Methodology for Determining Average Lattice Orientation and 
  Its Application to the  Characterization of Grain Substructure." 

    Inputs:
              misOri: the misorientation quaternions for all of the
                  frames of interest as calculated for the grain of interest.
              lcrd: the local crd of the grain mesh
              lcon: the local con of the grain mesh
              crd: the coordinates of each node at each time step
              crdOpt: if crdOpt is to be used

              grNum: The grain number of interest
      Outputs:
              stats: A structure with 14 different elements:
                     W     is a 3 x 3 x nframe matrix (A in Barton paper)
                     X     is a d x d x nframe matrix (M in Barton paper)
                     !WX    is a 3 x d x nframe matrix (cross-correlation
                           of normalized variables; X in
                           Barton paper)
                     wi    is 3 x n x nframe, the unnormalized axial vectors
                     xi    is d x n x nframe, the unnormalized spatial directions
                           from the centroid
                     Winv  is a 3 x 3 x nframe matrix (A inv in Barton Paper)
                     Xinv  is a 3 x 3 x nframe matrix (M inv in Barton Paper)
                     U     is a 3 x 3 x nframe matrix (U in Barton Paper)
                           Also the orientation eigenvectors of WX
                     V     is a 3 x 3 x nframe matrix (V in Barton Paper)
                           Also the spatial eigenvectors of WX
                     S     is a 3 x 3 x nframe matrix (S in Barton Paper)
                           Also the eigenvalues of WX
                     !xV    is 3 x n x nframe, a spatial correlation used in
                           Barton's titanium paper and is a scalar value
                     !wU    is 3 x n x nframe, a misorientation
                           correlation used in Barton's titanium paper and is a
                           scalar value
                GrainVol   is nframe, the volume of the grain during each
                           frame
                gSpread    is nframe, the grain spread as calculated from
                           Barton's paper using Winv. It is a scalar spread of
                           the misorientation blob.
               ! - vars are currently commented out inorder to save total space
                   used by the outputted dictionary

    '''
    
    misori = np.atleast_3d(misori)
    if misori.shape[0] == 1:
        misori = misori.T
    crd = np.atleast_3d(crd)
    if crd.shape[0] == 1:
        crd = crd.T
    
    tVol, wts = fe.calcVol(lcrd, lcon)
    cen = np.sum(fe.centroidTet(lcrd, lcon)*wts, axis=1)
    
    jk, nelems, nframes = misori.shape
    realMisori = np.zeros((3, 3, nframes))
    xs = np.zeros((3, 3, nframes))
    winv = np.zeros((3, 3, nframes))
    xsinv = np.zeros((3, 3, nframes))
    realBar = np.zeros((3, 3, nframes))
    us = np.zeros((3, 3, nframes))
    vs = np.zeros((3, 3, nframes))
    ss = np.zeros((3, 3, nframes))
    tVol = np.zeros((nframes,))
    gspread = np.zeros((nframes,))
    
    for i in range(nframes):
        tcrd = crd[:, :, i]
        tVol[i], wts = fe.calcVol(tcrd, lcon)
        elcen = fe.centroidTet(tcrd, lcon)
        if crdOpt:
            tstat = bartonStats(misori[:, :, i], tcrd)
        else:
            tstat = bartonStats(misori[:, :, i], elcen, *wts)
        realMisori[:, :, i] = tstat['W']
        xs[:, :, i] = tstat['X']
        winv[:, :, i] = tstat['Winv']
        xsinv[:, :, i] = tstat['Xinv']
        realBar[:, :, i] = tstat['WX']
        us[:, :, i], ss[:, :, i], vs[:, :, i] =np.linalg.svd(realBar[:, :, i])
        gspread[i] = np.sqrt(np.trace(winv[:, :, i]))
        
    stats = {'W':realMisori, 'X':xs, 'Winv':winv, 'Xinv':xsinv, 'WX':realBar,
            'U':us, 'S':ss, 'V':vs, 'GrainVol':tVol, 'centroid':cen, 'gSpread':gspread}
            
    return stats
    
    
def misorientationStats(misorient, *wts):
    '''
    MisorientationStats - Misorientation correlation statistics.

      USAGE:

      stats = MisorientationStats(misorient, locations)
      stats = MisorientationStats(misorient, locations, wts)

      INPUT:

      misorient is 4 x n, 
                a list of misorientation quaternions,
                assumed to have been derived from properly clustered 
                orientation data
      wts       is 1 x n, (optional)
                a list of weights; if not specified, uniform weights are used

      OUTPUT:

      stats is a structure with five components:

            W     is a 3 x 3 matrix (A in Barton paper)
            Winv  is a 3 x 3 matrix (A^-1 in Barton paper)
            wi    is 3 x n, the unnormalized axial vectors

      REFERENCE:  

      "A Methodology for Determining Average Lattice Orientation and 
      Its Application to the  Characterization of Grain Substructure",

      Nathan R. Barton and Paul R. Dawson,

      Metallurgical and Materials Transactions A,
      Volume 32A, August 2001, pp. 1967--1975
  

    '''
    misorient = utl.mat2d_row_order(misorient)
    d, n = misorient.shape  
    if len(wts) == 0:
        wts = np.tile(1.0/n, (3, n))
    else:
        wts = np.tile(wts, (3, 1))
        
    wts1 = np.tile(wts[0, :], (4, 1))
    misOriCen = sph.SphereAverage(misorient, **{'wts':wts1})
    misorient = misorient - np.tile(misOriCen, (1, n))
    
    ang = utl.mat2d_row_order(2*np.arccos(misorient[0, :]))
    wsc = np.zeros(ang.shape)
    
    limit = (ang < np.finfo(float).eps)
    nlimit = (ang > np.finfo(float).eps)
    
    angn = ang[nlimit]
    
    wsc[nlimit]= angn/np.sin(angn/2)
    wsc[limit] = 2
    
    wi = misorient[1:4, :]*np.tile(wsc.T, (3, 1))
    
    angax = np.zeros((4, n))
    angax[0, :] = np.linalg.norm(wi, axis = 0)
    angax[1:4, :] = normalize(wi, axis = 0)
    
    Winv = np.sum(utl.RankOneMatrix(wi*wts, wi), axis=2)
    
    #We needed to scale this up if it was close to being ill-conditioned
    if(np.abs(np.linalg.det(Winv)) < 1e-6):
        if(np.abs(np.linalg.det(Winv)) < 1e-16):
            W = np.zeros((3,3))
        else:
            Wtemp = np.multiply(1e9, Winv)
            W = np.multiply(1e9, np.linalg.inv(Wtemp))
    else:
        W = np.linalg.inv(Winv)
    
    stat = {'W':W, 'Winv':Winv, 'wi':wi, 'angaxis':angax}
    
    return stat
    
    
def misorientationTensor(misori, lcrd, lcon, crd , grnum, crdOpt = False):
    '''
        misOrientationTensor takes in the misorientation quaternions
    for the grain of interest and finds the various quantities located in
    Barton's paper "A Methodology for Determining Average Lattice Orientation and 
  Its Application to the  Characterization of Grain Substructure." 

    Inputs:
              misOri: the misorientation quaternions for all of the
                  frames of interest as calculated for the grain of interest.
              lcrd: the local crd of the grain mesh
              lcon: the local con of the grain mesh
              crd: the coordinates of each node at each time step
              crdOpt: if crdOpt is to be used

              grNum: The grain number of interest
      Outputs:
              stats: A structure with 4 different elements:
                     W     is a 3 x 3 x nframe matrix (A in Barton paper)
                     wi    is 3 x n x nframe, the unnormalized axial vectors
                     Winv  is a 3 x 3 x nframe matrix (A inv in Barton Paper)
                gSpread    is nframe, the grain spread as calculated from
                           Barton's paper using Winv. It is a scalar spread of
                           the misorientation blob.
                  angaxis  is 4 x n x nframe, the normalized axial vectors with 
                           the ang being 1st index  
                           
               ! - vars are currently commented out inorder to save total space
                   used by the outputted dictionary

    '''
    
    misori = np.atleast_3d(misori)
    if misori.shape[0] == 1:
        misori = misori.T
    crd = np.atleast_3d(crd)
    if crd.shape[0] == 1:
        crd = crd.T
    
    tVol, wts = fe.calcVol(lcrd, lcon)
    
    jk, nelems, nframes = misori.shape
    realMisori = np.zeros((3, 3, nframes))
    winv = np.zeros((3, 3, nframes))
    wi =  np.zeros((3, nelems, nframes))
    angaxis = np.zeros((4, nelems, nframes))
    gspread = np.zeros((nframes,))
    
    for i in range(nframes):
        tcrd = crd[:, :, i]
        tVol, wts = fe.calcVol(tcrd, lcon)
        if crdOpt:
            tstat = misorientationStats(misori[:, :, i])
        else:
            tstat = misorientationStats(misori[:, :, i], *wts)
        realMisori[:, :, i] = tstat['W']
        winv[:, :, i] = tstat['Winv']
        wi[:,:,i] = tstat['wi']
        angaxis[:,:,i] = tstat['angaxis']
        gspread[i] = np.sqrt(np.trace(winv[:, :, i]))
        
    stats = {'W':realMisori, 'wi':wi, 'angaxis':angaxis, 'Winv':winv,  'gSpread':gspread}
            
    return stats