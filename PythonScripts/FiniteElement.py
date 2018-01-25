import os
import numpy as np
import scipy as sci
import Misori as mis
import Rotations as rot
import Utility as utl
from scipy import optimize as sciop

'''
List of all functions available in this module
LoadQuadrature(qname)
calcVol(crd, con)
centroidTet(crd, con)
localConnectCrd(mesh, grNum)
concatConnArray(gconn, lconn, gupts, lupts, guelem, luelem)
deformationStats(defgrad, wts, crd, con, misrot,xtalrot, strain, kor)
nearvertex(sigs, vert, nverts, xtalrot, wts)
elem_fe_cen_val(crds, conn)
surface_quad_tet()
surfaceLoadArea(scrds, sconn, sig, wt2d, sfgdt)
sfmat()
gr_lstq_amat(conn, nsf, ncrds)
gr_lstq_solver(amat, q_mat, ncrds)
gr_nnlstq(amat, q_mat, ncrds)

'''



def LoadQuadrature(qname):
    '''
    LoadQuadrature - Load quadrature data rule.

      USAGE:

      qrule = LoadQuadrature(qname)

      INPUT:

      qname is a string, 
            the basename of the quadrature data files

      OUTPUT:

      qrule is a QRuleStructure, 
            it consists of the quadrature point locations and weights

      NOTES:

      *  It is expected that the quadrature rules are for simplices,
         and the last barycentric coordinate is appended to the file
         read from the data.

      *  Expected suffixes are .qps for the location and .qpw for
         the weights.

    '''
    path = os.getcwd()
    pathFile = path+'/data/Quadrature/'+qname
    
    try:
        pts = np.loadtxt(pathFile+'.qps').T
        wts = np.loadtxt(pathFile+'.qpw').T
    except FileNotFoundError as e:
        z = e
        print(z)
        
        raise ValueError('File name is wrong')

    n = pts.shape[1]

    pts = np.concatenate((pts, np.ones((1,n))-np.sum(pts, axis = 0)), axis = 0)

    qrule = {'pts':pts, 'wts':np.atleast_2d(wts)}

    return qrule
    
def calcVol(crd, con):
    '''
    Calculates the volume of an arbitary polyhedron made up of tetrahedron elems
    It also calculates the relative volume of each element compared to the
    polyhedron's volume, so elVol/polyVol and returns that as weight
    
    Input: crd - 3xn numpy array coordinate of the elements of the mesh
           con - 10xn numpy array connectivity array that says which nodes go
                 with what elements
    Output: vol - scalar value, total volume of the polyhedron
            wts - 1xn numpy array relative weight of each element for the polyhedron
    '''
    
    nelems = con.shape[1]
    wts = np.zeros((nelems,))
    
    for i in range(nelems):
        coord = np.squeeze(crd[:, con[[0, 2, 4, 9], i]])
        coord = np.concatenate((coord, [[1, 1, 1, 1]]), axis=0)
        wts[i] = 1.0/6.0*np.abs(np.linalg.det(coord))
        
    vol = np.sum(wts)
    wts = wts/vol
    
    return (vol, wts)
    
def centroidTet(crd, con):
    '''
    Calculates the centroid of a tetrahedron
    
    Input: crd - 3xn numpy array coordinate of the elements of the mesh
           con - 10xn numpy array connectivity array that says which nodes go
                 with what elements
    Output: cen - 3x1 numpy array centroid of the tetrahedron
    '''
    
    nelems = con.shape[1]
    centroid = np.zeros((3, nelems))
    for i in range(nelems):
        coord = np.squeeze(crd[:, con[[0, 2, 4, 9],i]])
        centroid[:, i] = np.sum(coord, axis=1)/4.0
        
    return centroid
    
def localConnectCrd(mesh, grNum):
    '''
    Calculates the local connectivity based upon the grain number provided
    
    Input: mesh - a dict structure given by the FePX_Data_and_Mesh module'
           grNum - an integer that corresponds to the grain of interest
    
    Output:con - 10xn numpy array connectivity array that says which nodes go
                 with what elements and that have corrected the node numbers to
                 correspond to only be 0 - (nelem-1) in the grain
           crd - 3xn numpy array of coordinates that correspond to this grain
    '''    
    
    logInd = mesh['grains'] == grNum
    
    lenlogInd = len(logInd)
    elemInd = np.r_[0:lenlogInd]
    uElem = elemInd[logInd]
    con = mesh['con'][:, logInd]
    nelem = con.shape[1]
    vecCon = con.reshape((10*nelem,1))
    uPts = np.int32(np.unique(vecCon))
    crd = mesh['crd'][:, uPts]
    count = 0
    for i in uPts:
        vecCon[vecCon == i] = count
        count +=1
        
    con = np.int_(vecCon.reshape((10, nelem)))
    
    return (con, crd,  uPts, uElem)

def localGrainConnectCrd(mesh, grNum):
    '''
    Calculates the local connectivity based upon the grain number provided
    
    Input: mesh - a dict structure given by the FePX_Data_and_Mesh module'
           grNum - an integer that corresponds to the grain of interest
    
    Output:con - 10xn numpy array connectivity array that says which nodes go
                 with what elements and that have corrected the node numbers to
                 correspond to only be 0 - (nelem-1) in the grain
           crd - 3xn numpy array of coordinates that correspond to this grain
    '''    
    
    logInd = mesh['grains'] == grNum
    
    lenlogInd = len(logInd)
    elemInd = np.r_[0:lenlogInd]
    uElem = elemInd[logInd]
    con = mesh['crd_con'][:, logInd]
    nelem = con.shape[1]
    vecCon = con.reshape((10*nelem,1))
    uPts = np.int32(np.unique(vecCon))
    count = 0
    for i in uPts:
        vecCon[vecCon == i] = count
        count +=1
        
    con = np.int_(vecCon.reshape((10, nelem)))
    
    return (con, uPts, uElem)

def concatConnArray(gconn, lconn, gupts, lupts, guelem, luelem):
    '''
        Takes in a "global" connectivity array and adds one with local node
        values to it after offsetting by the previous highest node number.
        This function essentially gives us multiple connectivity arrays
        glued together but not necessarily connected with each other.
        The main purpose of it is to have a number of grains containg their
        own mesh, but overall it looks like one big mesh.
        It also takes in a global and local unique pts array and just
        concatentate them together. Thus it is possible in this new array
        to have multiples of the same pts listed in it. The same concat
        is down with the unique element arrays but here we should not see
        the possibilities of multiple elements repeated.
        
        Input:
            gconn - a "global" connectivity array and it's 0 based
            lconn - a local connectivity array that needs to be offset using
                the largest value in gconn and it's 0 based
            gupts - a 1D numpy array that contains all the "unique" pts in
                in the conn array, but will really be used in rearranging
                nodal data from the original mesh.
            lupts - a 1D numpy array that is going to be added to gupts
            guelem - a 1D numpy array that contains all of the unique elem
                numbers. It's use is to rearrange the data matrix
            luelem - a 1D numpy array that is going to be added to gulem
        
        Output:
            fconn - the final connectivity array
            fupts - the final "unique" pts list
            fuelem - the final unique elements list
    '''
    
    if gconn.size == 0:
        mconn = 0
        fconn = lconn
    else:
        mconn = np.max(np.max(gconn))
        lconn[:,:] = lconn[:,:] + mconn + 1
        fconn = np.hstack((gconn, lconn))
        
    fupts = np.hstack((gupts, lupts))
    fuelem = np.hstack((guelem, luelem))
    
    return (fconn, fupts, fuelem)
    
    
    
def deformationStats(defgrad, wts, crd, con, misrot, xtalrot, strain, kor):
    '''
    Performs statistics on the deformation gradient for a particular mesh
    So, it gets the mean difference in minimum and maximum principle eig. vals.
    of the V matrix. Then it also returns the std. of the matrix
    It also gets the spread of the rotation matrix based on the values sent to
    the misorientationstats function
    
    Input:defgrad - a nx3x3 numpy array of the deformation gradient for the mesh
          wts - a n numpy vec of relative wts of the elements
          crd - a nx3 numpy array of the crd of the mesh (used in getting rSpread)
          con - a nx10 numpy array of the connectivity of the mesh (used in getting rSpread)
          
    Output:data - a dict that contains the following
           mFgrad - mean deformation gradient of the mesh
           mVpr - mean difference in the principal components of the
                  right stretch tensor, V, of the mesh
           sdVpr - standard deviation of the difference in the principal components
                   of the right stretch tensor, V, of the mesh
           rSpread - the mean kernal average of the spread of the rotation matrix
                     across the mesh
    '''

    vvec = np.zeros((6, defgrad.shape[0]))
    velasvec = np.zeros((6, defgrad.shape[0]))
    fvec = np.zeros((9, defgrad.shape[0]))
    fevec = np.zeros((9, defgrad.shape[0]))
    rkocks = np.zeros((defgrad.shape[1], defgrad.shape[0]))
    
    kocks = rot.OrientConvert(np.eye(3), 'rmat', 'kocks', 'degrees', 'degrees')
    
    wts = wts/np.sum(wts)
    wts1 = np.tile(wts, (6, 1))
    wts2 = np.tile(wts, (9, 1))
    
    fvec[0, :] = defgrad[:, 0, 0]
    fvec[1, :] = defgrad[:, 1, 1]
    fvec[2, :] = defgrad[:, 2, 2]
    
    fvec[3, :] = defgrad[:, 1, 2]
    fvec[4, :] = defgrad[:, 0, 2]
    fvec[5, :] = defgrad[:, 0, 1]
    
    fvec[6, :] = defgrad[:, 2, 1]
    fvec[7, :] = defgrad[:, 2, 0]
    fvec[8, :] = defgrad[:, 1, 0]
    
#    V = np.zeros((defgrad.shape[1], defgrad.shape[2], defgrad.shape[0]))
#    R = np.zeros((defgrad.shape[1], defgrad.shape[2], defgrad.shape[0]))    
#    eV = np.zeros((1, defgrad.shape[0]))
#    eVe = np.zeros((1, defgrad.shape[0]))
#    rchange = np.zeros((1, defgrad.shape[0]))
#    print(defgrad.shape)
#    rpm = np.zeros((defgrad.shape[0], 3, 3))
#    Fpm = np.zeros((defgrad.shape[0], 3, 3))
#    veinv = np.zeros((defgrad.shape[0], 3, 3))
#    vem = np.zeros((defgrad.shape[0], 3, 3))
#    rpkocks = np.zeros((defgrad.shape[1], defgrad.shape[0]))
#    Rm = np.zeros((defgrad.shape[0], 3, 3))
#    Vm = np.zeros((defgrad.shape[0], 3, 3))
#    upm = np.zeros((defgrad.shape[0], 3, 3))
#    reye = np.zeros((defgrad.shape[0], 3, 3))
#    defplgrad = np.zeros((defgrad.shape[0], 3, 3))

    for i in range(defgrad.shape[0]):    
        
        R, V = sci.linalg.polar(defgrad[i, :, :], 'left')
        rkocks[:,i] = np.squeeze(rot.OrientConvert(R, 'rmat', 'kocks', 'degrees', 'degrees'))
        
        vvec[0, i] = V[0,0]
        vvec[1, i] = V[1,1]
        vvec[2, i] = V[2,2]
        vvec[3, i] = V[1,2]
        vvec[4, i] = V[0,2]
        vvec[5, i] = V[0,1]

        rxtal = np.squeeze(rot.OrientConvert(misrot[:, i], 'quat', 'rmat', 'degrees', 'degrees'))
        xtalrmat = np.squeeze(rot.OrientConvert(xtalrot[:, i], kor, 'rmat', 'degrees', 'degrees'))
        velas = np.eye(3) +  xtalrmat.dot(strain[i, :, :].dot(xtalrmat.T)) #convert strain from lattice to sample

        elasdefgrad = velas.dot(rxtal)
        
        velasvec[0, i] = velas[0,0]
        velasvec[1, i] = velas[1,1]
        velasvec[2, i] = velas[2,2]
        velasvec[3, i] = velas[1,2]
        velasvec[4, i] = velas[0,2]
        velasvec[5, i] = velas[0,1]
    
        fevec[0, i] = elasdefgrad[0, 0]
        fevec[1, i] = elasdefgrad[1, 1]
        fevec[2, i] = elasdefgrad[2, 2]
    
        fevec[3, i] = elasdefgrad[1, 2]
        fevec[4, i] = elasdefgrad[0, 2]
        fevec[5, i] = elasdefgrad[0, 1]
    
        fevec[6, i] = elasdefgrad[2, 1]
        fevec[7, i] = elasdefgrad[2, 0]
        fevec[8, i] = elasdefgrad[1, 0]
    
#        velasinv = sci.linalg.inv(velas)
#        vem[i, :, :] = velas
#        veinv[i, :, :] = velasinv
#        reye[i, :, :] = xtalrmat.T.dot(xtalrmat)
#        ftemp = sci.linalg.inv(velas).dot(defgrad[i, :, :]) 
#        ftemp = velasinv.dot(defgrad[i, :, :])
#        Fp = rxtal.T.dot(ftemp)
#        defplgrad[i, :, :] = Fp   
#        Fpm[i, :, :] = Fp
#        Rp, Up = sci.linalg.polar(Fp, 'right')
#        Rp = np.around(Rp, decimals=7)
#        Fpm[i, :, :] = Fp
#        rpm[i, :, :] = Rp
#        upm[i, :, :] = Up
#        rdecomp = rxtal.dot(Rp)
#        rdiff = R.T.dot(rdecomp)
#        rchange[:, i] = np.trace(rdiff-np.eye(3))
#        rpkocks[:,i] = np.squeeze(rot.OrientConvert(Rp, 'rmat', 'kocks', 'degrees', 'degrees'))
#        eV[:, i] = np.max(temp) - np.min(temp)
#        temp, junk = np.linalg.eig(velas)
#        eVe[:, i] = np.max(temp) - np.min(temp)
#        print(R.shape)
        
    cen = utl.mat2d_row_order(np.sum(velasvec*wts1, axis=1))
    vi = velasvec - np.tile(cen, (1, defgrad.shape[0]))
    vinv = np.sum(utl.RankOneMatrix(vi*wts1, vi), axis=2)
    vespread = np.atleast_2d(np.sqrt(np.trace(vinv[:, :])))
    
    cen = utl.mat2d_row_order(np.sum(vvec*wts1, axis=1))
    vi = vvec - np.tile(cen, (1, defgrad.shape[0]))
    vinv = np.sum(utl.RankOneMatrix(vi*wts1, vi), axis=2)
    vspread = np.atleast_2d(np.sqrt(np.trace(vinv[:, :])))
    
    cen = utl.mat2d_row_order(np.sum(fvec*wts2, axis=1))
    vi = fvec - np.tile(cen, (1, defgrad.shape[0]))
    vinv = np.sum(utl.RankOneMatrix(vi*wts2, vi), axis=2)
    fSpread = np.atleast_2d(np.sqrt(np.trace(vinv[:, :])))
    
    cen = utl.mat2d_row_order(np.sum(fevec*wts2, axis=1))
    vi = fevec - np.tile(cen, (1, defgrad.shape[0]))
    vinv = np.sum(utl.RankOneMatrix(vi*wts2, vi), axis=2)
    feSpread = np.atleast_2d(np.sqrt(np.trace(vinv[:, :])))
    
    misAngs, misQuats = mis.misorientationGrain(kocks, rkocks, [0], kor)
    stats = mis.misorientationTensor(misQuats, crd, con, crd, [0])
    rSpread = stats['gSpread']
    
#    misAngs, misQuats = mis.misorientationGrain(kocks, rpkocks, [0], kor)
#    stats2 = mis.misorientationBartonTensor(misQuats, crd, con, crd, [0])
#    indG = np.squeeze(misAngs > 0.0001)
#    rpSpread = stats2['gSpread']
#    mFgrad = np.average(defgrad, axis=0, weights=wts)
#    mVpr = np.atleast_2d(np.average(eV, axis=1, weights=wts)).T
#    var = np.atleast_2d(np.average((eV-mVpr)**2, axis=1, weights=wts)).T
#    sdVpr = np.sqrt(var)
#    
#    mVpre = np.atleast_2d(np.average(eVe, axis=1, weights=wts)).T
#    var = np.atleast_2d(np.average((eVe-mVpre)**2, axis=1, weights=wts)).T
#    sdVpre = np.sqrt(var)
#    mFpgrad = np.average(defplgrad, axis=0, weights=wts)
#    rchg = np.atleast_2d(np.average(rchange,axis=1, weights=wts)).T
#    data = {'mVpr':mVpr, 'sdVpr':sdVpr, 'rSpread':rSpread, 'mFgrad':mFgrad, 
#    'rchg':rchg, 'mVpre':mVpre, 'sdVpre':sdVpre, 'rpSpread':rpSpread, 'mFpgrad':mFpgrad,
#    'vespread':vespread, 'vspread':vspread}

    data = {'veSpread':vespread, 'vSpread':vspread, 'rSpread':rSpread, 'fSpread':fSpread, 'feSpread':feSpread}
    
    return data
    
def nearvertex(sigs, vert, nverts, xtalrot, wts):
    '''
    Finds angle of nearest nearest vertice to the deviatoric stress of the crystal.
    The stress should be in the crystal reference frame first, since the vertices
    are based upon crystal reference frame values and not sample frame values.
    
    Input:
        sig = nelemx3x3 stress in the sample frame
        vert = nverts*3 x 3 vertice values taken from the FEPX vertice file
        nverts = number of vertices depends on xtal type
        angs = nelemx3 kocks angles of xtal in grain
    Output:
        angs = mean smallest absolute angle value from zero and shows how close
               the stress is from one of the initial vertices across grain.
    '''
    
    nelem = sigs.shape[0]

    xtalsig = np.zeros((nelem, 3, 3))    
    
    ind = np.r_[0, 3, 5]
    
    
    for i in range(nelem):
        maxang = np.pi/2
        xtalrmat = np.squeeze(rot.OrientConvert(xtalrot[:, i], 'kocks', 'rmat', 'degrees', 'degrees'))
        xtalsig[i, :, :] = xtalrmat.T.dot(sigs[i, :, :].dot(xtalrmat)) # convert stress from sample to xtal basis
    
    xtalsig = np.average(xtalsig, axis=0, weights=wts)    
    sig = np.atleast_2d(np.ravel(xtalsig)[np.r_[0,1,2,4,5,8]])
        
    for j in range(nverts):
            dsig = sig[0, :] - 1/3*np.sum(sig[0, ind])*np.asarray([1,0,0,1,0,1])
            tvert = np.ravel(vert[np.r_[j*3:(j+1)*3], :])
            dotp = np.sum(dsig*tvert[np.r_[0,1,2,4,5,8]])
            ang = np.arccos(dotp/(np.linalg.norm(dsig)*np.linalg.norm(tvert[np.r_[0,1,2,4,5,8]])))
            
            if abs(ang) < maxang:
                maxang = ang
            
    mangs = maxang
        
    return mangs


def elem_fe_cen_val(crds, conn):
    '''
        Takes in the raw values at the coordinates and gives the value at
        the centroid of a quadratic tetrahedral element using finite
        element shape functions.
        
        Input: crds - the 3d vector at each node of the mesh
               conn - the connectivity array that takes the crd values and
                      gives the elemental values
        Output: ecrds - the elemental 3d vector at the centroid of each
                    element in the mesh
                    
    '''
    
    nelems = conn.shape[1]
    ecrds = np.zeros((3, nelems))
    tcrds = np.zeros((3, 10))
    
    loc_ptr = np.ones(3) * 0.25
    
    sfvec_ptr = np.zeros((10))
    
    NT = np.zeros((10,1))
    
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
    
    NT[:, 0] = sfvec_ptr[:]
    
    
    for i in range(nelems):
        tcrds = crds[:, conn[:, i]]
        ecrds[:, i] = np.squeeze(np.dot(tcrds, NT))

    return ecrds

def surface_quad_tet():
    '''
    Outputs: quadrature points for quad tet surface
             quadrature weights for quad tet surface
             sf for quad tet surface
             grad sf for quad tet surface
    '''
    
    # ** 6-noded triangular element **

    # quadrature points
    
    qp2d = np.zeros((2,7))
    wt2d = np.zeros(7)
    sf = np.zeros((6, 7))
    sfgd = np.zeros((2, 6, 7))

    qp2d[0, 0] = 0.33333333333333           
    qp2d[0, 1] = 0.05971587178977            
    qp2d[0, 2] = 0.47014206410512            
    qp2d[0, 3] = 0.47014206410512            
    qp2d[0, 4] = 0.79742698535309            
    qp2d[0, 5] = 0.10128650732346            
    qp2d[0, 6] = 0.10128650732346            

    qp2d[1, 0] = 0.33333333333333
    qp2d[1, 1] = 0.47014206410512
    qp2d[1, 2] = 0.05971587178977
    qp2d[1, 3] = 0.47014206410512
    qp2d[1, 4] = 0.10128650732346
    qp2d[1, 5] = 0.79742698535309
    qp2d[1, 6] = 0.10128650732346

    # weight

    wt2d[0] = 0.1125
    wt2d[1] = 0.06619707639425
    wt2d[2] = 0.06619707639425
    wt2d[3] = 0.06619707639425
    wt2d[4] = 0.06296959027241
    wt2d[5] = 0.06296959027241
    wt2d[6] = 0.06296959027241
     
    for i in range(7):
       xi  = qp2d[0, i]
       eta = qp2d[1, i]
       zeta = 1.0 - xi - eta
       # nodal locations:
       #
       # 3
       # 42
       # 561
       #
       #
       sf[0, i] = (2.0 * xi - 1.0) * xi
       sf[1, i] = 4.0 * eta * xi
       sf[2, i] = (2.0 * eta - 1.0) * eta
       sf[3, i] = 4.0 * eta * zeta
       sf[4, i] = (2.0 * zeta - 1.0) * zeta
       sf[5, i] = 4.0 * xi * zeta
       
       sfgd[0, 0, i] = 4.0 * xi - 1.0
       sfgd[0, 1, i] = 4.0 *eta
       sfgd[0, 2, i] = 0.0
       sfgd[0, 3, i] = -4.0 * eta
       sfgd[0, 4, i] = -4.0 * zeta + 1.0
       sfgd[0, 5, i] = 4.0 * zeta - 4.0 * xi
       
       sfgd[1, 0, i] = 0.0
       sfgd[1, 1, i] = 4.0 * xi
       sfgd[1, 2, i] = 4.0 * eta - 1.0
       sfgd[1, 3, i] = 4.0 * zeta - 4.0 * eta
       sfgd[1, 4, i] = -4.0 * zeta + 1.0
       sfgd[1, 5, i] = -4.0 * xi
       
    return (qp2d, wt2d, sf, sfgd)

def surfaceConn(scrds, sconn, surf):
    '''
        All of the surface nodes and connectivity matrix are taken in and
        only the ones related to the surface of interest are returned.
        This function assummes a cubic/rectangular mesh for now.
        The surf number of interest is given where:
            z1 = min z surface
            z2 = max z surface
            y1 = min y surface
            y2 = max y surface
            x1 = min x surface
            x2 = max x surface
        Input: scrds = a 3xnsurf_crds size array where all of the surface
                coords are given
               sconn = a 7xnsurf_elem size array where all of the surface
                conn are given. It should also be noted that the 1st elem
                is the element number that the surface can be found on
               surf = a string with the above surf numbers as its values
        Output: gconn = the global surface connectivity array for the
                     surface of interest
                lconn = the local surface connectivity array for the
                     surface of interest
    '''
    
    nelems = sconn.shape[1]
    
    logInd = np.zeros(nelems, dtype=bool)

    if surf == 'x1':
        ind = 0
        val = np.min(scrds[ind,:])
    elif surf == 'x2':
        ind = 0
        val = np.max(scrds[ind,:])
    elif surf == 'y1':
        ind = 1
        val = np.min(scrds[ind,:])
    elif surf == 'y2':
        ind = 1
        val = np.max(scrds[ind,:])
    elif surf == 'z1':
        ind = 2
        val = np.min(scrds[ind,:])
    else:
        ind = 2
        val = np.max(scrds[ind,:])
    
    for i in range(nelems):
        ecrds = scrds[ind, sconn[1:7, i]]
        logInd[i] = np.all(ecrds == val)
    
    gconn = sconn[:, logInd]
    
    lconn = np.copy(gconn)
    
    nelem = lconn.shape[1]
    
    vecCon = lconn[0, :]
    uCon = np.int32(np.unique(vecCon))
    count = 0
    for i in uCon:
        vecCon[vecCon == i] = count
        count +=1
    
    lconn[0, :] = np.int_(vecCon)
    
    vecCon = lconn[1:7, :].reshape((6*nelem,1))
    uCon = np.int32(np.unique(vecCon))
    count = 0
    for i in uCon:
        vecCon[vecCon == i] = count
        count +=1
        
    lconn[1:7, :] = np.int_(vecCon.reshape((6, nelem)))
    
    return (gconn, lconn)    
    
def surfaceLoadArea(scrds, sconn, sig, wt2d, sfgdt):
    '''
    Takes in the surface coordinates and surface element centroidal stress
    Then computes the area on the surface and load on the surface
    Input: scrds = a 3xnsurf_coords
           sconn = a 6xnsurf_elems vector of the surface connectivity
           sig = a 6xnsurf_elems vector of the Cauchy stress
           wt2d = a n length vector of the surface quad pt weights
           sfgdt = a 6x2xn length vector of the trans grad surf interp array 
    Output: load = a vector of size 3 equal to the load on the surface
            area = a scalar value equal to the surface area
    '''
    
    load = np.zeros(3)
    area = 0.0
    
    nselem = sig.shape[1]
    nqpts = wt2d.shape[0]
    
    tangent = np.zeros((3, 2))
    normal = np.zeros(3)
    sjac = 0.0
    
    nmag = 0.0
    
    for i in range(nselem):
        tangent = 0.0
        normal = 0.0
        #Get an array of the element surface crds
        ecrds = scrds[:, sconn[:, i]]
        for j in range(nqpts):
            #The two tangent vectors are just ecrds*sfgdt
            tangent = ecrds.dot(sfgdt[:, :, j])
            #The normal is just t1 x t2
            normal = np.cross(tangent[:, 0], tangent[:, 1])
            #The normal is just the L2 norm of n
            nmag = np.sqrt(np.inner(normal, normal[:]))
            #sjac is just the L2 norm of the normal vec
            sjac = nmag
            #Normalize the normal vec
            normal = normal / nmag
            #Now compute the loads and area given above info
            load[0] += wt2d[j] * sjac * (sig[0, i]*normal[0] +\
                sig[1, i]*normal[1] + sig[2, i]*normal[2])
            load[1] += wt2d[j] * sjac * (sig[1, i]*normal[0] +\
                sig[3, i]*normal[1] + sig[4, i]*normal[2])
            load[2] += wt2d[j] * sjac * (sig[2, i]*normal[0] +\
                sig[4, i]*normal[1] + sig[5, i]*normal[2])
            area += wt2d[j] * sjac
    
    return (load, area)
    
def sfmat():
    '''
    Outputs the shape function matrix for a 10 node tetrahedral element
    Pretty much just using FEpX
    Output: N - The isoparametric shape functions for all 15 quadrature points
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

def iso_dndx():
    '''
        Returns the isoparametric dndx matrix
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
    
    iso_dndx = np.zeros((3,10,15))
    dndx_ptr = np.zeros((30))    
    
    for i in range(15):
        loc_ptr = qp3d_ptr[i*3:(i+1)*3]
        dndx_ptr[0]  = 4.0e0 * (loc_ptr[0] + loc_ptr[1] + loc_ptr[2]) - 3.0e0;
        dndx_ptr[1]  = 4.0e0 * (loc_ptr[0] + loc_ptr[1] + loc_ptr[2]) - 3.0e0;
        dndx_ptr[2]  = 4.0e0 * (loc_ptr[0] + loc_ptr[1] + loc_ptr[2]) - 3.0e0;
        dndx_ptr[3]  = -4.0e0 * (2.0e0 * loc_ptr[0] + loc_ptr[1] +loc_ptr[2] - 1.0e0);
        dndx_ptr[4]  = -4.0e0 * loc_ptr[0];
        dndx_ptr[5]  = -4.0e0 * loc_ptr[0];
        dndx_ptr[6]  = 4.0e0 * loc_ptr[0] - 1.0e0;
        dndx_ptr[7]  = 0.0e0;
        dndx_ptr[8]  = 0.0e0;
        dndx_ptr[9]  = 4.0e0 * loc_ptr[1];
        dndx_ptr[10] = 4.0e0 * loc_ptr[0];
        dndx_ptr[11] = 0.0e0;
        dndx_ptr[12] = 0.0e0;
        dndx_ptr[13] = 4.0e0 * loc_ptr[1] - 1.0e0;
        dndx_ptr[14] = 0.0e0;
        dndx_ptr[15] = -4.0e0 * loc_ptr[1];
        dndx_ptr[16] = -4.0e0 * (loc_ptr[0] + 2.0e0 * loc_ptr[1] + loc_ptr[2] - 1.0e0);
        dndx_ptr[17] = -4.0e0 * loc_ptr[1];
        dndx_ptr[18] = -4.0e0 * loc_ptr[2];
        dndx_ptr[19] = -4.0e0 * loc_ptr[2];
        dndx_ptr[20] = -4.0e0 * (loc_ptr[0] + loc_ptr[1] + 2.0e0 * loc_ptr[2] - 1.0e0);
        dndx_ptr[21] = 4.0e0 * loc_ptr[2];
        dndx_ptr[22] = 0.0e0;
        dndx_ptr[23] = 4.0e0 * loc_ptr[0];
        dndx_ptr[24] = 0.0e0;
        dndx_ptr[25] = 4.0e0 * loc_ptr[2];
        dndx_ptr[26] = 4.0e0 * loc_ptr[1];
        dndx_ptr[27] = 0.0e0;
        dndx_ptr[28] = 0.0e0;
        dndx_ptr[29] = 4.0e0 * loc_ptr[2] - 1.0e0;
        iso_dndx[:,:,i] = np.reshape(dndx_ptr, (3, 10), order='F')

    return iso_dndx

def local_gradient_shape_func(iso_dndx, elem_crd, iqpt):
    '''
    This function takes in element coordinates, and then using the parent
     gradient shape functions that are constant through out the simulation
     it calculates the local gradient shape functions for each element at
     each quadrature point.
     Input:
         iso_dndx - the isoparametric dndx matrix.
         elem_crd - the elemental coordinate array - nnpex3xnelems
         iqpt     - the quadrature point are we interested in examining
     
     Output: 
         loc_dndx - The local gradient shape functions at the quadrature point
             for each element. It has a shape of 3xnnpexnelems
          qpt_det - The determinant at the quadrature point for each element.
                It has a shape of nelems
    '''
    
    nelems = elem_crd.shape[2]
    nnpe = elem_crd.shape[0]
    
    loc_dndx = np.zeros((3, nnpe, nelems), dtype='float64', order='F')
    qpt_det = np.zeros((nelems), dtype='float64', order='F')
    jac = np.zeros((3,3), dtype='float64', order='F')
    ijac = np.zeros((3,3), dtype='float64', order='F')
    
    
    for i in range(nelems):
        jac = np.dot(iso_dndx[:,:,iqpt], elem_crd[:,:,i])
        ijac = np.linalg.inv(jac)
        qpt_det[i] = ((ijac[0,0]*ijac[1,1]*ijac[2,2] + ijac[0,1]*ijac[1,2]*ijac[2,0] + ijac[0,2]*ijac[1,0]*ijac[2,1])
                    - (ijac[0,2]*ijac[1,1]*ijac[2,0] + ijac[0,1]*ijac[1,0]*ijac[2,2] + ijac[0,0]*ijac[1,2]*ijac[2,1]))
        loc_dndx[:,:,i] = np.dot(ijac, iso_dndx[:,:,iqpt])
        
    return (loc_dndx, qpt_det)
 
def get_scalar_grad(scalar, loc_dndx):
    '''
    This function uses the local gradient shape functions from each element
    to calculate the gradient of a scalar field.
    
    Input:
        scalar - the scalar field which is a nnpexnelem
        loc_dndx - the local gradient shape function which is a 3xnnpexnelems
    Output:
        scalar_grad - the scalar gradient at some quadrature point within the
        element. It has a size of 3xnelems
    '''
    
    nelems = scalar.shape[1]
    
    scalar_grad = np.zeros((3, nelems), dtype='float64', order='F')
    
    for i in range(nelems):
        scalar_grad[:, i] = np.dot(scalar[:, i], loc_dndx[:,:,i].T)

    return scalar_grad    


def get_vec_grad(vec, loc_dndx):
    '''
    This function uses the local gradient shape functions from each element
    to calculate the gradient of a vector field.
    
    Input:
        vec - the vector field which is a 3xnnpexnelem
        loc_dndx - the local gradient shape function which is a 3xnnpexnelems
    Output:
        scalar_grad - the vector gradient at some quadrature point within the
        element. It has a size of 3x3xnelems
    '''
    
    nelems = vec.shape[2]
    
    vec_grad = np.zeros((3, 3, nelems), dtype='float64', order='F')
    
    for i in range(nelems):
        vec_grad[:, :, i] = vec[:,:,i].dot(loc_dndx[:,:,i].T)

    return vec_grad 

def get_nye_tensor(vec_grad):
    '''
    This function uses the lattice orientation gradient field
    and computes the nye tensor.
    
    Input:
        vec_grad - the vector field which is a 3x3xnelem
    Output:
        nye_ten - the nye tensor which is a 3x3xnelem
    '''
    nelems = vec_grad.shape[2]
    vec_gradT = np.swapaxes(vec_grad, 0, 1)
    trT = np.zeros((3,3,nelems), dtype='float64', order='F')
    for i in range(nelems):
        trT[:,:,i] = 0.5 * np.eye(3)*np.trace(np.squeeze(vec_gradT[:,:,i]))
    nye_ten = vec_gradT - trT
    
    return nye_ten

def get_l2_norm_dd(nye_ten, l2mat):
    '''
    It takes in the nye tensor and outputs the dislocation density using
    the L2 norm method. It also takes in the L2 matrix from Arsenlis 1999 paper.
    
    Input: nye_ten - the nye tensor as calculated in the above function which is
                    a 3x3xnelem
           l2mat - the L2 mat as given in Arsenlis 1999 paper for FCC materials. If
                  you had a different system then the L2 matrix would need to be calculated
                  ahead of time. The size is nslip x 9
           
    Output: dd - the dislocation density for the 12 main slip systems in an FCC material if
                 the L2 mat is from the 1999 Arsenlis paper. If it isn't then it is however
                 many main slip systems in your material. The size is nslip x nelems
    
    '''
    
    nelems = nye_ten.shape[2]
    dd = np.zeros((12, nelems), dtype='float64', order='F')
    alpha = nye_ten.reshape(9, nelems)
    
    for i in range(nelems):
        dd[:, i] = l2mat.dot(alpha[:,i])
        
    return dd

def get_l2_matrix():
    '''
    The L2 mat from Arsenlis 1999 paper for GND calculations for an FCC material
    Output: l2mat - The l2mat as given in the paper which is 12 x 9.
    '''
    
    a = np.sqrt(3.0)/9.0
    c = np.sqrt(3.0)/84.0
    z = 0.0
    
    l2mat = np.zeros((12, 9), dtype='float64', order='F')
    
    l2mat[:, 0] = [a, -a, z, a, -a, z, a, -a, z, a, -a, z]
    l2mat[:, 1] = [7.0*c, 13.0*c, c, -7.0*c, -13.0*c, -c, -7.0*c, -13.0*c, -c, 7.0*c, 13.0*c, c]
    l2mat[:, 2] = [-13.0*c, -7.0*c, -c, 13.0*c, 7.0*c, c,-13.0*c, -7.0*c, -c, 13.0*c, 7.0*c, c]
    l2mat[:, 3] = [7.0*c, c, 13.0*c, -7.0*c, -c, -13.0*c, -7.0*c, -c, -13.0*c, 7.0*c, c, 13.0*c]
    l2mat[:, 4] = [-a, z, a, -a, z, a, -a, z, a, -a, z, a]
    l2mat[:, 5] = [13.0*c, c, 7.0*c, 13.0*c, c, 7.0*c, -13.0*c, -c, -7.0*c, -13.0*c, -c, -7.0*c]
    l2mat[:, 6] = [c, 7.0*c, 13.0*c, -c, -7.0*c, -13.0*c, c, 7.0*c, 13.0*c, -c, -7.0*c, -13.0*c]
    l2mat[:, 7] = [-c, -13.0*c, -7.0*c, -c, -13.0*c, -7.0*c, -c, 13.0*c, 7.0*c, c, 13.0*c, 7.0*c]
    l2mat[:, 8] = [z, a, -a, z, a, -a, z, a, -a, z, a, -a]
    
    return l2mat

def sf_qpt_wts():
    '''
    Quadrature point weights for a 10 node tet
    '''
    
    wtqp = np.zeros(15)
    wtqp[0:4]   = 0.602678571428571597e-2
    wtqp[4]     = 0.302836780970891856e-1
    wtqp[5:9]   = 0.116452490860289742e-1
    wtqp[9:15]  = 0.109491415613864534e-1

    return wtqp    

def sftsfmat():
    '''
        Creates a NTN array that has the appropriate weights applied at
        each quadratutre point.
        Also return NT with appropriate weight applied to it
    '''
    qpt_wts = sf_qpt_wts()
    N = sfmat()
    NT = N.T
    
    NTN = np.zeros((10,10,15))
    
    for i in range(15):
        NTN[:,:,i] = np.outer(N[i,:], N[i,:]) * qpt_wts[i]
        NT[:,i] = NT[:,i] * qpt_wts[i]

    return (NTN, NT)        

def gr_lstq_amat(conn, nsf, ncrds):
    '''
        Inputs:
                conn - the local connectivity array a nelem x 10 size array
                nsf - the shape function matrix
                ncrds - number of coordinates/nodal points in the grain
        Output:
                amat - the matrix used in our least squares problem for the grain
                     It will be constant through out the solution.
    '''
    
    nelems = conn.shape[1]
    nqpts = nsf.shape[0]
    amat = np.zeros((nelems*nqpts, ncrds))
    #Build up our A matrix to be used in a least squares solution
    j = 0
    k = 0
    for i in range(nelems):
        j = i * nqpts
        k = (i + 1) * nqpts
        ecrds = np.squeeze(conn[:, i])
        amat[j:k,  ecrds] = nsf
    
    return amat
    
def gr_lstq_solver(amat, q_mat, ncrds):
    '''
        Inputs:
                conn - the local connectivity array a nelem x 10 size array
                q_mat - vector at each quad point
                        size = nqpts x nvec x nelems
                ncrds - number of coordinates/nodal points in the grain
        Output:
                nod_mat - the nodal values of a grain for the q_mat
                residual - the residual from the least squares
                
        A  least squares  routine is used to solve for the solution. 
        It'll find the nodal values of the points at the quadrature mat for
        a grain.
    '''
    
    nvec = q_mat.shape[1]
    nqpts = q_mat.shape[0]
    nelems = q_mat.shape[2]
    nod_mat = np.zeros((nvec,ncrds), dtype='float64')
    b = np.zeros((nqpts*nelems))
    residual = np.zeros(nvec)
    
    for i in range(nvec):
        b[:] = np.ravel(q_mat[:, i, :], order = 'F')
        nod_mat[i, :], residual[i], t1, t2 = np.linalg.lstsq(amat, b)
        
    return (nod_mat, residual)

def gr_nnlstq(amat, q_mat, ncrds):
    '''
        Inputs:
                conn - the local connectivity array a nelem x 10 size array
                q_mat - vector at each quad point
                        size = nqpts x nvec x nelems
                ncrds - number of coordinates/nodal points in the grain
        Output:
                nod_agamma - the nodal values of a grain for the q_mat
                residual - the residual from the least squares
                
        A nonnegative nonlinear least squares optimization routine is used to solve for
        the solution. It'll find the nodal values of the absolute q_mat for
        a grain.
    '''
    
    nvec = q_mat.shape[1]
    nqpts = q_mat.shape[0]
    nelems = q_mat.shape[2]
    nod_mat = np.zeros((nvec,ncrds), dtype='float64')
    b = np.zeros((nqpts*nelems))
    residual = np.zeros(nvec)
    
    for i in range(nvec):
        b[:] = np.ravel(q_mat[:, i, :], order = 'F')
        nod_mat[i, :], residual[i] = sciop.nnls(amat, b)
        
    return (nod_mat, residual)

def superconvergence_mat(NTN, qpt_det, conn, ncrds):
    '''
    Input:
        NTN - the shape function transpose shape function outer product
            matrix with dimensions - nnpe x nnpe x nqpts
        qpt_det - the determinate of the jacobian matrix for each
                quadrature point of an element - dimensions nelem x nqpts
        conn - the connectivity array
        ncrds - the number of coordinates
    Output:
        amat - the superconvergence matrix
    '''
    nelems = conn.shape[0]
    nqpts = NTN.shape[2]
    nnpe = NTN.shape[0]
    amat = np.zeros((ncrds, ncrds))
    
    for i in range(nelems):
        for j in range(nqpts):
            for k in range(nnpe):
                ind = conn[i, k]
                amat[ind, conn[i, :]] = amat[ind, conn[i, :]] + NTN[k,:,j] * qpt_det[i,j] 
                 
    return amat 

def superconvergence_vec(NT, qpt_det, conn, qpt_vec, ncrds):
    '''
    Input
    NT - the transpose shape function
    qpt_det - the determinate of the jacobian matrix for each
            quadrature point of an element - dimensions nelem x nqpts
    conn - the connectivity array
    qpt_vec - vector at each quad point for nvecs
              size = nqpts x nvec x nelems
    ncrds - the number of coordinates
    
    Output:
        bvec - the integration of NT*qpt_vec over the domain product
            size is ncrds x nvec
    '''
    
    nqpts = qpt_det.shape[1]
    nelems = conn.shape[0]
    nvec = qpt_vec.shape[1]
    nnpe = conn.shape[1]
    bvec = np.zeros((ncrds, nvec))
    tarr = np.zeros((nqpts))
    tind = np.zeros((nnpe), dtype='int32')
    
    for i in range(nvec):
        for j in range(nelems):
            tind[:] = conn[j, :]
            tarr[:] = qpt_vec[:,i,j]*qpt_det[j, :]
            bvec[tind, i] = bvec[tind, i] + NT.dot(tarr)
            
    return bvec
    

def superconvergnce_solve(amat, bvec):
    '''
    Solves the superconvergence patch test problem to obtain values at the
    nodal coordinates
    Input:
        amat - our superconvergence matrix
        bvec - our superconvergence bvec with a size of ncrds x nvec
    Output
        xvec - our superconvergence nodal solutions with a size of ncrds x nvec
    '''
    
    xvec = np.linalg.solve(amat, bvec)

    return xvec.T 

def jacobian_lin(mesh):
    '''
    jacobian_lin - Compute Jacobian of linear mesh mappings.
    
      USAGE:
    
      jac = jacobian_lin(mesh)
    
      INPUT:
    
      mesh is a MeshStructure,
           with simplicial element type
    
      OUTPUT:
    
      jac is 1 x m, 
          the Jacobian of each element
    
      NOTES:
    
      *  The mesh may be embedded in a space of higher 
         dimension than the reference element.  In that
         case, the Jacobian is computed as (sqrt(det(J'*J))
         and is always positive.  When the target space is
         of the same dimension as the reference element,
         the Jacobian is computed as usual and can be
         positive or negative.
    
      *  Only simplicial (linear) element types are allowed. 
    '''
    
    crd = mesh['crd']
    con = mesh['con']
    
    e = con.shape[0]
    ddom = e - 1
    dtar = crd.shape[0]
    
    nels = con.shape[1]
    
    jac = np.zeros(nels)
    
    if (ddom == dtar):
        for i in range(nels):
            simp = crd[:, con[:, i].T]
            mat = simp[:, 0:ddom] - np.tile(simp[:, e] (1, ddom))
            jac[i] = np.linalg.det(mat)
    else:
        for i in range(nels):
            simp = crd[:, con[:, i].T]
            mat = simp[:, 0:ddom] - np.tile(simp[:, e] (1, ddom))
            mat = mat.T.dot(mat)
            jac[i] = np.sqrt(np.linalg.det(mat))
    
    return jac
    