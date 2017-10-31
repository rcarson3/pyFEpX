import numpy as np
import scipy as sci
import scipy.spatial as scisp
import Utility as utl
import FiniteElement as fem
import importlib

'''
This program was written by Robert Carson on June 10th, 2015.
It is based upon the OdfPf library that the Deformation Processing Lab has written in MATLAB.

The following functions are available in this module:

EqualArea
ProjStereo
PSphDistance
PSphGaussian
SphBaseMesh
XYZOfThetaPhi
SphCrdMesh
SpherePz
SphDistanceFunc

'''


def EqualArea(p3d, *args):
    '''
    EqualArea - Equal area projection on sphere.
  
      USAGE:

      p2d = EqualArea(p3d)
      p2d = EqualArea(p3d, basis)

      INPUT:

      p3d   is 3 x n,
            a list of n unit vectors
      basis is a 3 x 3 matrix, (optional)
            it's columns form an orthonormal basis used to find the
            projection; defaults to the identity

      OUTPUT:

      p2d is a real 2 x n array:
          the equal area projections of points in `p3d'

      NOTES:

      *  The equal area projection is computed using the 
         third basis vector as the pole.  Planar components
         are given relative to the first two basis vectors.


    '''
    p3d = np.atleast_2d(p3d)

    if len(args) == 0:
        pcrd = p3d
    else:
        pcrd = args[0].T * p3d

    zfac = np.sqrt(2 / (1 + pcrd[2, :]))
    p2d = np.tile(zfac, (2, 1)) * pcrd[0:2, :]

    return p2d


def ProjStereo(p3d, *args):
    '''
    ProjStereo - stereographic projection
  
      USAGE:

      p2d = ProjStereo(p3d)
      p2d = ProjStereo(p3d, basis)

      INPUT:

      p3d   is 3 x n,
            a list of n unit vectors
      basis is a 3 x 3 matrix, (optional)
            it's columns form an orthonormal basis used to find the
            projection; defaults to the identity

      OUTPUT:

      p2d is a real 2 x n array:
          the stereographic projections of points in `p3d'

      NOTES:

      *  The projection is computed from the south pole relative
         to the basis.
      *  Also not that there are many stereographic projections depending on
         where the projection plane is put relative to the pole.


    '''
    p3d = np.atleast_2d(p3d)

    if len(args) == 0:
        pcrd = p3d
    else:
        pcrd = args[0].T * p3d

    zfac = 1 + pcrd[2, :]
    p2d = np.concatenate((np.atleast_2d(pcrd[0, :] / zfac),
                          np.atleast_2d(pcrd[1, :] / zfac)), axis=0)

    return p2d


def PSphDistance(pt, ptlist):
    '''
    PSphDistance - Distance on projective sphere.

      USAGE:

      dist = PSphDistance(pt, ptlist)

      INPUT:

      pt     is 3 x 1, 
             a point on the unit sphere (S^2)
      ptlist is 3 x n, 
             a list of points on the unit sphere

      OUTPUT:

      dist is 1 x n, 
           the distance from `pt' to each point in the list

      NOTES:

      *  The distance between two points on the sphere is the angle
         in radians between the two vectors.  On the projective
         sphere, antipodal points are considered equal, so the 
         distance is the minimum of the distances obtained by
         including the negative of pt as well.

    '''

    pt = np.atleast_2d(pt)
    ptlist = np.atleast_2d(ptlist)

    n = ptlist.shape[1]
    ptmat = np.tile(pt, (1, n))

    pt1 = np.arccos(np.sum(ptmat.conj() * ptlist, axis=0))
    pt2 = np.arccos(np.sum((-1 * ptmat.conj()) * ptlist, axis=0))

    dist2 = np.concatenate((np.atleast_2d(pt1), np.atleast_2d(pt2)), axis=0)

    dist = np.min(dist2, axis=0)

    return dist


def PSphGaussian(center, pts, stdev):
    '''
    PSphGaussian - Gaussian distribution for smoothing on projective sphere.
  
      USAGE:

      fsm = PSphGaussian(center, pts, stdev)

      INPUT:

      center is 3 x 1, 
             the center of the distribution
      pts    is 3 x n, 
             a list of points on the sphere; antipodal
             points are considered equal
      stdev  is 1 x 1, 
             the (1D) standard deviation

      OUTPUT:

      fsm is 1 x n, 
          the list of values at each point of pts

      Notes:  

      *  The result is not normalized, so this may have to be
         done after the fact.
      *  The distribution is a 1-D normal distribution applied
         to the distance function on the projective sphere.
      *  The actual scaling factor to give unit integral over 
         the projective sphere is not computed; the result
         is not normalized.


    '''

    twosigsq = 2 * (stdev ** 2)
    theta = PSphDistance(center, pts)
    minusthetasq = -1 * theta * theta
    fsm = (1 / (stdev * np.sqrt(2 * np.pi))) * np.exp(minusthetasq / twosigsq)

    return fsm


def SphBaseMesh(dim, **kwargs):
    '''
    SphBaseMesh - Generate base mesh for spheres.
    
      USAGE:

      mesh = SphBaseMesh(dim)
      mesh = SphBaseMesh(dim, 'param', 'value')

      INPUT:

      dim is a positive integer,
          the dimension of the sphere (2 for the usual sphere S^2)

      These arguments can be followed by a list of
      parameter/value pairs which specify keyword options.
      Available options include:

      'Hemisphere'  'True | False'
                    to mesh only the upper hemisphere


      OUTPUT:

      mesh is a MeshStructure,
           on the sphere of the specified dimension

      NOTES:

      * For S^2, the normals may be mixed, some outer, some inner.  
        This needs to be fixed.


    '''

    if len(kwargs) == 0:

        HEMI = False

    else:

        HEMI = kwargs['Hemisphere']

    n = dim

    caxes = np.diag(np.arange(1, n + 2))
    pts = np.concatenate((np.zeros((n + 1, 1)), caxes, -caxes), axis=1)

    conlen = 2 ** (n + 1)

    if HEMI:
        pts = pts[:, 0:-1]
        conlen = 2 ** n

    tcon = scisp.Delaunay(np.transpose(pts), qhull_options='QJ').simplices.T
    con = tcon - 1
    con = np.atleast_2d(con[con > -1]).reshape((n + 1, conlen))

    mesh = {}

    mesh['con'] = con
    mesh['crd'] = utl.UnitVector(pts[:, 1:])
    mesh['simplices'] = tcon

    return mesh


def XYZOfThetaPhi(thetaphi):
    '''
    XYZOfThetaPhi - Map spherical coordinates to sphere.

      USAGE:

      xyz = XYZOfThetaPhi(thetaphi)

      INPUT:

      thetaphi is 2 x n, 
               the spherical coordinates for a list of n points; 
               theta is the angle that the projection onto x-y plane 
               makes with the x-axis, and phi is the angle with z-axis

      OUTPUT:

      xyz is 3 x n, 
          the Cartesian coordinates of the points described by (theta, phi)


    '''

    thetaphi = np.atleast_2d(thetaphi)

    if thetaphi.shape[0] == 1:
        thetaph = thetaphi.T

    theta = thetaphi[0, :]
    phi = thetaphi[1, :]

    ct = np.cos(theta)
    st = np.sin(theta)

    sp = np.sin(phi)

    xyz = np.concatenate((np.atleast_2d(sp * ct), np.atleast_2d(sp * st), np.atleast_2d(np.cos(phi))), axis=0)
    return xyz


def SphCrdMesh(ntheta, nphi, **kwargs):
    '''
    SphCrdMesh - Generate a hemisphere mesh based on spherical coordinates.
  
      USAGE:

      smesh = SphCrdMesh(ntheta, nphi)

      INPUT:

      ntheta is a positive integer,
             the number of subdivisions in theta
      nphi   is a positive integer,
             the number of subdivisions in phi

      These arguments can be followed by a list of
      parameter/value pairs which specify keyword options.
      Available options are listed below with default values
      shown in brackets.

      'MakeL2ip'     {'on'}|'off'
                     computes the L2 inner product matrix and 
                     adds it to the mesh structure as a field .l2ip
      'QRule'        string  {'qr_trid06p12'}
                     name of a quadrature rule for triangles, to be used
                     in building the l2ip matrix
      'PhiMax'       scalar
                     largest angle with the vertical z-axis, i.e. for 
                     incomplete pole figure data

      OUTPUT:

      smesh is a MeshStructure,
            on the hemisphere (H^2)

      NOTES:

      * No equivalence array is produced.
      
    '''
    makel2ip = kwargs.get('MakeL2ip', True)
    qrule = kwargs.get('QRule', 'qr_trid06p12')
    phimax = kwargs.get('PhiMax', np.pi / 2)

    tdiv = 2 * np.pi * np.arange(ntheta + 1) / ntheta
    pdiv = phimax * np.arange(nphi + 1) / nphi

    phi, theta = np.meshgrid(pdiv, tdiv)
    npts = (ntheta + 1) * (nphi + 1)

    thetaphi = np.concatenate((theta.T.reshape((1, npts)), phi.T.reshape((1, npts))), axis=0)

    xyz = XYZOfThetaPhi(thetaphi)

    nt1 = ntheta + 1
    np1 = nphi + 1

    leftedge = np.arange(0, 1 + nt1 * nphi, nt1)
    rightedge = leftedge + ntheta

    SeeNodes = np.arange(npts)
    SeeNodes[rightedge] = leftedge
    SeeNodes[0:nt1] = 0

    UseThese = SeeNodes >= np.arange(npts)
    nreduced = sum(UseThese)

    scrd = xyz[:, UseThese]

    NewNode = np.arange(npts)
    Masters = NewNode[UseThese]
    NewNode[Masters] = np.arange(nreduced)

    top = np.arange(nt1 * nphi, npts)
    OldNodes = np.arange(npts)
    OldNodes[rightedge] = -1
    OldNodes[top] = -1
    NodeOne = np.atleast_2d(OldNodes[OldNodes > -1])

    tcon1 = np.concatenate((NodeOne, NodeOne + 1, NodeOne + nt1 + 1))
    tcon2 = np.concatenate((NodeOne, NodeOne + nt1 + 1, NodeOne + nt1))
    tmpind = np.concatenate((tcon1, tcon2), axis=1)
    tmpcon = NewNode[SeeNodes[tmpind]]

    Eq12 = (tmpcon[1, :] - tmpcon[0, :]) == 0
    Eq13 = (tmpcon[2, :] - tmpcon[0, :]) == 0
    Eq23 = (tmpcon[1, :] - tmpcon[2, :]) == 0

    Degenerate = np.any([Eq12, Eq13, Eq23], axis=0)

    scon = tmpcon[:, np.logical_not(Degenerate)]
    smesh = {'crd': scrd, 'con': scon, 'eqv': []}

    '''
    Need to still add section where L2ip is calculated for the sphere
    and then added to mesh dict:
    Uncomment with all of this is implemented
    if makel2ip:
        smesh['l2ip'] = SphGQRule(smesh, LoadQuadrature(qrule))
    '''

    return smesh


'''
Won't be able to test this part just quite yet
'''


def SphDifferential(mesh, refpts):  # not finished yet need to work on this later
    '''
    SphDifferential - Compute differential of mapping to sphere.
  
      USAGE:

      diff = SphDifferential(mesh, refpts)

      INPUT:

      mesh   is a mesh,
             on a sphere of any dimension
      refpts is d x n, 
             a list of points in the reference element,
             usually the quadrature points, given in barycentric
             coordinates

      OUTPUT:

      diff is d x (d-1) x nq, 
           a list of tangent vectors at each reference
           point for each element; nq is the number of global 
           quadrature points, that is n x ne, where ne is the
           number of elements

    '''

    crd = mesh['crd']
    con = mesh['con']

    dr = refpts.shape
    dc = crd.shape
    d = con.shape

    if dc[0] != d[0]:
        raise ValueError('dimension mismatch: coords and elements')
    if dr[0] != d[0]:
        raise ValueError('dimension mismatch: ref pts and elements')

    dm1 = d - 1

    '''
    
     First compute tangent vectors at intermediate
     mapping to inscribed simplex.  Make a copy for 
     each quadrature point.


    '''
    
    
def SpherePz(x, nargout=1):
    '''
    
    SpherePZ - Generate point, gradients and Hessians of map to sphere.
  
      USAGE:
    
      sk           = SpherePZ(x)
      [sk, gk]     = SpherePZ(x)
      [sk, gk, Hk] = SpherePZ(x)
    
      INPUT:
    
      x is d x 1, 
        a vector with norm <= 1
        
     nargaout is number of outputted arguments and this is needed since python
         does not have it's own equivalent of nargout from matlab and defaults
         to 1 nargout
    
      OUTPUT:
    
      sk is e x 1, 
         a point on the sphere (sqrt(1-x^2), x)
      gk is d x e, 
         the gradients of each component of sk
      Hk is d x d x e, 
         the Hessians of each component of sk

    '''
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = x.T
    d = x.shape[0]
    
    Nx = np.sqrt(1-np.dot(x.T, x))
    
    sk = np.concatenate((Nx,x), axis=0)

    
    if nargout == 1:
        return sk
    
    mNxi = -1/Nx
    gN = mNxi*x

    gk = np.concatenate((gN, np.identity(d)), axis=1)
    
    if nargout == 2:
        return (sk, gk)
        
    dzero = np.zeros((d,d*d))
    lmat = np.identity(d)+np.dot(gN, gN.T)
    Hk = np.concatenate((mNxi*lmat, dzero), axis=1)
    
    Hk = Hk.T.reshape((d+1,d,d)).T
    
    return (sk, gk, Hk)
    
def SphDistanceFunc(x, pts, Sofx, nargout=1):
    '''
    
    SphDistanceFunc - Return half sum of squared distances on sphere.
  
      USAGE:
    
      f           = SphDistanceFunc(x, pts, @Sofx)
      [f, gf]     = SphDistanceFunc(x, pts, @Sofx)
      [f, gf, Hf] = SphDistanceFunc(x, pts, @Sofx)
    
      INPUT:
    
      x    is d x 1, 
           a point in parameter space 
      pts  is (d+1) x n, 
           a list of n points on the sphere
      Sofx is a function handle and is a string, 
           returning parameterization component quantities 
           (function, gradient and Hessian)           
      nargaout is number of outputted arguments and this is needed since python
         does not have it's own equivalent of nargout from matlab and defaults
         to 1 nargout
    
      OUTPUT:
    
      f  is a scalar, 
         the objective function at x
      gf is a vector, 
         the gradient of f at x
      Hf is a matrix, 
         the Hessian of f at x
    
      NOTES:
    
      *  See MisorientationStats
    
    '''
    
    x = np.atleast_2d(x)
    
    if x.shape[0] == 1:
        x = x.T
    
    pts = np.atleast_2d(pts)
    
    if pts.shape[0] == 1:
        pts = pts.T
    d1, n = pts.shape
    d = d1 - 1  # dimension of parameter space
    
    # this part creates a function evaluation handle to behave like matlab's 
    # feval   
    modStr = utl.findModule(Sofx)
    feval = getattr(importlib.import_module(modStr), Sofx)
    
    if nargout == 1:
        s = feval(x, nargout)  # function value only
    elif nargout == 2:
        s, gs = feval(x, nargout)  # gradient now included
    else:
        s, gs, Hs = feval(x, nargout)  # hessian now included
        Hs = Hs.T.reshape((d1, d*d)).T  # more efficient form for later use
    
    # Return function value    
    ctheta = np.minimum(1, np.dot(s.T, pts))
    ctheta = np.atleast_2d(ctheta)
    thetai = np.arccos(ctheta)
    
    f = 0.5*np.dot(thetai,thetai.T)
    
    if nargout == 1:
        return f
        
    # Compute gradient
        
    gc = np.dot(gs, pts)
    
    stheta = np.sin(thetai)
    
    limit = (thetai <= np.finfo(float).eps)  # below machine eps
    nlimit = (thetai > np.finfo(float).eps)  # above machine eps
    
    thfac1 = np.zeros(stheta.shape)
    
    thfac1[nlimit] = thetai[nlimit]/stheta[nlimit]
    thfac1[limit] = 1
    
    gf = -1*np.dot(gc, thfac1.T)
    
    if nargout == 2:
        return (f, gf)
     
    # Compute Hessian
     
    Hc = np.dot(Hs, pts)
    
    limit = (thetai <= np.power(np.finfo(float).eps,1.0/3.0))  # below machine eps
    nlimit = (thetai > np.power(np.finfo(float).eps,1.0/3.0))  # above machine eps
    
    thfac3 = np.zeros(stheta.shape)
    
    thfac3[nlimit] = (stheta[nlimit] - thetai[nlimit]*ctheta[nlimit])/(stheta[nlimit]*stheta[nlimit]*stheta[nlimit])
    thfac3[limit] = 1.0/3.0
    
    gcgct = utl.RankOneMatrix(gc).T.reshape((n, d*d)).T
    
    Hf = np.dot(gcgct, thfac3.T) - np.dot(Hc, thfac1.T)
    
    Hf = Hf.T.reshape(d,d).T
    
    return (f, gf, Hf)
    
def SphereAverage(pts, **kwargs):
    '''
    SphereAverage - Find `average' of list of points on sphere.
  
      USAGE:
    
      avg = SphereAverage(pts)
      [avg, optdat] = SphereAverage(pts, Pzation, nlopts)
    
      INPUT:
    
      pts     is m x n, 
              a list of n points in R^m of unit length
      kwargs:
          x0      is the initial guess,
                  in given parameterization
          nlopts  are options to be passed to the nonlinear minimizer.
    
      OUTPUT:
    
      avg is m x 1, 
             is a unit vector representing the "average" of `pts'
      optdat is a cell array, 
             with three members,  {fval, exitflag, output}
             (see documentation for `fminunc')
    
      NOTES:
    
      *  If only one argument is given, the average returned is 
         the arithmetic average of the points.  If all three arguments
         are given, then the average is computed using unconstrained
         minimization of the sum of squared angles from the data points,
         using the parameterization specified and the options given.
         
      *  See the matlab builtin `fminunc' for details.
    
      *  This routine needs to be fixed.  Currently it uses the
         parameterization given by `SpherePZ' instead of the 
         function handle `PZation'.
    
    '''
    
    pts = np.atleast_2d(pts)
    
    if pts.shape[0] == 1:
        pts = pts.T
        
    avg = utl.UnitVector(np.sum(pts, axis=1)) # all that misorienataionstats uses
    if len(kwargs) == 0:
        return avg
    else:
        wts = kwargs.get('wts', None)
        if wts is not None:
#            wts = np.tile(wts, (m,1))
            avg = utl.UnitVector(np.sum(pts*wts, axis=1))
            return avg
            
    #==============================================================================
    #          x0 = kwargs.get('x0',None)
    #          nlopts = kwargs.get('nlopts',None)
    #          
    #      fun = 'SphDistanceFunc'
    #==============================================================================
     
    '''
     optimization part maybe have it feed in the average from above as an
     initial guess for where the center should be located though the avg values
     should only be the vector values and not the scalar so avg[1:4] and not 
     avg[:]
    '''
