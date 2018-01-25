import numpy as np
import scipy.linalg as scila
# from Misori import Misorientation
import mesh as msh # not yet written
import PolytopeStructure as PStruc  # not yet written
import Utility as utl
from sklearn.preprocessing import normalize

'''
This program was written by Robert Carson on June 10th, 2015.
It is based upon the OdfPf library that the Deformation Processing Lab has written in MATLAB.

The following functions are available in this module:

*Orientation Conversion functions

BungeOfKocks
BungeOfRMat
KocksOfBunge
RMatOfBunge
RMatOfQuat
QuatOfRMat
RodOfQuat
QuatOfRod
QuatOfAngleAxis
OrientConvert

*Quaternion math functions

QuatProd
MeanQuat

*Fundamental region functions

ToFundamentalRegionQ
ToFundamentalRegion

*Crystal symmetry functions

CubSymmetries
HexSymmetries
OrtSymmetries

*Rodrigues space math functions

Misorientations *used to be inside Misori module but had to move here for comp
                 reasons
RodDistance
RodGaussian # needs to be tested but showed work



The following rotations functions are missing or need to be worked on from that matlab library:

CubBaseMesh*
CubPolytope*

HexBaseMesh*
HexPolytope*
OrtBaseMesh*
OrtPolytope*

QuatGradient # quite a bit
QuatReorVel # quite a bit
RodDifferential # quite a bit
RodMetric # quite a bit

*The functions need to have their dependent libraries added

Functions not ported over due to their lack of dependencies/general use:

QuatOfLaueGroup - several of the xtal sym already included
RMatOfGRAINDEXU
RotConvert - OrientConvert replaced it
RodDistance - dependencies no longer included in OdfPf


'''


'''
Various orientation convertion functions in this first section
'''


def BungeOfKocks(kocks=None, units=None):
    '''
        BungeOfKocks - Bunge angles from Kocks angles.
        
        USAGE:
        
        bunge = BungeOfKocks(kocks, units)
        
        INPUT:
        
        kocks is n x 3,
        the Kocks angles for the same orientations
        units is a string,
        either 'degrees' or 'radians'
        
        OUTPUT:
        
        bunge is n x 3,
        the Bunge angles for n orientations
        
        NOTES:
        
        *  The angle units apply to both input and output.
        
    '''
    if kocks is None or kocks.__len__() == 0 or units is None or units.__len__() == 0:
        print('need two arguments: kocks, units')
        raise ValueError('need two arguments: kocks, units')

    if units == 'degrees':
        pi_over_2 = 90
    else:
        pi_over_2 = np.pi/2

    kocks = utl.mat2d_row_order(kocks)

    bunge = np.copy(kocks)
    bunge[0, :] = kocks[0, :]+pi_over_2
    bunge[2, :] = pi_over_2 - kocks[2, :]

    return bunge


def BungeOfRMat(rmat=None, units=None):
    '''
        BungeOfRMat - Bunge angles from rotation matrices.
        
        USAGE:
        
        bunge = BungeOfRMat(rmat, units)
        
        INPUT:
        
        rmat  is 3 x 3 x n,
        an array of rotation matrices
        units is a string,
        either 'degrees' or 'radians' specifying the output
        angle units
        
        OUTPUT:
        
        bunge is n x 3,
        the array of Euler angles using Bunge convention
        
    '''
    if rmat is None or rmat.__len__() == 0 or units is None or units.__len__() == 0:
        print('need two arguments: kocks, units')
        raise ValueError('need two arguments: kocks, units')
    if units == 'degrees':
        indeg = True
    elif units == 'radians':
        indeg = False
    else:
        print('units needs to be either radians or degrees')
        raise ValueError('angle units need to be specified:  ''degrees'' or ''radians''')

    rmat = np.atleast_3d(rmat)

    if rmat.shape[0] != 3:
        rmat = rmat.T

    n = rmat.shape[2]

    c2 = np.copy(rmat[2, 2, :])
    c2 = np.minimum(c2[:], 1.0)
    c2 = np.maximum(c2[:], -1.0)

    myeps = np.sqrt(np.finfo(float).eps)
    near_pole = (np.absolute(c2) > 1-myeps)
    not_near_pole = (np.absolute(c2) < 1-myeps)

    s2 = np.zeros((n))
    c1 = np.zeros((n))
    s1 = np.zeros((n))
    c3 = np.zeros((n))
    s3 = np.zeros((n))

    s2[not_near_pole] = np.sqrt(1 - c2[not_near_pole]*c2[not_near_pole])

    c1[not_near_pole] = -1.0*rmat[1, 2, not_near_pole]/s2[not_near_pole]
    s1[not_near_pole] = rmat[0, 2, not_near_pole]/s2[not_near_pole]
    c3[not_near_pole] = rmat[2, 1, not_near_pole]/s2[not_near_pole]
    s3[not_near_pole] = rmat[2, 0, not_near_pole]/s2[not_near_pole]

    c1[near_pole] = rmat[0, 0, near_pole]
    s1[near_pole] = rmat[1, 0, near_pole]
    c3[near_pole] = 1.0
    s3[near_pole] = 0.0

    bunge = np.array([np.arctan2(s1, c1), np.arccos(c2), np.arctan2(s3, c3)])

    bunge[bunge < 0] = bunge[bunge < 0] + 2*np.pi

    if indeg:
        bunge = bunge*180/np.pi

    return bunge


def KocksOfBunge(bunge=None, units=None):
    '''
    KocksOfBunge - Kocks angles from Bunge angles.

      USAGE:

      kocks = KocksOfBunge(bunge, units)

      INPUT:

      bunge is 3 x n,
            the Bunge angles for n orientations 
      units is a string,
          either 'degrees' or 'radians'

      OUTPUT:

      kocks is 3 x n,
            the Kocks angles for the same orientations

      NOTES:

      *  The angle units apply to both input and output.
    '''

    if bunge is None or units is None:
        print('need two arguments: bunge, units')
        raise ValueError('need two arguments: bunge, units')

    if units == 'degrees':
        indeg = True
    elif units == 'radians':
        indeg = False
    else:
        print('units needs to be either radians or degrees')
        raise ValueError('angle units need to be specified:  ''degrees'' or ''radians''')

    if indeg:
        pi_over_2 = 90
    else:
        pi_over_2 = np.pi/2

    bunge = utl.mat2d_row_order(bunge)


    kocks = bunge.copy()

    kocks[0, :] = bunge[0, :] - pi_over_2
    kocks[2, :] = pi_over_2 - bunge[2, :]

    return kocks


def RMatOfBunge(bunge, units):
    '''
    RMatOfBunge - Rotation matrix from Bunge angles.

      USAGE:

      rmat = RMatOfBunge(bunge, units)

      INPUT:

      bunge is 3 x n,
            the array of Bunge parameters
      units is a string,
            either 'degrees' or 'radians'

      OUTPUT:

      rmat is 3 x 3 x n,
           the corresponding rotation matrices


    '''

    if bunge is None or units is None:
        print('need two arguments: bunge, units')
        raise ValueError('need two arguments: bunge, units')

    if units == 'degrees':
        indeg = True
        bunge = bunge*(np.pi/180)
    elif units == 'radians':
        indeg = False
    else:
        print('units needs to be either radians or degrees')
        raise ValueError('angle units need to be specified:  ''degrees'' or ''radians''')
    
    bunge = utl.mat2d_row_order(bunge)
    n = bunge.shape[1]
    cbun = np.cos(bunge)
    sbun = np.sin(bunge)

    rmat = np.asarray([[cbun[0, :]*cbun[2, :]-sbun[0, :]*cbun[1, :]*sbun[2, :]],
                      [sbun[0, :]*cbun[2, :] + cbun[0, :]*cbun[1, :]*sbun[2, :]],
                      [sbun[1, :]*sbun[2, :]],
                      [-1*cbun[0, :]*sbun[2, :] - sbun[0, :]*cbun[1, :]*cbun[2, :]],
                      [-1*sbun[0, :]*sbun[2, :] + cbun[0, :]*cbun[1, :]*cbun[2, :]],
                      [sbun[1, :]*cbun[2, :]],
                      [sbun[0, :]*sbun[1, :]],
                      [-1*cbun[0, :]*sbun[1, :]],
                      [cbun[1, :]]])

    rmat = rmat.T.reshape((n, 3, 3)).T

    return rmat


def RMatOfQuat(quat):
    '''
    RMatOfQuat - Convert quaternions to rotation matrices.

      USAGE:

      rmat = RMatOfQuat(quat)

      INPUT:

      quat is 4 x n, 
           an array of quaternion parameters

      OUTPUT:

      rmat is 3 x 3 x n, 
           the corresponding array of rotation matrices

      NOTES:

      *  This is not optimized, but still does okay
         (about 6,700/sec on intel-linux ~2GHz)
    '''
    quat = utl.mat2d_row_order(quat)
    n = quat.shape[1]
    rmat = np.zeros((3, 3, n), order='F')

    zeroTol = 1.0e-7  # sqrt(eps) due to acos()

    for i in range(0, n):

        theta = 2*np.arccos(quat[0, i])

        if theta > zeroTol:
            w = theta/np.sin(theta/2)*quat[1:4, i]
        else:
            w = np.asarray([0, 0, 0]).T

        wSkew = [[0, -1*w[2], w[1]], [w[2], 0, -1*w[0]], [-1*w[1], w[0], 0]]
        rmat[:, :, i] = scila.expm(wSkew)

    return rmat


def QuatOfRMat(rmat):
    '''
    QuatOfRMat - Quaternion from rotation matrix

      USAGE:

      quat = QuatOfRMat(rmat)

      INPUT:

      rmat is 3 x 3 x n,
           an array of rotation matrices

      OUTPUT:

      quat is 4 x n,
           the quaternion representation of `rmat'

    '''
    rmat = np.atleast_3d(rmat)
    if rmat.shape[0] != 3:
        rmat = rmat.T
    rsize = rmat.shape

    ca = 0.5*(rmat[0, 0, :]+rmat[1, 1, :]+rmat[2, 2, :]-1)
    ca = np.minimum(ca, 1)
    ca = np.maximum(ca, -1)
    angle = np.squeeze(np.arccos(ca)).T

    '''
    
     Three cases for the angle:  

     *   near zero -- matrix is effectively the identity
     *   near pi   -- binary rotation; need to find axis
     *   neither   -- general case; can use skew part


    '''
    tol = 1.0e-4
    anear0 = angle < tol
    nnear0 = np.sum(anear0)
    angle[anear0] = 0

    raxis = [[rmat[2, 1, :]-rmat[1, 2, :]], [rmat[0, 2, :]-rmat[2, 0, :]], [rmat[1, 0, :]-rmat[0, 1, :]]]
    raxis = utl.mat2d_row_order(np.squeeze(raxis))    
    
    if nnear0 > 0:
        if rsize[2] == 1:
            raxis[:, 0] = 1
        else:
            raxis[:, anear0] = 1

    special = angle > np.pi-tol
    nspec = np.sum(special)

    if nspec > 0:
        angle[special] = np.tile(np.pi, (1, nspec))
        if rsize[2] == 1:
            tmp = np.atleast_3d(rmat[:, :, 0])+np.tile(np.atleast_3d(np.identity(3)), (1, 1, nspec))
        else:
            tmp = rmat[:, :, special]+np.tile(np.atleast_3d(np.identity(3)), (1, 1, nspec))
        tmpr = tmp.T.reshape(3, 3*nspec)
        dp = np.sum(tmpr.conj()*tmpr, axis=0)
        tmpnrm = dp.reshape(3, nspec)
        ind = np.argmax(tmpnrm, axis=0)
        ind = ind + list(range(0, 3*nspec-1, 3))
        saxis = np.atleast_2d(np.squeeze(tmpr[:, ind]))

        if rsize[2] == 1:
            raxis[:, 0] = saxis
        else:
#            print(special.shape)
            if saxis.shape[0] == 1:
                raxis[:, special] = saxis.T
            else:
                raxis[:, special] = saxis
            

    quat = QuatOfAngleAxis(angle, raxis)

    return quat


def RodOfQuat(quat):
    '''    
    RodOfQuat - Rodrigues parameterization from quaternion.

      USAGE:

      rod = RodOfQuat(quat)

      INPUT:

      quat is 4 x n, 
           an array whose columns are quaternion paramters; 
           it is assumed that there are no binary rotations 
           (through 180 degrees) represented in the input list

      OUTPUT:

     rod is 3 x n, 
         an array whose columns form the Rodrigues parameterization 
         of the same rotations as quat

    '''

    rod = quat[1:4, :]/np.tile(quat[0, :], (3, 1))

    return rod


def QuatOfRod(rod):
    '''
    QuatOfRod - Quaternion from Rodrigues vectors.

      USAGE:

      quat = QuatOfRod(rod)

      INPUT:

      rod  is 3 x n, 
           an array whose columns are Rodrigues parameters

      OUTPUT:

      quat is 4 x n, 
           an array whose columns are the corresponding unit
           quaternion parameters; the first component of 
           `quat' is nonnegative

    '''

    rod =  utl.mat2d_row_order(rod)   
    
    cphiby2 = np.cos(np.arctan(np.sqrt(np.sum(rod.conj()*rod, axis=0))))

    quat = np.asarray([[cphiby2], np.tile(cphiby2, (3, 1))*rod])

    quat = np.concatenate(quat, axis=0)

    return quat

'''
Various Quaternion functions thourh still missing: QuatReorVel, QuatGradient going to leave Laue group out
'''

def QuatProd(q2, q1):
    '''
    QuatProd - Product of two unit quaternions.

      USAGE:

       qp = QuatProd(q2, q1)

      INPUT:

       q2, q1 are 4 x n, 
              arrays whose columns are quaternion parameters

      OUTPUT:

       qp is 4 x n, 
          the array whose columns are the quaternion parameters of 
          the product; the first component of qp is nonnegative

       NOTES:

       *  If R(q) is the rotation corresponding to the
          quaternion parameters q, then 

          R(qp) = R(q2) R(q1)


    '''

    a = np.atleast_2d(q2[0, :])
    a3 = np.tile(a, (3, 1))
    b = np.atleast_2d(q1[0, :])
    b3 = np.tile(b, (3, 1))

    avec = np.atleast_2d(q2[1:4, :])
    bvec = np.atleast_2d(q1[1:4, :])

    qp1 = np.atleast_2d(a*b - np.sum(avec.conj()*bvec, axis=0))
    if q1.shape[1] == 1:
        qp2 = np.atleast_2d(np.squeeze(a3*bvec + b3*avec + np.cross(avec.T, bvec.T).T)).T
    else:
        qp2 = np.atleast_2d(np.squeeze(a3*bvec + b3*avec + np.cross(avec.T, bvec.T).T))

    qp = np.concatenate((qp1, qp2), axis=0)

    q1neg = np.nonzero(qp[0, :] < 0)

    qp[:, q1neg] = -1*qp[:, q1neg]

    return qp

def QuatMean(quats):
    '''
    QuatMean finds the average quaternion based upon the methodology defined in
    Quaternion Averaging by Markley, Cheng, Crassidis, and Oshman
    
    Input:
        quats - A list of quaternions of that we want to find the average quaternion
    Output:
        mquats - the mean quaternion of the system
    '''
    if(quats.shape[0] == 4):
        n = quats.shape[1]
        mmat = 1/n*quats.dot(quats.T)
    else:
        n = quats.shape[0]
        mmat = 1/n*quats.T.dot(quats)
    bmmat = mmat - np.eye(4)
    
    eig, eigvec = np.linalg(bmmat)
    mquats = np.squeeze(eigvec[:, np.argmax(eig)])
    
    return mquats
    

def QuatOfAngleAxis(angle, raxis):
    '''
    QuatOfAngleAxis - Quaternion of angle/axis pair.

      USAGE:

      quat = QuatOfAngleAxis(angle, axis)

      INPUT:

      angle is an n-vector, 
            the list of rotation angles
      axis is 3 x n, 
            the list of rotation axes, which need not
            be normalized (e.g. [1 1 1]'), but must be nonzero

      OUTPUT:

      quat is 4 x n, 
           the quaternion representations of the given
           rotations.  The first component of quat is nonnegative.
   '''

    halfAngle = 0.5*angle.T
    cphiby2 = np.atleast_2d(np.cos(halfAngle))
    sphiby2 = np.sin(halfAngle)
    rescale = sphiby2/np.sqrt(np.sum(raxis.conj()*raxis, axis=0))
    scaledAxis = np.tile(rescale, (3, 1))*raxis
    quat = np.concatenate((cphiby2, scaledAxis), axis=0)
    q1neg = np.nonzero(quat[0, :] < 0)
    quat[:, q1neg] = -1*quat[:, q1neg]

    return quat

def AngleAxisOfRod(rod):
    '''
    Takes in a Rodrigues Vector and returns the angle axis pair
    '''
    
    rod =  utl.mat2d_row_order(rod)   
    
    angle = 2*np.arctan(np.linalg.norm(rod, axis=0))

    ang_axis = angle*normalize(rod, axis=0)
    
    return ang_axis
    
    

'''
Universal convertion function
'''


def OrientConvert(inOri, inConv, outConv, inDeg=None, outDeg=None):
    '''
     OrientConvert - Convert among orientation conventions.

      STATUS:  in development

      USAGE:

      out = OrientConvert(in, inConv, outConv)
      out = OrientConvert(in, inConv, outConv, inDeg, outDeg)

      INPUT:

      in      is d x n 
              input parameters (e.g. Euler angles)
      inConv  is a string
              input convention
      outConv is a string
              output convention
      inDeg   is a string
              either 'degrees' or 'radians'
      outDeg  is a string
              either 'degrees' or 'radians'

      OUTPUT:

      out is e x n 
             output parameters
      NOTES:

      * Conventions are 'kocks', 'bunge', 'rmat', 'quat', 'rod'
      * If any Euler angle conventions are specified, then the
        degrees convention must also be specified



     Convert input to rotation matrices.

    '''

    if inConv == 'kocks':
        rmat = RMatOfBunge(BungeOfKocks(inOri, inDeg), inDeg)
    elif inConv == 'bunge':
        rmat = RMatOfBunge(inOri, inDeg)
    elif inConv == 'rmat':
        rmat = inOri
    elif inConv == 'rod' or inConv == 'rodrigues':
        rmat = RMatOfQuat(QuatOfRod(inOri))
    elif inConv == 'quat' or inConv == 'quaternion':
        rmat = RMatOfQuat(inOri)
    else:
        print('input convention not matched')
        raise ValueError('input convention not matched')

    if outConv == 'kocks':
        out = KocksOfBunge(BungeOfRMat(rmat, outDeg), outDeg)
    elif outConv == 'bunge':
        out = BungeOfRMat(rmat, outDeg)
    elif outConv == 'rmat':
        out = rmat
    elif outConv == 'rod' or outConv == 'rodrigues':
        out = RodOfQuat(QuatOfRMat(rmat))
    elif outConv == 'quat' or outConv == 'quaternion':
        out = QuatOfRMat(rmat)
    else:
        print('output convention not matched')
        raise ValueError('output convention not matched')

    return np.require(out, requirements=['F'])

'''
Functions used to bring orientations back into the fundamental region
The OrientConvert function can be used to bring them back into any other space other than Rod or Quat space
'''


def ToFundamentalRegionQ(quat, qsym):
    '''
    ToFundamentalRegionQ - To quaternion fundamental region.

      USAGE:

      q = ToFundamentalRegionQ(quat, qsym)

      INPUT:

      quat is 4 x n, 
           an array of n quaternions
      qsym is 4 x m, 
           an array of m quaternions representing the symmetry group

      OUTPUT:

      q is 4 x n, the array of quaternions lying in the
                  fundamental region for the symmetry group 
                  in question

      NOTES:  

      *  This routine is very memory intensive since it 
         applies all symmetries to each input quaternion.


    '''
    quat = utl.mat2d_row_order(quat)    
    qsym = utl.mat2d_row_order(qsym)
    n = quat.shape[1]
    m = qsym.shape[1]

    qMat = np.tile(quat, (m, 1))

    qSymMat = np.tile(qsym, (1, n))

    qeqv = QuatProd(qMat.T.reshape(m*n, 4).T, qSymMat)

    q0_abs = np.abs(np.atleast_2d(qeqv[0, :]).T.reshape(n, m)).T

    imax = np.argmax(q0_abs, axis=0)

    ind = np.arange(n)*m + imax

    q = qeqv[:, ind]

    return q


def ToFundamentalRegion(quat, qsym):
    '''
    ToFundamentalRegion - Put rotation in fundamental region.

      USAGE:

      rod = ToFundamentalRegion(quat, qsym)

      INPUT:

      quat is 4 x n, 
           an array of n quaternions
      qsym is 4 x m, 
           an array of m quaternions representing the symmetry group

      OUTPUT:

      rod is 3 x n, 
          the array of Rodrigues vectors lying in the fundamental 
          region for the symmetry group in question

      NOTES:  

      *  This routine is very memory intensive since it 
         applies all symmetries to each input quaternion.


    '''

    q = ToFundamentalRegionQ(quat, qsym)
    rod = RodOfQuat(q)

    return rod

'''
The cubic, hexagonal, and orthorhombic symmetry groups for rotations necessary to form a fundamental region around the origin in rodrigues space, and they are given here as quaternions.

If any other symmetries groups are desired they would need to be programmed in

'''


def CubSymmetries():
    ''' CubSymmetries - Return quaternions for cubic symmetry group.

       USAGE:

       csym = CubSymmetries

       INPUT:  none

       OUTPUT:

       csym is 4 x 24, 
            quaternions for the cubic symmetry group
    '''

    '''
        array index 1 = identity
        array index 2-4 = fourfold about x1
        array index 5-7 = fourfold about x2
        array index 8-9 = fourfold about x9
        array index 10-11 = threefold about 111
        array index 12-13 = threefold about 111
        array index 14-15 = threefold about 111
        array index 16-17 = threefold about 111
        array index 18-24 = twofold about 110
    
    '''
    angleAxis = [
        [0.0, 1, 1, 1],
        [np.pi*0.5, 1, 0, 0],
        [np.pi, 1, 0, 0],
        [np.pi*1.5, 1, 0, 0],
        [np.pi*0.5, 0, 1, 0],
        [np.pi, 0, 1, 0],
        [np.pi*1.5, 0, 1, 0],
        [np.pi*0.5, 0, 0, 1],
        [np.pi, 0, 0, 1],
        [np.pi*1.5, 0, 0, 1],
        [np.pi*2/3, 1, 1, 1],
        [np.pi*4/3, 1, 1, 1],
        [np.pi*2/3, -1, 1, 1],
        [np.pi*4/3, -1, 1, 1],
        [np.pi*2/3, 1, -1, 1],
        [np.pi*4/3, 1, -1, 1],
        [np.pi*2/3, -1, -1, 1],
        [np.pi*4/3, -1, -1, 1],
        [np.pi, 1, 1, 0],
        [np.pi, -1, 1, 0],
        [np.pi, 1, 0, 1],
        [np.pi, 1, 0, -1],
        [np.pi, 0, 1, 1],
        [np.pi, 0, 1, -1]]
    #
    angleAxis = np.asarray(angleAxis).transpose()
    angle = angleAxis[0, :]
    axis = angleAxis[1:4, :]
    #
    #  Axis does not need to be normalized it is done
    #  in call to QuatOfAngleAxis.
    #
    csym = QuatOfAngleAxis(angle, axis)

    return csym


def HexSymmetries():
    '''
    HexSymmetries - Quaternions for hexagonal symmetry group.

      USAGE:

      hsym = HexSymmetries

      INPUT:  none

      OUTPUT:

      hsym is 4 x 12,
           it is the hexagonal symmetry group represented
           as quaternions


    '''
    p3 = np.pi/3
    p6 = np.pi/6
    ci = np.atleast_2d(np.cos(p6*(np.arange(6))))
    si = np.atleast_2d(np.sin(p6*(np.arange(6))))
    z6 = np.zeros((1, 6))
    w6 = np.ones((1, 6))
    pi6 = np.tile(np.pi, [1, 6])
    p3m = np.atleast_2d(p3*(np.arange(6)))

    sixfold = np.concatenate((p3m, z6, z6, w6))
    twofold = np.concatenate((pi6, ci, si, z6))

    angleAxis = np.asarray(np.concatenate((sixfold, twofold), axis=1))
    angle = angleAxis[0, :]
    axis = angleAxis[1:4, :]
    #
    #  Axis does not need to be normalized it is done
    #  in call to QuatOfAngleAxis.
    #
    hsym = QuatOfAngleAxis(angle, axis)

    return hsym


def OrtSymmetries():
    '''
    OrtSymmetries - Orthorhombic symmetry group.

      USAGE:

      osym = OrtSymmetries

      INPUT:  none

      OUTPUT:

      osym is 4 x 4, 
           the quaternions for the symmetry group


    '''
    angleAxis = [
        [0.0, 1, 1, 1],
        [np.pi, 1, 0, 0],
        [np.pi, 0, 1, 0],
        [np.pi, 0, 0, 1]]

    angleAxis = np.asarray(angleAxis).transpose()
    angle = angleAxis[0, :]
    axis = angleAxis[1:4, :]
    #
    #  Axis does not need to be normalized it is done
    #  in call to QuatOfAngleAxis.
    #
    osym = QuatOfAngleAxis(angle, axis)

    return osym

'''
Unfinished code that don't yet have their dependencies programmed in yet
'''


def CubBaseMesh():
    '''
        
        CubBaseMesh - Return base mesh for cubic symmetries
        
        USAGE:
        
        m = CubBaseMesh
        
        INPUT:  no inputs
        
        OUTPUT:
        
        m is a MeshStructure,
        on the cubic fundamental region
        
        
        '''
    m = msh.LoadMesh('cub-base')
    m['symmetries'] = CubSymmetries()

    return m

def HexBaseMesh():
    '''
        
        HexBaseMesh - Return base mesh for hexagonal symmetries
        
        USAGE:
        
        m = HexBaseMesh
        
        INPUT:  no inputs
        
        OUTPUT:
        
        m is a MeshStructure,
        on the hexagonal fundamental region
        
        
        '''
    m = msh.LoadMesh('hex-base')
    m['symmetries'] = HexSymmetries()

    return m

def OrtBaseMesh():
    '''
        
        OrtBaseMesh - Return base mesh for orthorhombic symmetries
        
        USAGE:
        
        m = OrtBaseMesh
        
        INPUT:  no inputs
        
        OUTPUT:
        
        m is a MeshStructure,
        on the orthorhombic fundamental region
        
        
        '''
    m = msh.LoadMesh('ort-base')
    m['symmetries'] = OrtSymmetries()

    return m

def CubPolytope():

    '''
        CubPolytope - Polytope for cubic fundamental region.
        
        USAGE:
        
        cubp = CubPolytope
        
        INPUT:  none
        
        OUTPUT:
        
        cubp is a PolytopeStructure:
        it gives the polytope for the cubic
        fundamental region including the vertex
        list and the faces component (for plotting)
        
        
        
        Compute faces (constraints).
        '''

    b1 = np.tan(np.pi/8)
    b2 = np.tan(np.pi/6)

    utmp = np.array([[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]]).T

    n111 = utl.UnitVector(utmp)
    matrix = np.concatenate((np.identity(3), n111.T), axis=0)
    matrix = np.concatenate((matrix, -1.0*matrix), axis=0)

    pass

'''
Different Rodrigues functions
'''


def RodDistance(pt, ptlist, sym):
    '''
    RodDistance - Find angular distance between rotations.
  
      USAGE:

      dist = RodDistance(pt, ptlist, sym)

      INPUT:

      pt     is 3 x 1, 
             a point given in Rodrigues parameters
      ptlist is 3 x n, 
             a list of points, also Rodrigues 
      sym    is 4 x m, 
             the symmetry group in quaternions

      OUTPUT:

      dist   is 1 x n, 
             the distance between `pt' and each point in `ptlist'


    '''
    pt = utl.mat2d_row_order(pt)
    ptlist = utl.mat2d_row_order(ptlist)

    q1 = QuatOfRod(pt)
    q2 = QuatOfRod(ptlist)

    dist, mis = Misorientation(q1, q2, sym)

    return dist


def Misorientation(q1, q2, sym):
    '''
    Misorientation - Return misorientation data for quaternions.

      USAGE:

      angle = Misorientation(q1, q2, sym)
      [angle, mis] = Misorientation(q1, q2, sym)

      INPUT:

      q1 is 4 x n1, 
         is either a single quaternion or a list of n quaternions
      q2 is 4 x n,  
         a list of quaternions

      OUTPUT:

      angle is 1 x n, 
            the list of misorientation angles between q2 and q1
      mis   is 4 x n, (optional) 
            is a list of misorientations in the fundamental region 
            (there are many equivalent choices)

      NOTES:

      *  The misorientation is the linear tranformation which
         takes the crystal basis given by q1 to that given by
         q2.  The matrix of this transformation is the same
         in either crystal basis, and that is what is returned
         (as a quaternion).  The result is inverse(q1) * q2.
         In the sample reference frame, the result would be
         q2 * inverse(q1).  With symmetries, the result is put
         in the fundamental region, but not into the Mackenzie cell.


    '''
    q1 = utl.mat2d_row_order(q1)
    q2 = utl.mat2d_row_order(q2)

    f1 = q1.shape
    f2 = q2.shape

    if f1[1] == 1:
        q1 = np.tile(q1, (1, f2[1]))

    q1i = np.concatenate((np.atleast_2d(-1*q1[0, :]), np.atleast_2d(q1[1:4, :])), axis=0)

    mis = ToFundamentalRegionQ(QuatProd(q1i, q2), sym)

    angle = 2*np.arccos(np.minimum(1, mis[0, :]))

    return (angle, mis)


def RodGaussian(cen, pts, stdev, sym):
    '''
    RODGAUSSIAN - Gaussian distribution on angular distance.

      USAGE:

      gauss = RodGaussian(cen, pts, stdev, sym)

      INPUT:

      cen   is 3 x 1, 
            the center of the distribution (in Rodrigues parameters)
      pts   is 3 x n, 
            a list of points (Rodrigues parameters)
      stdev is 1 x 1, 
            the standard deviation of the distribution
      sym   is 4 x k, 
            the symmetry group (quaternions)

      OUTPUT:

      gauss is 1 x n, 
            the list of values at each input point

      NOTES:

      *  This returns the values of a (not normalized) 1D Gaussian 
         applied to angular distance from the center point 
      *  The result is not normalized to have unit integral.


    '''
    twosigsq = 2*(stdev**2)
    theta = RodDistance(cen, pts, sym)
    minusthetasq = -1*theta*theta
    gauss = np.exp(minusthetasq, twosigsq)

    return gauss
