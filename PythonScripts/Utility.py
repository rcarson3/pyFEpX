import numpy as np
import scipy as sci
import glob
import re
import os
import importlib

'''
Various different utility functions have been implemented from the Deformation Processing Lab OdfPf 
library plus an additional function to help evaluate unknown functions

The following functions are available in this module:

findModule

AggregrateFunction
MetricGij
RankOneMatrix
UniqueVectors
UnitVector

Other functions available in there are not rewritten here because of the fact they are not implemented
in any of the other codes. They could be added if the need for them exists.
'''


def findModule(fhandle, fileDir=None):
    '''
    When trying to evaluate a function and the module is not known where it's located this function
    will search the current active directory to see if it can find the module in that directory.
    The directory can also be specified if one is using this package out side of the current directory.
    The module is then returned in which that function is located. It will also only work with Unix and 
    Windows machines, since those have well defined path directories

    Input: fhandle - a string of the function handle being examined
           (optional) fileDir - a string of the directory to look in
    Output: modS - a string of the module that the function is located in

    If the function can not be found then a value error is thrown
    '''

    if fileDir is None:

        fileDir = os.getcwd()
        sysName = os.name

        if sysName == 'nt':  # Windows operating systems 
            dSep = '\\'
        else:  # Unix operating systems
            dSep = '/'
        fileDir += dSep

    fPath = fileDir + '*.py'
    files = glob.glob(fPath)  # Get all of the python files in the current directory
    phrase = 'def ' + fhandle  # Looks for the function handle in the file

    count = 0

    for f in files:  # Search all of the files
        with open(f) as fpy:
            data = fpy.readlines()
            for line in data:  # Search line by line of the file opened
                flag = re.search(phrase, line)
                if flag:
                    count += 1
                    break

        if count > 0:
            outS = re.sub(r'{}.*?'.format(re.escape(fileDir)), '', f)
            modS = re.sub(r'{}.*?'.format(re.escape('.py')), '', outS)
            break

        count = 0

    if count == 0:
        print('Function could not be found in the current working directory')
        raise ValueError('Function not in current directory')

    return modS


def AggregrateFunction(pts, agg, wts, pointFun, varargin):
    '''
    AggregateFunction - Create a function from an aggregate of points.

      USAGE:

      aggf = AggregateFunction(pts, agg, wts, PointFun)
      aggf = AggregateFunction(pts, agg, wts, PointFun, args)

      INPUT:

      pts is d x n, 
          the set of points on which to evaluate the aggregate function
      agg is d x m, 
          a collection of points (the aggregate)
      wts is 1 x m, 
          the weighting associated with points in `agg'
      pointFun is a function handle string,
          the function which evaluates the distribution associated 
          with each point of the aggregate;
          the required interface to PointFun is:

          PointFun(center, points [, args])
                   center is d x 1, the center of the distribution
                   points is d x n, a list of points to evaluate
                   args are function-specific arguments

      Remaining arguments are passed to PointFun and should be inputed as a
      dictionary whose keys are the same name as the functions input name

      OUTPUT:

      aggf is 1 x n, 
           the values of the aggregate function at each point in `pts';

      NOTES:

      * Each point in the aggregate is the center of a distribution
        over the whole space, given by PointFun; all of these 
        distributions are superposed to give the resulting 
        aggregate function, which is then evaluated at the 
        specified point.


    '''

    keys = varargin.keys()
    pts = mat2d_row_order(pts)
    agg = mat2d_row_order(agg)
    wts = np.atleast_1d(wts)
    n = pts.shape
    m = agg.shape
    wtscheck = len(wts)

    if m[1] != wtscheck:
        print('dimension mismatch: wts and agg (length)')
        raise ValueError('dimension mismatch between wts and agg (length)')

    if n[0] != m[0]:
        print('dimension mismatch: pts and agg (first dimension)')
        raise ValueError('dimension mismatch between pts and agg (1st dim)')

    aggf = np.zeros((1, n[1]))

    modStr = findModule(pointFun)
    feval = getattr(importlib.import_module(modStr), pointFun)

    for i in range(m[1]):
        aggf = aggf + wts[i] * feval(agg[:, i], pts, **varargin)

    return aggf


def MetricGij(diff):
    '''
    MetricGij - Compute components of metric from differential.

      USAGE:

      gij = MetricGij(diff)

      INPUT:

      diff is m x n x l,
           the array of n tangent vectors of dimension m at each 
           of l points

      OUTPUT:

      gij is n x n x l, 
          the metric components (dot(ti, tj)) at each of the l points

    '''

    m = diff.shape
    gij = np.zeros((m[0], m[0], m[2]))

    for i in range(m[2]):
        gij[:, :, i] = np.dot(np.transpose(diff[:, :, i], (1, 0, 2)), diff[:, :, i])

    return gij


def RankOneMatrix(vec1, *args):
    '''
    RankOneMatrix - Create rank one matrices (dyadics) from vectors. It therefore simply computes the 
    outer product between two vectors, $v_j \otimes v_i$

      USAGE:

      r1mat = RankOneMatrix(vec1)
      r1mat = RankOneMatrix(vec1, vec2)

      INPUT:

      vec1 is m1 x n, 
           an array of n m1-vectors 
      vec2 is m2 x n, (optional) 
           an array of n m2-vectors

      OUTPUT:

      r1mat is m1 x m2 x n, 
            an array of rank one matrices formed as c1*c2' 
            from columns c1 and c2

      With one argument, the second vector is taken to
      the same as the first.

      NOTES:

      *  This routine can be replaced by MultMatArray.


    '''

    vec1 = mat2d_row_order(vec1)

    if len(args) == 0:
        vec2 = vec1.copy()
    else:
        vec2 = np.atleast_2d(args[0])

    m = vec1.shape
    n = vec2.shape[0]

    if m[0] != n:
        print('dimension mismatch: vec1 and vec2 (first dimension)')
        raise ValueError('dimension mismatch between vec1 and vec2 (1st dim)')

    rrom = np.zeros((m[0], n, m[1]))

    for i in range(m[1]):
        rrom[:, :, i] = np.outer(vec1[:, i], vec2[:, i])

    return rrom


def UniqueVectors(vec, *args):
    '''
    UniqueVectors - Remove near duplicates from a list of vectors.
  
      USAGE:

      [uvec, ord, iord] = UniqueVectors(vec)
      [uvec, ord, iord] = UniqueVectors(vec, tol)

      INPUT:

      vec is d x n, 
          an array of n d-vectors
      tol is a scalar, (optional) 
          the tolerance for comparison; it defaults to 1.0e-14

      OUTPUT:

      uvec is d x m, 
           the set of unique vectors; two adjacent vectors are considered
           equal if each component is within the given tolerance
      ord  is an m-vector, (integer)
           which relates the input vector to the output vector, 
           i.e. uvec = vec(:, ord)
      iord is an n-vector, (integer)
           which relates the reduced vector to the original vector, 
           i.e. vec = uvec(:, iord)

      NOTES:

      *  After sorting, only adjacent entries are tested for equality
         within the tolerance.  For example, if x1 and x2 are within
         the tolerance, and x2 and x3 are within the tolerance, then 
         all 3 will be considered the same point, even though x1 and
         x3 may not be within the tolerance.  Consequently, if you
         make the tolerance too large, all the points will be
         considered the same.  Nevertheless, this routine should be 
         adequate for the its intended application (meshing), where
         the points fall into well-separated clusters.


    '''

    vec = mat2d_row_order(vec)

    if len(args) == 0:
        tol = 1.0e-14
    else:
        tol = args[0]

    d = vec.shape
    n = d[1]

    ivec = np.zeros((d[0], d[1]))

    for row in range(d[0]):
        tmpsrt = np.sort(vec[row, :])
        tmpord = np.argsort(vec[row, :])

        tmpcmp = np.abs(tmpsrt[1:n] - tmpsrt[0:n - 1])

        indep = np.hstack((True, tmpcmp > tol))

        rowint = np.cumsum(indep)

        ivec[row, tmpord] = rowint

    utmp, orde, iord = np.unique(ivec.T, return_index=True, return_inverse=True)

    orde = np.int_(np.floor(orde / 3))

    uvec = vec[:, orde]
    orde = orde.T
    iord = iord.T

    return (uvec, orde, iord)


def UnitVector(vec, *args):
    '''
    UnitVector - Normalize an array of vectors.

      USAGE:

      uvec = UnitVector(vec)
      uvec = UnitVector(vec, ipmat)

      INPUT:

      vec   is m x n, 
            an array of n nonzero vectors of dimension m
      ipmat is m x m, (optional)
            this is a (SPD) matrix which defines the inner product
            on the vectors by the rule:  
               norm(v)^2 = v' * ipmat * v

            If `ipmat' is not specified, the usual Euclidean 
            inner product is used.

      OUTPUT:

      uvec is m x n,
           the array of unit vectors derived from `vec'


    '''

    vec = mat2d_row_order(vec)

    m = vec.shape[0]

    if len(args) > 0:
        ipmat = args[0]
        nrm2 = np.sum(vec.conj() * np.dot(ipmat, vec), axis=0)
    else:
        nrm2 = np.sum(vec.conj() * vec, axis=0)

    nrm = np.tile(np.sqrt(nrm2), (m, 1))
    uvec = vec / nrm

    return uvec
    
def mat2d_row_order(mat):
    '''
    It takes in a mat nxm or a vec of n length and returns a 2d numpy array
    that is nxm where m is a least 1 instead of mxn where m is 1 like the 
    numpy.atleast_2d() will do if a vec is entered
    
    Input: mat - a numpy vector or matrix with dimensions of n or nxm
    output: mat - a numpy matrix with dimensions nxm where m is at least 1    
    
    '''
    
    mat = np.atleast_2d(mat) 
    if mat.shape[0] == 1:
        mat = mat.T
        
    return mat
    
