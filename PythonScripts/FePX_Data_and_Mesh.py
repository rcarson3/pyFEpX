import numpy as np
import textadapter as ta
#import iopro
import Utility as util

'''
List of functions available:
readMesh(fileLoc, fileName)
mesh_node_neigh(conn, nd_conn)
mesh_node_conn(conn, nnode)
wordParser(listVals)
readData(fileLoc, nProc, frames=None, fepxData=None, restart=False)
readGrainData(fileLoc, grainNum, frames=None, grData=None)
readLOFEMData(fileLoc, nProc, nqpts=15, frames=None, lofemData=None)
fixStrain(epsVec)
findComments(fileLoc)
selectFrameTxt(fileLoc, frames, comments='%')
'''

def readMesh(fileLoc, fileName):
    ''' 
        Takes in the file location and file name and it then generates a dictionary structure from those files for the mesh.
        Input: fileLoc = a string of the loaction of file on your computer
               fileName = a string of the name of the file assuming they are all equal for .mesh, .kocks, and .grain
        Outpute: mesh = a dictionary that contains the following fields in it:
            name = file location
            eqv = any equivalence nodes currently this is an empty nest
            grains = what grain each element corresponds to
            con = connectivity of the mesh for each element
            crd = coordinates of each node
            surfaceNodes = surface nodes of the mesh
            kocks = kocks angles for each grain
            phases = phase number of each element
    '''
    surfaceNodes = []
    con = []
    crd = []
    eqv = []
    name = fileLoc
    meshLoc = fileLoc + fileName + '.mesh'
    grainLoc = fileLoc + fileName + '.grain'
    kockLoc = fileLoc + fileName + '.kocks'
    grains = []
    phases = []
    kocks = []
    mesh = {}
    mesh['name'] = name
    mesh['eqv'] = []

    with open(meshLoc) as f:
        #        data = f.readlines()
        for line in f:
            words = line.split()
            #            print(words)
            lenWords = len(words)
            if not words:
                continue
            if lenWords == 4:
                nums = wordParser(words)
                crd.append(nums[1:4])
            if lenWords == 7:
                nums = wordParser(words)
                surfaceNodes.append(nums[0:7])
            if lenWords == 11:
                nums = wordParser(words)
                con.append(nums[1:11])

    grains = np.genfromtxt(grainLoc, usecols=(0), skip_header=1, skip_footer=0)
    phases = np.genfromtxt(grainLoc, usecols=(1), skip_header=1, skip_footer=0)
    kocks = np.genfromtxt(kockLoc, usecols=(0, 1, 2), skip_header=2, skip_footer=1)
    mesh['con'] = np.require(np.asarray(con, order='F', dtype=np.int32).transpose(), requirements=['F'])
    mesh['crd'] = np.require(np.asarray(crd, order='F').transpose(), requirements=['F'])
    mesh['surfaceNodes'] = np.require(np.asarray(surfaceNodes, order='F',dtype=np.int32).transpose(), requirements=['F'])
    mesh['grains'] = np.asfortranarray(grains.transpose(), dtype=np.int32)
    mesh['kocks'] = util.mat2d_row_order(np.asfortranarray(kocks.transpose()))
    mesh['phases'] = np.asfortranarray(phases.transpose(),dtype=np.int8)

    return mesh

def mesh_node_neigh(conn, nd_conn):
    '''
        Creates a list of all of a nodes neighbors given the connectivity
        array and the node elem connectivity array. 
        
        Input: conn = a numpy array of the mesh connectivity array
               nd_conn = a list of sets for each node and what elems they
                       are connected to
               
        Output: nd_neigh = a numpy list of sets of a nodes neighbors
        
        Note: This should work but it still needs a slightly more extensive testing
    '''

    nnode = len(nd_conn)
    nd_neigh = [set() for _ in range(nnode)]
    ncrds = conn.shape[0]

    #There's got to be a faster way to do this...    
    for i in range(nnode):
        for j in nd_conn[i]:
            for k in range(ncrds):
                tmp = conn[k,j]
                nd_neigh[i].add(tmp)
        #Get rid of your own node...        
        nd_neigh[i].discard(i)
        
    return nd_neigh
    
def mesh_node_conn(conn, nnode):
    '''
        Takes in the element connectivity array and computes the inverse
        array or the nodal connectivity array.
        
        Input: conn = a numpy array of the mesh element connectivity array
               nnode = the number of nodes in the mesh
               
        Output: nd_conn = a list of sets for each node and what elems they
                        are connected to
                        
        Note: This should work but it still needs more extensive testing
    '''
    
    nd_conn = [set() for _ in range(nnode)]
    
    ncrds, nelems = conn.shape
    
    for i in range(nelems):
        for j in range(ncrds):
            tmp = conn[j,i]
            nd_conn[tmp].add(i)
    
    return nd_conn

def grain_conn_mesh(ndconn, conn, grains, nnode):
    '''
    Takes in the nodal and elemental connectivity arrays. It then goes through
    the list of nodes and increments the node count for those in different grains
    in the elemental connectivity array. It also will update all of the other
    nodes by the most current incremental count.
    
    Input: ndconn = a list of sets for each node and what elems they are
                connected to
           conn = a numpy array of the mesh element connectivity array
           grains = a numpy array corresponding to what grain each element is
                in
           nnode = the number of nodes in the mesh
    Output: conn = a numpy array that is the updated connectivity array
    '''
    #We don't increment anything to start off with
    incr = 0
    nodes = np.zeros(10, dtype='int32')
    #Really wish I didn't have to make a copy of this...
    conn_orig = np.copy(conn)
    
    for i in range(nnode):
        #We want a numpy array of all the elements connected to that node
        ndelems =  np.array(list(ndconn[i]), dtype='int32')
        #We also want to know how many unique grains we actually have
        ugrns = np.unique(grains[ndelems])
        #Our inner loop that were going to use to go through the data
        for j in ndelems:
            #First we get all the nodes
            nodes[:] = conn_orig[:, j]
            #Then we simply get the index of our node
            ind = nodes == i
            #Finally we increment the conn array
            conn[ind, j] = incr + i + np.where(ugrns == grains[j])[0][0]
                  
        #We don't need to increment anything if there is only one grain for
        #that node
        nincr = ugrns.shape[0] - 1
        incr = incr + nincr

    return conn



def wordParser(listVals):
    '''
        Read in the string list and parse it into a floating list
        Input: listVals = a list of strings
        Output: numList = a list of floats
    '''
    numList = []
    for str in listVals:
        num = float(str)
        numList.append(num)

    return numList


def readData(fileLoc, nProc, frames=None, fepxData=None, restart=False):
    '''
        Reads in the data files that you are interested in across all the processors
        and only for the frames that you are interested in as well
        Input: fileLoc = a string of the file location
               nProc = an integer of the number of processors used in the simulation
               frames = what frames you are interested in, default value is all of them
               fepxData = what data files you want to look at, default value is:
                    .ang, .strain, .stress, .adx, .advel, .dpeff, .eqplstrain, .crss
        Output: data = a dictionary that contains a list/ndarray of all read in data files. If other files other than default are wanted than the keys for those
            values will be the file location. The default files have the following key
            values:
            coord_0: a float array of original coordinates
            hard_0: a float array of original crss_0/g_0 for each element
            angs_0: a float array of original kocks angles for each element
            vel_0: a float array of original velocity at each node
            coord: a float array of deformed coordinates
            hard: a float array of crss/g for each element
            angs: a float array of evolved kocks angles for each element
            stress: a float array of the crystal stress for each element
            strain: a float array of the sample strain for each element
            pldefrate: a float of the plastic deformation rate for each element
            plstrain: a float of the plastic strain for each element
            vel: a float array of the velocity at each node
    '''
    flDflt = False
    frDflt = False
    data = {}
    proc = np.arange(nProc)

    if fepxData is None:
        fepxData = ['ang', 'strain', 'stress', 'adx', 'advel', 'dpeff', 'eqplstrain', 'crss']
        flDflt = True
    if frames is None:
        file = fileLoc + 'post.' + 'stress' + '.0'
        nFrames = findComments(file)
        frames = np.arange(1, nFrames + 1)
        frDflt = True
    else:
        nFrames = np.size(frames)
        frames = np.asarray(frames) + 1

    for fName in fepxData:
        print(fName)
        tmp = []
        tproc = []
        temp = []
        tFrames = []
        if fName == 'ang' or fName == 'adx' or fName == 'advel' or fName == 'crss' or fName == 'rod':
            tnf = nFrames + 1
            if restart:
                tnf = nFrames
            tFrames = frames.copy()
            if (not frDflt):
                tFrames = np.concatenate(([1], tFrames))

        else:
            tnf = nFrames
            tFrames = frames.copy()
        npf = 0
        for p in proc:
#            print(p)
            tmp = []
            tmp1 = []
            fLoc = fileLoc + 'post.' + fName + '.' + str(p)

            if frDflt:
                tmp = ta.genfromtxt(fLoc, comments='%')
            else:
                tmp = selectFrameTxt(fLoc, tFrames, comments='%')

            vec = np.atleast_2d(tmp).shape
            if vec[0] == 1:
                vec = (vec[1], vec[0])
            npf += vec[0] / tnf
            tmp1 = np.reshape(np.ravel(tmp),(tnf, np.int32(vec[0] / tnf), vec[1])).T
            tproc.append(tmp1)

        temp = np.asarray(np.concatenate(tproc, axis=1))

#        temp = tproc.reshape(vec[1], npf, tnf, order='F').copy()

        # Multiple setup for the default data names have to be changed to keep comp saved
        # First two if and if-else statements are for those that have default values
        if fName == 'ang':
            if restart:
                data['angs'] = np.atleast_3d(temp[1:4, :, :])
            else:
                data['angs_0'] = np.atleast_3d(temp[1:4, :, 0])
                data['angs'] = np.atleast_3d(temp[1:4, :, 1::1])

        elif fName == 'adx' or fName == 'advel' or fName == 'crss' or fName == 'rod':
            if fName == 'adx':
                tName = 'coord'
            elif fName == 'advel':
                tName = 'vel'
            elif fName == 'rod':
                tName = 'rod'
            else:
                tName = 'crss'
            if restart:
                data[tName] = np.atleast_3d(temp)
            else:
                data[tName + '_0'] = np.atleast_3d(temp[:, :, 0])
                data[tName] = np.atleast_3d(temp[:, :, 1::1])

        elif fName == 'dpeff':
            tName = 'pldefrate'
            data[tName] = np.atleast_3d(temp)

        elif fName == 'eqplstrain':
            tName = 'plstrain'
            data[tName] = np.atleast_3d(temp)
        elif fName == 'stress_q':
            nvec = temp.shape[0]
            nqpts = 15
            nelems = np.int32(temp.shape[1]/nqpts)
            temp1d = np.ravel(temp)
            temp4d = temp1d.reshape(nvec, nelems, nqpts, nFrames)
            data[fName] = np.swapaxes(np.swapaxes(temp4d, 0, 2), 1, 2)

        else:
            data[fName] = np.atleast_3d(temp)

    return data

def readGrainData(fileLoc, grainNum, frames=None, grData=None):
    '''
        Reads in the grain data that you are interested in. It can read the
        specific rod, gammadot, and gamma files.
        Input: fileLoc = a string of the file location
               grainNum = an integer of the grain number
               frames = what frames you are interested in, default value is all of them
               lofemData = what data files you want to look at, default value is:
                    ang, gamma, gammadot
        Output: data a dictionary that contains an ndarray of all the values
                read in the above file.
                rod_0: a float array of the original orientation at
                    each nodal point of the grain.
                rod: a float array of the orientation at each nodal point
                    of the grain through each frame.
                gamma: a float array of the integrated gammadot at each nodal
                    point of the grain through each frame.
                gdot: a float array of the gammadot at each nodal point
                    of the grain through each frame.
    '''
    
    flDflt = False
    frDflt = False
    data = {}
    
    if grData is None:
        grData = ['ang', 'gamma', 'gdot']
        flDflt = True
    if frames is None:
        strgrnum = np.char.mod('%4.4d', np.atleast_1d(grainNum))[0]
        if grData[0] == 'ang':
            fend = '.rod'
        else:
            fend = '.data'
        file = fileLoc + 'gr_' + grData[0] + strgrnum + fend
        nFrames = findComments(file)
        if grData[0] == 'ang':
            nFrames = nFrames - 1
        frames = np.arange(1, nFrames + 1)
        frDflt = True
    else:
        nFrames = np.size(frames)
        frames = np.asarray(frames) + 1
    
    for fName in grData:
        print(fName)
        tFrames = []
        if fName == 'ang':
            tnf = nFrames + 1
            tFrames = frames.copy()
            fend = 'rod'
            if (not frDflt):
                tFrames = np.concatenate(([1], tFrames)) 
        
        else:
            tnf = nFrames
            tFrames = frames.copy()
            fend = 'data'
        
        tmp = []
        strgrnum = np.char.mod('%4.4d', np.atleast_1d(grainNum))[0]
        fLoc = fileLoc + 'gr_' + fName + strgrnum + '.' + fend

        if frDflt:
            tmp = ta.genfromtxt(fLoc, comments='%')
        else:
            tmp = selectFrameTxt(fLoc, tFrames, comments='%')

        vec = np.atleast_2d(tmp).shape
        if vec[0] == 1:
            vec = (vec[1], vec[0])
        temp = np.reshape(np.ravel(tmp),(tnf, np.int32(vec[0] / tnf), vec[1])).T
        
        if fName == 'ang':
            data['angs_0'] = np.atleast_3d(temp[:,:,0])
            data['angs'] = np.atleast_3d(temp[:, :, 1::1])
        else:
            data[fName] = np.atleast_3d(temp)
        
    return data

def readLOFEMData(fileLoc, nProc, nqpts=15, frames=None, lofemData=None):
    '''
        Reads in the data files that you are interested in across all the processors
        and only for the frames that you are interested in as well
        Input: fileLoc = a string of the file location
               nProc = an integer of the number of processors used in the simulation
               frames = what frames you are interested in, default value is all of them
               lofemData = what data files you want to look at, default value is:
                    .strain, .stress,.crss, .agamma
        Output: data = a dictionary that contains a list/ndarray of all read in data files. If other files other than default are wanted than the keys for those
            values will be the file location. The default files have the following key
            values:
            coord_0: a float array of original coordinates
            hard_0: a float array of original crss_0/g_0 for each element
            angs_0: a float array of original kocks angles for each element
            vel_0: a float array of original velocity at each node
            coord: a float array of deformed coordinates
            hard: a float array of crss/g for each element
            angs: a float array of evolved kocks angles for each element
            stress: a float array of the crystal stress for each element
            strain: a float array of the sample strain for each element
            pldefrate: a float of the plastic deformation rate for each element
            plstrain: a float ofp the plastic strain for each element
            vel: a float array of the velocity at each node
    '''
    flDflt = False
    frDflt = False
    data = {}
    proc = np.arange(nProc)

    if lofemData is None:
        lofemData = ['strain', 'stress', 'crss', 'agamma']
        flDflt = True
    if frames is None:
        file = fileLoc + 'lofem.' + 'stress' + '.0'
        nFrames = findComments(file)
        frames = np.arange(1, nFrames + 1)
        frDflt = True
    
    else:
        nFrames = np.size(frames)
        frames = np.asarray(frames) + 1
    for fName in lofemData:
        print(fName)
        tmp = []
        tproc = []
        temp = []
        tFrames = []
        if fName == 'ang' or fName == 'adx' or fName == 'advel' or fName == 'crss' or fName == 'rod':
            tnf = nFrames + 1
            tFrames = frames.copy()
            if (not frDflt):
                tFrames = np.concatenate(([1], tFrames))
        else:
            tnf = nFrames
            tFrames = frames.copy()
        npf = 0
        for p in proc:
#            print(p)
            tmp = []
            tmp1 = []
            fLoc = fileLoc + 'lofem.' + fName + '.' + str(p)

            if frDflt:
                tmp = ta.genfromtxt(fLoc, comments='%')
            else:
                tmp = selectFrameTxt(fLoc, tFrames, comments='%')

            vec = np.atleast_2d(tmp).shape
            if vec[0] == 1:
                vec = (vec[1], vec[0])
            npf += vec[0] / tnf
            tmp1 = np.reshape(np.ravel(tmp),(tnf, np.int32(vec[0] / tnf), vec[1])).T
            tproc.append(tmp1)

        temp = np.asarray(np.concatenate(tproc, axis=1))


#        temp = tproc.reshape(vec[1], npf, tnf, order='F').copy()

        # Multiple setup for the default data names have to be changed to keep comp saved
        # First two if and if-else statements are for those that have default values
        if fName == 'ang':
            data['angs_0'] = np.atleast_3d(temp[1:4, :, 0])
            data['angs'] = np.atleast_3d(temp[1:4, :, 1::1])

        elif fName == 'adx' or fName == 'advel' or fName == 'crss' or fName == 'rod':
            if fName == 'adx':
                tName = 'coord'
            elif fName == 'advel':
                tName = 'vel'
            elif fName == 'rod':
                tName = 'rod'
            else:
                tName = 'crss'
            data[tName + '_0'] = np.atleast_3d(temp[:, :, 0])
            data[tName] = np.atleast_3d(temp[:, :, 1::1])

        elif fName == 'dpeff':
            tName = 'pldefrate'
            data[tName] = np.atleast_3d(temp)

        elif fName == 'eqplstrain':
            tName = 'plstrain'
            data[tName] = np.atleast_3d(temp)
        elif fName == 'agamma':
            nslip = temp.shape[0]
            nqpts = 15
            nelems = np.int32(temp.shape[1]/nqpts)
            temp1d = np.ravel(temp)
            temp4d = temp1d.reshape(nslip, nelems, nqpts, nFrames)
            data[fName] = np.swapaxes(np.swapaxes(temp4d, 0, 2), 1, 2)
        else:
            data[fName] = np.atleast_3d(temp)

    return data


def fixStrain(epsVec):
    '''
    Converts the strain vector into a strain tensor
    '''    
    vec = epsVec.shape
    
    indices = [0, 1, 2, 1, 3, 4, 2, 4, 5]    
    
    strain = np.zeros((vec[0], 3, 3))

    strain = np.reshape(epsVec[:, indices], (vec[0], 3, 3))
    
    return strain
        

def findComments(fileLoc):
    '''
    Takes in a file path and then returns the number of fortran comments in that file
    Input: fileLoc-a string of the file path way
    Output: an integer of the number of comments in that file
    '''
    i = 0
    with open(fileLoc) as f:
        for line in f:
            tmp = line.split()
            if tmp[0][0] == '%':
                i += 1
    return i


def selectFrameTxt(fileLoc, frames, comments='%'):
    '''
    Takes in a file name and frames that one wants to examine and collects the data that
    relates to those frames
    Input: fileLoc=a string of the file path
           frames=a ndarray of the frames that one is interested in
           comments=a string containing the comments starting character
    Output: a list of the data refering to those frames
    '''
    i = 0
    count = 0
    nList = []
    tframes = frames.tolist()

    with open(fileLoc) as f:
        for line in f:
            tmp = line.split()
            count += 1
#            if len(tmp) > 3:
#                print(tmp)
#                print(count)
            if tmp[0] == comments:
                count = 0
                i += 1
                continue
            if i in tframes:
                nList.append(np.float_(tmp))
#    print(count)
    return np.asarray(nList)

