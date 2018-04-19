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
def readMesh(fileLoc, fileName, LOFEM=False):
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
    ugrains = np.unique(grains)
    phases = np.genfromtxt(grainLoc, usecols=(1), skip_header=1, skip_footer=0)
    kocks = np.genfromtxt(kockLoc, usecols=(0, 1, 2), skip_header=2, skip_footer=1)
    if not kocks.shape[0] == ugrains.shape[0]:
        kocks = np.genfromtxt(kockLoc, usecols=(0, 1, 2), skip_header=2, skip_footer=0)
    mesh['con'] = np.require(np.asarray(con, order='F', dtype=np.int32).transpose(), requirements=['F'])
    mesh['crd'] = np.require(np.asarray(crd, order='F').transpose(), requirements=['F'])
    mesh['surfaceNodes'] = np.require(np.asarray(surfaceNodes, order='F',dtype=np.int32).transpose(), requirements=['F'])
    mesh['grains'] = np.asfortranarray(grains.transpose(), dtype=np.int32)
    mesh['kocks'] = util.mat2d_row_order(np.asfortranarray(kocks.transpose()))
    mesh['phases'] = np.asfortranarray(phases.transpose(),dtype=np.int8)
    
    if (LOFEM):
        crd_meshLoc = fileLoc + fileName + '.cmesh'
        crd_grainLoc = fileLoc + fileName + '.cgrain'
        
        cgrains = ta.genfromtxt(crd_grainLoc, usecols=(0))
        cphases = ta.genfromtxt(crd_grainLoc, usecols=(1))
        ccon = ta.genfromtxt(crd_meshLoc, skip_header=1)
        
        mesh['crd_con'] = np.asfortranarray(ccon.transpose(), dtype=np.int32) - 1
        mesh['crd_grains'] = np.asfortranarray(cgrains.transpose(), dtype=np.int32)
        mesh['crd_phases'] = np.asfortranarray(cphases.transpose(), dtype=np.int8)

    return mesh

def mesh_node_neigh(conn, nd_conn):
    '''
        Creates a list of all of a nodes neighbors given the connectivity
        array and the node elem connectivity array. 
        
        Input: conn = a numpy array of the mesh connectivity array
               nd_conn = a list of sets for each node and what elems they
                       are connected to
               
        Output: nd_neigh = a list of sets of a nodes neighbors
        
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
    conn2 = np.copy(conn)
    
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
            conn2[ind, j] = incr + i + np.where(ugrns == grains[j])[0][0]
                  
        #We don't need to increment anything if there is only one grain for
        #that node
        nincr = ugrns.shape[0] - 1
        incr = incr + nincr
            
    return conn2

def grain_boundary_nodes(ndconn, grains, nnode):
    '''
    Takes in the nodal and elemental connectivity arrays. It then goes through
    the list of nodes and finds all of the nodes that belong to a GB.
    Later we would need to find what elements share that surface. We would
    need elements with 6 or more elements on a surface to be connected to
    the other elements. This would be used in our global nodal connectivity
    matrix.
    
    Input: ndconn = a list of sets for each node and what elems they are
                connected to
           conn = a numpy array of the mesh element connectivity array
           grains = a numpy array corresponding to what grain each element is
                in
           nnode = the number of nodes in the mesh
    Output: gbnodes = a dictionary of dictionary. The first key is the original
                      node number and the second key is the grain number for that
                      node. It will then return the new node number
            nincr = a numpy array containing the number of increments made
                for each node. This can be used for many things and one
                such thing is finding the GB elements by finding the elements
                with 6 or more GB nodes
    '''
    
    nincr = np.zeros(nnode, dtype='int32')
    incr = 0
    
    for i in range(nnode):
        #We want a numpy array of all the elements connected to that node
        ndelems =  np.array(list(ndconn[i]), dtype='int32')
        #We also want to know how many unique grains we actually have
        ugrns = np.unique(grains[ndelems])
        #Our inner loop that were going to use to go through the data
        #The node doesn't need to be incremented if it isn't on a GB.
        nincr[i] = ugrns.shape[0] - 1
        
    nodes = np.int32(np.where(nincr > 0)[0])
    #Going ahead and initiallizing our set all at once
#    gbnodes = [np.zeros((3, nincr[i]+1), dtype='int32') for i in nodes]
    gbnodes = dict.fromkeys(nodes, {})


    #Cycle through the index of all the nodes that were on the boundary
    for i in nodes:
        #We want a numpy array of all the elements connected to that node
        ndelems =  np.array(list(ndconn[i]), dtype='int32')
        #We also want to know how many unique grains we actually have
        ugrns = np.unique(grains[ndelems])
        gbnodes[i] = dict.fromkeys(ugrns, None)
        tmp = set()
        for j in ndelems:
            #Finally we increment the conn array
            new_index = incr + i + np.where(ugrns == grains[j])[0][0]
            tmp.add((i, new_index, grains[j]))
            
        for item in tmp:
            gbnodes[i][np.int32(item[2])] = np.int32(item[1])
        
        incr = incr + nincr[i]

    return (gbnodes, nincr)

def grain_boundary_elements(nincr, gbnodes, nd_conn, nd_conn_gr, conn, grain):
    '''
    It takes in the number of grains a coord originally belongs to. It takes
    in the list of grain boundary coordinates and there respectively updated
    coords. It takes in the original nd_conn array before the nodes on the
    grain boundary were updated with new values. It finally takes in the 
    updated nodal connectivity array which will be used to generate the
    connectivity array for the surface grain boundary elements. Finally,
    it will output a list of the GB element index and its paired element.
    Input:
        gbnodes   =  a dictionary of dictionary. The first key is the original
                      node number and the second key is the grain number for that
                      node. It will then return the new node number 
        nincr     = a numpy array containing the number of increments made
                    for each node. This can be used for many things and one
                    such thing is finding the GB elements by finding the elements
                    with 6 or more GB nodes
        ndconn    = a list of sets for each node and what elems they are
                     connected to
        conn      = a numpy array of the mesh element connectivity array
        grain     = a numpy array corresponding to what grain each element is
                    in
        ndconn_gr = a list of sets for each node after being renumbered for 
                    so that gb nodes are on seperate nodes 
                    and what elems they are connected to
                    
    Output:
        gb_elems    = A numpy array of all of the elements on the grain boundary with
                      6 or more nodes on the surface. In other words we want
                      to know what elements have an element face on the grain
                      boundary.
        gb_conn     = The element connectivity array for each grain boundary
                      element. It allows for the reconstruction of the triangular
                      prism elements. It is given as a numpy array. The last
                      two elements in the array tell us what elements correspond
                      to that particular connectivity array.
        gb_elem_set = A set of frozen sets that contains all of grain boundary
                      element pairs. One can then use this in conjunction with
                      the gb_conn to build up our grain boundary surface elements.
                      It could also be used in a number of different areas.        
    '''
    
    nelems = conn.shape[1]
    
    gb_elem_set = set()
    nsurfs = np.zeros(nelems, dtype='int8')
    
    for i in range(nelems):
        tconn = np.squeeze(conn[:, i])
        tincr = np.sum(nincr[tconn] > 0)
        nsurfs[i] = tincr
        
    gb_elems = np.where(nsurfs > 5)[0]
    gb_elems_iter = np.where(nsurfs > 5)[0]
    surf_index = np.zeros((gb_elems.shape[0], 5), dtype=np.int32)
    
    j = 0
    for i in gb_elems_iter:
        tconn = np.squeeze(conn[:, i])
        index = np.where(nincr[tconn] > 0)[0]
        surf_index[j, 4] = i
        tmp = 0
        #surface 1 of 10 node tet
        if(np.any(index == 1) & np.any(index == 5)):
            i0 = tconn[0]
            i1 = tconn[1]
            i2 = tconn[2]
            i3 = tconn[3]
            i4 = tconn[4]
            i5 = tconn[5]
            #All need to be on the surface
            gb_surf_test = (i0 in gbnodes) & (i1 in gbnodes) & (i2 in gbnodes) & (i3 in gbnodes) & (i4 in gbnodes) & (i5 in gbnodes)
            if gb_surf_test:
                #Finds the intersection of the 6 different node set
                s1 = nd_conn[i0] & nd_conn[i1] & nd_conn[i2] & nd_conn[i3] & nd_conn[i4] & nd_conn[i5]
                #We need to check that the length is greater than 1 if it isn't than we toss
                #that point
                ugrns = np.unique(grain[list(s1)])
                if(len(s1) > 1) & (ugrns.shape[0] > 1):
                    tmp = tmp + 1
                    surf_index[j, 0] = list(s1 - {i})[0]
                    gb_elem_set.add(frozenset(s1))
        #surface 2 of 10 node tet
        if(np.any(index == 1) & np.any(index == 9)):
            i0 = tconn[0]
            i1 = tconn[1]
            i2 = tconn[2]
            i3 = tconn[7]
            i4 = tconn[9]
            i5 = tconn[6]
            #All need to be on the surface
            gb_surf_test = (i0 in gbnodes) & (i1 in gbnodes) & (i2 in gbnodes) & (i3 in gbnodes) & (i4 in gbnodes) & (i5 in gbnodes)
            if gb_surf_test:
                #Finds the intersection of the 6 different node set
                s1 = nd_conn[i0] & nd_conn[i1] & nd_conn[i2] & nd_conn[i3] & nd_conn[i4] & nd_conn[i5]
                #We need to check that the length is greater than 1 if it isn't than we toss
                #that point. It also turns out that somehow we could end up with elements that are on the
                #same grain. I'm not sure how since they should all be unique.
                ugrns = np.unique(grain[list(s1)])
                if(len(s1) > 1) & (ugrns.shape[0] > 1):
                    tmp = tmp + 1
                    surf_index[j, 1] = list(s1 - {i})[0]
                    gb_elem_set.add(frozenset(s1))
        #surface 3 of 10 node tet
        if(np.any(index == 3) & np.any(index == 9)):
            i0 = tconn[2]
            i1 = tconn[3]
            i2 = tconn[4]
            i3 = tconn[8]
            i4 = tconn[9]
            i5 = tconn[7]
            #All need to be on the surface
            gb_surf_test = (i0 in gbnodes) & (i1 in gbnodes) & (i2 in gbnodes) & (i3 in gbnodes) & (i4 in gbnodes) & (i5 in gbnodes)
            if gb_surf_test:
                #Finds the intersection of the 6 different node set         
                s1 = nd_conn[i0] & nd_conn[i1] & nd_conn[i2] & nd_conn[i3] & nd_conn[i4] & nd_conn[i5]
                #We need to check that the length is greater than 1 if it isn't than we toss
                #that point
                ugrns = np.unique(grain[list(s1)])
                if(len(s1) > 1) & (ugrns.shape[0] > 1):
                    tmp = tmp + 1
                    surf_index[j, 2] = list(s1 - {i})[0]
                    gb_elem_set.add(frozenset(s1))
        #surface 4 of 10 node tet
        if(np.any(index == 5) & np.any(index == 9)):
            i0 = tconn[4]
            i1 = tconn[5]
            i2 = tconn[0]
            i3 = tconn[6]
            i4 = tconn[9]
            i5 = tconn[8]
            #All need to be on the surface
            gb_surf_test = (i0 in gbnodes) & (i1 in gbnodes) & (i2 in gbnodes) & (i3 in gbnodes) & (i4 in gbnodes) & (i5 in gbnodes)
            if gb_surf_test:
                #Finds the intersection of the 6 different node set
                s1 = nd_conn[i0] & nd_conn[i1] & nd_conn[i2] & nd_conn[i3] & nd_conn[i4] & nd_conn[i5]
                #We need to check that the length is greater than 1 if it isn't than we toss
                #that point
                ugrns = np.unique(grain[list(s1)])
                if(len(s1) > 1) & (ugrns.shape[0] > 1):
                    tmp = tmp + 1
                    surf_index[j, 3] = list(s1 - {i})[0]  
                    gb_elem_set.add(frozenset(s1))
        #We need to check and make sure that if a grain boundary element
        #is actually a grain boundary element.
        #If it did not share a face at all then we remove it from the grain
        #boundary element array
        if(tmp > 0):
            j = j + 1
        else:
            gb_elems = gb_elems[gb_elems != i]
        
    nsurf_elems = len(gb_elem_set)
    
    gb_conn = np.zeros((14, nsurf_elems), dtype=np.int32)
    
#    nb_gbnodes = len(gbnodes)
    
    j = 0
    for gb_els in gb_elem_set:
        elems = np.asarray(list(gb_els), dtype=np.int32)
        gb_conn[12:14, j] = elems
        tgrains = np.int32(grain[elems])
        
        ind2 = np.where(elems[0] == np.squeeze(surf_index[:,4]))[0]
        surf = np.where(elems[1] == np.squeeze(surf_index[ind2, 0:4]))[0]
        tconn = np.squeeze(conn[:, elems[0]])
        
        if(surf == 0):
            ind = np.asarray([0,1,2,3,4,5])
        elif(surf == 1):
            ind = np.asarray([0,1,2,7,9,6])  
        elif(surf == 2):
            ind = np.asarray([2,3,4,8,9,7]) 
        else:
            ind = np.asarray([4,5,5,6,9,8]) 
                    
        nodes_orig = np.int32(np.squeeze(tconn[ind]))
        
        i1 = nodes_orig[0]
        i2 = nodes_orig[1]
        i3 = nodes_orig[2]
        i4 = nodes_orig[3]
        i5 = nodes_orig[4]
        i6 = nodes_orig[5]
        
        #Manual creation of the index since there's no easy way to do this
        #through the use of a loop with the current ordering of the conn
        #array
        gb_conn[0, j] = gbnodes[i1][tgrains[0]]
        gb_conn[6, j] = gbnodes[i2][tgrains[0]]        
        gb_conn[1, j] = gbnodes[i3][tgrains[0]]        
        gb_conn[7, j] = gbnodes[i4][tgrains[0]]        
        gb_conn[2, j] = gbnodes[i5][tgrains[0]]        
        gb_conn[8, j] = gbnodes[i6][tgrains[0]]
        
        gb_conn[3, j] = gbnodes[i1][tgrains[1]]        
        gb_conn[9, j] = gbnodes[i2][tgrains[1]]        
        gb_conn[4, j] = gbnodes[i3][tgrains[1]]
        gb_conn[10, j] = gbnodes[i4][tgrains[1]]
        gb_conn[5, j] = gbnodes[i5][tgrains[1]]        
        gb_conn[11, j] = gbnodes[i6][tgrains[1]]
        
        j = j + 1
            
    
    return (gb_elems, gb_conn, gb_elem_set)

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
        fName = fepxData[0]
        file = fileLoc + 'post.' + fName + '.0'
        nFrames = findComments(file)
        if fName == 'ang' or fName == 'adx' or fName == 'advel' or fName == 'crss' or fName == 'rod':
            if restart:
                frames = np.arange(1, nFrames + 1)
            else:
                frames = np.arange(1, nFrames)
                nFrames = nFrames - 1
        else: 
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

def mpi_partioner(nprocs, ncrds, nelems):
    '''
    Returns the ncrd and nelem partion per processor
    '''
    
    proc_elems = np.zeros((nprocs), dtype='int32')
    proc_crds = np.zeros((nprocs), dtype='int32')
    
    for i in range(nprocs):
        nlocal  = np.int(nelems / nprocs)
        s       = i * nlocal + 1
        deficit = nelems%nprocs
        s       = s + min(i, deficit)
        if (i < deficit):
            nlocal = nlocal + 1
        e = s + nlocal - 1
        if ((e > nelems) or (i == (nprocs - 1))):
            e = nelems
            
        proc_elems[i] = e - s + 1
        
        nlocal  = np.int(ncrds / nprocs)
        s       = i * nlocal + 1
        deficit = ncrds%nprocs
        s       = s + min(i, deficit)
        if (i < deficit):
            nlocal = nlocal + 1
        e = s + nlocal - 1
        if ((e > ncrds) or (i == (nprocs - 1))):
            e = ncrds
            
        proc_crds[i] = e - s + 1
    
    return (proc_elems, proc_crds)

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

def readLOFEMData(fileLoc, nProc, nstps=None, nelems=None, ncrds=None, nqpts=15, frames=None, lofemData=None, restart=False):
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
        lofemData = ['strain', 'stress', 'crss', 'agamma', 'ang']
        flDflt = True
    if frames is None:
        if nstps is None:
            fName = lofemData[0]
            if fName == 'ang':
                strgrnum = np.char.mod('%4.4d', np.atleast_1d(0))[0]
                file = fileLoc + 'gr_' + fName + strgrnum + '.rod'
            else:
                file = fileLoc + 'lofem.' + fName + '.0'
            nFrames = findComments(file)
            if fName == 'ang' or fName == 'adx' or fName == 'advel' or fName == 'crss' or fName == 'rod':
                if restart:
                    frames = np.arange(1, nFrames + 1)
                else:
                    frames = np.arange(1, nFrames)
                    nFrames = nFrames - 1
            else: 
                frames = np.arange(1, nFrames + 1)
        frDflt = True
    
    else:
        if nstps is None:
            fName = lofemData[0]
            if fName == 'ang':
                strgrnum = np.char.mod('%4.4d', np.atleast_1d(0))[0]
                file = fileLoc + 'gr_' + fName + strgrnum + '.rod'
            else:
                file = fileLoc + 'lofem.' + fName + '.0'
            nstps = findComments(file)
            if fName == 'ang' or fName == 'adx' or fName == 'advel' or fName == 'crss' or fName == 'rod':
                if not restart:
                    nstps = nstps - 1
#            frames = np.arange(1, nstps + 1)
        proc_elems, proc_crds = mpi_partioner(nProc, ncrds, nelems)
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
            if fName == 'ang':
                strgrnum = np.char.mod('%4.4d', np.atleast_1d(p))[0]
                fLoc = fileLoc + 'gr_' + fName + strgrnum + '.rod'
            else:
                fLoc = fileLoc + 'lofem.' + fName + '.' + str(p)

            if frDflt:
                tmp = ta.genfromtxt(fLoc, comments='%')
            else:
                if fName == 'ang' or fName == 'adx' or fName == 'advel' or fName == 'rod':
                    skipst = proc_crds[p] * (tFrames[0] - 1)
                    skipft = proc_crds[p] * (nstps - (tFrames[0]))
                elif fName == 'agamma_q' or fName == 'gamma_q' or fName == 'gammadot_q':
                    skipst = proc_elems[p] * (tFrames[0] - 1) * nqpts
                    skipft = proc_elems[p] * tFrames[0] * nqpts
                else:
                    skipst = proc_elems[p] * (tFrames[0] - 1)
                    skipft = proc_elems[p] * (nstps - (tFrames[0]))
                    nvals = skipft - skipst
                nvals = skipft - skipst
                tmp = np.genfromtxt(fLoc, comments='%', skip_header=skipst, max_rows=nvals)
#                tmp = selectFrameTxt(fLoc, tFrames, comments='%')

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

        if fName == 'adx' or fName == 'advel' or fName == 'crss' or fName == 'ang':
            if fName == 'adx':
                tName = 'coord'
            elif fName == 'advel':
                tName = 'vel'
            elif fName == 'ang':
                tName = 'angs'
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
        elif fName == 'agamma_q' or fName == 'gamma_q' or fName == 'gammadot_q':
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
    
    strain = np.zeros((3, 3, vec[1]))

    strain = np.reshape(epsVec[indices, :], (3, 3, vec[1]))
    
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

