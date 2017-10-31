import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plotTriangleMesh(mesh, **kwargs):
    '''
    It requires a mesh with a connectivity in it and the coordinates.
    If singular values are needed to be plotted that those must be specified in the
    dictionary as 'spatial' those will than be added to the plot as well
    
    Input: mesh['crd'] should be 3xn where n > = 3
    '''

    if len(kwargs) == 0:
        scalar = np.ones((mesh['crd'].shape[1], 1))
    else:
        scalar = kwargs['spatial']

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    triang = mtri.Triangulation(x=mesh['crd'][0, :].T, y=mesh['crd'][1, :].T, triangles=mesh['con'].T)

    ax.plot_trisurf(triang, mesh['crd'][2, :])

    plt.show()


def plotSurface(mesh, **kwargs):
    '''
    It will plot the surface of any polygon that has been meshed up. Plotting of spatial data is also
    allowed and this is accomplished by providing a set of singular data points for each being plotted of
    the surface of the polygon. If spatial data is not provided than the surface will all be plotted the
    same color 'blue'
    
    Input: mesh a dictionary containing a set of coordinates/nodal value and a set of connectivity of 
           each face
           
           mesh['crd'] should be a 3xn where n > = 3. The x, y, and z coords are located at 
           mesh['crd'][0,:], mesh['crd'][1,:], and mesh['crd'][2,:] respectively. If a 2d surface is to
           be plotted other matplotlib functions might be more appropriate, but one can do it here by
           setting all of the out of plane coordinates to be equal to a singular constant.
           
           mesh['con'] should be a a 3xn where n >= 1. The connectivity should follow a standard clockwise
           or counter-clockwise order around the face of the surface polygon. If it is not then no
           insurance can be made that junk won't be plotted that doesn't represent what one was hoping to
           get out.
           
           kwargs input: "scalar" - the spatial data corresponding to what one wants to plot on each fac
                             of the surface.
                         "colorMap" - the color map that one wants to use with the scalar values, default
                             value is jet
                         Other inputs are the same as those used in the Poly3DCollection object, so
                             "facecolors", "edgecolors", "alpha", "zsort".
                                 The facecolors arg is replaced with the scalar data mapping if that is 
                                 provided.
                                 
    Output: a poly3dcollection is outputted 
    '''

    if mesh['crd'].shape[0] != 3 or mesh['crd'].shape[1] < 3:
        print('The inputted mesh[''crd''] is not correct. mesh[''crd''] needs to have dim of 3xn where n>=3')
        raise ValueError('The inputted mesh[''crd''] is not correct. mesh[''crd''] needs to have dim of 3xn where n>=3')

    if mesh['con'].shape[0] is not 3:
        print('The inputted mesh[''con''] is not correct. mesh[''con''] needs to have dim of 3x1 where n>=3')
        raise ValueError('The inputted mesh[''con''] is not correct. mesh[''con''] needs to have dim of 3xn where n>=1')

    condim = np.atleast_2d(mesh['con']).shape

    # Check to see if connectivity was originally one dimension
    # and fix it if it to be 2d and in the right order
    if condim[0] == 1 and condim[1] == 3:
        condim = condim([1,0])
        mesh['con'] = np.atleast_2d(mesh['con']).T

    scalars = kwargs.pop('scalar', None)
    colormap = kwargs.pop('colorMap', None)
    facecolors = kwargs.pop('facecolors', None)

    # set up the color options

    if colormap is None:
        colormap = cm.jet

    if scalars is None and facecolors is None:
        facecolors = colormap(np.zeros(condim[1]))
    elif scalars is not None and facecolors is None:
        N = scalars / scalars.max()
        facecolors = colormap(N)
    elif scalars is None and facecolors is not None:
        N = np.random.rand(condim[1])
        print(N.shape)
        facecolors = colormap(N)
        
    print(facecolors)

    '''
    Creating the polygon/surface vertices
    poly_verts is initially initiallized to be a zeros 3d matrix
    '''

    poly_verts = np.zeros((condim[1], 3, 3))

    ind = 0
    
    minx = np.max(mesh['crd'][0, :])
    maxx = np.min(mesh['crd'][0, :])
    miny = np.max(mesh['crd'][1, :])
    maxy = np.min(mesh['crd'][1, :])
    minz = np.max(mesh['crd'][2, :])
    maxz = np.min(mesh['crd'][2, :])

    for con in mesh['con'].T:
        
        x = mesh['crd'][0, np.int_(con)]
        y = mesh['crd'][1, np.int_(con)]
        z = mesh['crd'][2, np.int_(con)]
        
        tminx = np.min(x)
        tmaxx = np.max(x)
        tminy = np.min(y)
        tmaxy = np.max(y)
        tminz = np.min(z)
        tmaxz = np.max(z)
        
        if tminx < minx:
            minx = tminx
        if tminy < miny:
            miny = tminy
        if tminz < minz:
            minz = tminz
            
        if tmaxx > minx:
            maxx = tmaxx
        if tmaxy > maxy:
            maxy = tmaxy
        if tmaxz > maxz:
            maxz = tmaxz
        

        vertices = np.asarray(list(zip(x, y, z)))

        poly_verts[ind, :, :] = vertices

        ind += 1

    coll = Poly3DCollection(poly_verts, facecolors=facecolors, **kwargs)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.add_collection(coll)

    xlim = [minx * 1.1, maxx * 1.1]
    ylim = [miny * 1.1, maxy * 1.1]
    zlim = [minz * 1.1, maxz * 1.1]

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_zlim(zlim[0], zlim[1])
    ax.elev = 50

    plt.show()

    return coll

def plotPolygon(mesh, **kwargs):
    '''
    It will plot the surface of any polygon that has been meshed up. Plotting of spatial data is also
    allowed and this is accomplished by providing a set of singular data points for each being plotted of
    the surface of the polygon. If spatial data is not provided than the surface will all be plotted the
    same color 'blue'. Currently, only the following element types are taken in: standard linear
    tetrahedral, standard quadratic tetrahedral, and FePX quadratic tetrahedral element order. In the 
    future, the following element types will be added standard 8 node brick element and 20 node brick 
    element order.
    
    Input: mesh a dictionary containing a set of coordinates/nodal value and a set of connectivity of 
           each face
           
           mesh['crd'] should be a 3xn where n > = 3. The x, y, and z coords are located at 
           mesh['crd'][0,:], mesh['crd'][1,:], and mesh['crd'][2,:] respectively. If a 2d surface is to
           be plotted other matplotlib functions might be more appropriate, but one can do it here by
           setting all of the out of plane coordinates to be equal to a singular constant.
           
           mesh['con'] should be a a 4xn where n >= 1. The connectivity should follow a standard clockwise
           or counter-clockwise order around the face of the surface polygon. If it is not then no
           insurance can be made that junk won't be plotted that doesn't represent what one was hoping to
           get out.
           
           kwargs input: "scalar" - the spatial data corresponding to what one wants to plot on each
                             element
                         "colorMap" - the color map that one wants to use with the scalar values, default
                             value is jet
                         "fepx" - the element type used is fepx type
                         Other inputs are the same as those used in the Poly3DCollection object, so
                             "facecolors", "edgecolors", "alpha", "zsort".
                                 The facecolors arg is replaced with the scalar data mapping if that is 
                                 provided.
                                 
    Output: a poly3dcollection is outputted 
    '''
    
    condim = np.atleast_2d(mesh['con']).shape

    # Check to see if connectivity was originally one dimension
    # and fix it if it to be 2d and in the right order
    if condim[0] == 1 and condim[1] > 1:
        mesh['con'] = np.atleast_2d(mesh['con']).T

    numnode = mesh['con'].shape[0]
    
    print(numnode)
    
    if numnode == 4:
        elem = 'ltet'
    elif numnode == 10:
        etype = kwargs.pop('fepx', None)
        if etype is None:
            elem = 'qtet'
        else:
            elem = 'fepx'
            
#     scalar = kwargs.pop('scalars', None)
    
    mesh['con'] = getelemface(mesh['con'], elem)
    
    coll = plotSurface(mesh, **kwargs)
        
    return coll    
        
    
def getelemface(con, eltype):
    '''
    It takes in the connectivity of the nodes of the element and returns the appropriate surface
    connectivity of the element. So if a tetrahedral element is taken in then the surface connectivity
    now is a [3 x 4n] array where n is the number of elements.
    
    Input: con - a [m x n] array where m is atleast 4 and corresponds to the number of nodes in the
                 element. Then n is the number of elements
           eltype - a string that describes the element type and is one of the following:
                    'ltet' - a standard linear tetrahedral element
                     'qtet' - a standard quadratic tetrahedral element
                     'fepx' - a quadratic tetrahedral element that corresponds to fepx propram input
    '''
    
    nelem = con.shape[1]
    
    surfcon = np.zeros((3, nelem*4))
    
    print(eltype)
    
    j = 0
    ind = 0
    
    for i in con.T:
        
        if eltype == 'ltet' or eltype == 'qtet':
            surfcon[:, j] = i[[0, 1, 2]]
            surfcon[:, j+1] = i[[0, 1, 3]]
            surfcon[:, j+2] = i[[1, 2, 3]]
            surfcon[:, j+3] = i[[2, 0, 3]]
        elif eltype == 'fepx':
            surfcon[:, j] = i[[0, 2, 4]]
            surfcon[:, j+1] = i[[0, 2, 9]]
            surfcon[:, j+2] = i[[2, 4, 9]]
            surfcon[:, j+3] = i[[4, 0, 9]]
            
        ind +=1
        j = ind*4
                
    return surfcon


    

'''
Example of the above function:

import Graphics
import Sphere
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.close('All')
mesh = Sphere.SphBaseMesh(2)

G=np.ones((8,))*10
G[3]=5
G[2]=0.1
N=G/G.max()

coll = Graphics.plotSurface(mesh, **{'scalar':N,'colorMap':cm.Blues,'edgecolors':'none'})


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.add_collection(coll)

xlim=[mesh['crd'][0,:].min()*1.5,mesh['crd'][0,:].max()*1.5]
ylim=[mesh['crd'][1,:].min()*1.5,mesh['crd'][1,:].max()*1.5]
zlim=[mesh['crd'][2,:].min()*1.5,mesh['crd'][2,:].max()*1.5]

ax.set_xlim(xlim[0], xlim[1])
ax.set_ylim(ylim[0], ylim[1])
ax.set_zlim(zlim[0], zlim[1])
ax.elev = 50

plt.show()

'''
