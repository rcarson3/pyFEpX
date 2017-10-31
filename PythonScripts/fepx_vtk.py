#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:39:13 2017

@author: robertcarson
"""

import numpy as np
import evtk.vtk
from evtk.hl import unstructuredGridToVTK

'''
List of functions available in this module
fepxconn_2_vtkconn(conn)
evtk_conn_offset_type_creation(conn, wedge_conn = None)
evtk_elem_data_creation(data, uconn, nelems, wedge_nelems)
evtk_pts_data_creation(data, uorder)
evtk_xyz_crd_creation(coords, uorder)
evtk_fileCreation(fileLoc, xcrd, ycrd, zcrd, conn, offsets, cell_types, cellData=None, cellKeys=None, ptsData=None, ptsKeys=None)
evtk_groupVTKData(fileLoc, fOutList, simTimes)
'''


def fepxconn_2_vtkconn(conn):
    '''
        Takes in the fepx connectivity array and switches it to the vtk
        format.
        
        Input:
            conn - a numpy array of the elemental connectivity array for a
                quadratic tetrahedral element with FePX nodal ordering
        Output:
            vtk_conn - a numpy array of the elemental connectivity array
                given in the vtk nodal order for a quadratic tetrahedral 
                element
    
    '''
   
    #Rearrangement of array
    
    vtko = np.array([0, 2, 4, 9, 1, 3, 5, 6, 7, 8], dtype = np.int8)
    
    vtk_conn = conn[vtko, :]
     
    
    return vtk_conn

def evtk_conn_offset_type_creation(conn, wedge_conn = None):
    '''
        Takes in an elemental connectivity array and if given a wedge 
        element connectivity array and outputs the EVTK conn and offset
        arrays.
        
        Input:
            conn - a 2D numpy array of the elemental connectivity array
                for a quadratic tetrahedral element given in the vtk format
            (Optional) wedge_conn - a 2D numpy array of the elemental wedge
                connectivity array. It is an optional input and is not
                needed.
        Output: evtk_conn  - 1D array that defines the vertices associated 
                    to each element. Together with offset define the 
                    connectivity or topology of the grid. It is assumed 
                    that vertices in an element are listed consecutively. 
                evtk_offset - 1D array with the index of the last vertex of
                    each element in the connectivity array. It should have 
                    length nelem, where nelem is the number of cells or 
                    elements in the grid.
                evtk_cell_types - 1D array with an integer that defines the
                    cell type of each element in the grid. It should have 
                    size nelem. This should be assigned from 
                    evtk.vtk.VtkXXXX.tid, where XXXX represent the type of 
                    cell. Please check the VTK file format specification 
                    for allowed cell types. 
    
    '''
    
    tet_off = 10
    wedge_off = 12
    
    conn_nelem = conn.shape[1]
    
    conn1D = conn.flatten(order = 'F')
    
    conn_off = np.arange(9, conn_nelem*tet_off, tet_off)
    
    quadTet = evtk.vtk.VtkQuadraticTetra.tid
    quadLinWedge = evtk.vtk.VtkQuadraticLinearWedge.tid 
    
    cell_type = quadTet*np.ones(conn_nelem, dtype=np.uint8, order='F')
    
    if wedge_conn is not None:
        wedge_nelem = wedge_conn.shape[1]
        wedge1D = wedge_conn.flatten(order = 'F')
        wedge_off = np.arange(0, wedge_nelem*wedge_off, wedge_off)
        wedge_type = quadLinWedge * np.ones(wedge_nelem, dtype=np.uint8, order='F')
    else:
        wedge1D = np.array([], dtype=np.int32)
        wedge_off = np.array([], dtype=np.int32)
        wedge_type = np.array([], dtype=np.uint8)
        
    
    evtk_conn = np.hstack((conn1D, wedge1D))
    evtk_offset = np.hstack((conn_off, wedge_off)) + 1
    evtk_cell_type = np.hstack((cell_type, wedge_type))
    
    return (evtk_conn, evtk_offset, evtk_cell_type)

def evtk_elem_data_creation(data, uconn, nelems, wedge_nelems):
    '''
        Arranges the data given in the global numbers to the unique
        connectivity array if there's a difference in how things are numbered.
        This would help if one has each grain as it's own "mesh" without having
        their nodes connected to the main mesh. Therefore, the data would
        need to be rearranged to account for this new difference.
        If one does have connectivity to other grains through a wedge element
        then the wedge elements have a data value of 0.
        
        Input: data - global data that we're going to rearrange. It should
                be a 2D numpy array with the 2 axis having a length equal
                to the number of elements
            
               uconn - a 1D numpy array with the unique indices corresponding
                to how the data should be rearranged to fit the new order
                of the global connectivity array
            
               nelems - a scalar value telling us how many elements are in
                data. We will use it as a sanity check to make sure the data
                doesn't have a shape that conforms to what we expect it to.
            
               wedge_nelems - a scalar value telling us how many wedge
                elements are going to end up being in the connectivity array
               
        Output: evtk_data - the rearranged data array that can now be used
                 in our visualizations using paraview.   
    '''
    if(data.ndim == 1):
        data = np.atleast_2d(data)
    
    dlen = data.shape[0]
    
    dnelems = nelems + wedge_nelems
    
    evtk_data = np.zeros((dlen, dnelems), dtype='float64')
    
    evtk_data[:,0:nelems] = data[:, uconn]
    
    if(dlen == 6):
        vec = evtk_data.shape
        indices = [0, 1, 2, 1, 3, 4, 2, 4, 5]    
#        temp = np.zeros((3, 3, vec[1]))
        temp = np.reshape(evtk_data[indices, :], (3, 3, vec[1]))
        evtk_data = temp
    
    return evtk_data

def evtk_pts_data_creation(data, uorder):
    '''
        Arranges the data given in the global numbers to the unique
        connectivity array if there's a difference in how things are numbered.
        This would help if one has each grain as it's own "mesh" without having
        their nodes connected to the main mesh. Therefore, the data would
        need to be rearranged to account for this new difference. 
    
        Input: data - global data that we're going to rearrange. It should
                be a 2D numpy array with the 2nd axis having a length equal
                to the number of number of nodal positions in the mesh.
               
              uorder - a 1D numpy array with the indices corresponding
               to how the crds should be rearranged to the new nodal
               arrangment of the mesh. It's possible that there are repeated
               indices in this array that would correspond to the nodes that
               were lying on the grain boundary.
        
        Output: evtk_pts - the rearranged data array that can now be used
                 in our visualizations using paraview
    
    '''
    
    epts_len = uorder.shape[0]
    
    evtk_pts = np.zeros((3, epts_len), dtype='float64')
    
    evtk_pts[:,:] = data[:, uorder]

    return evtk_pts

def evtk_xyz_crd_creation(coords, uorder):
    '''
        Arranges the coords given in the global numbers to the unique
        connectivity array if there's a difference in how things are numbered.
        This would help if one has each grain as it's own "mesh" without having
        their nodes connected to the main mesh. Therefore, the data would
        need to be rearranged to account for this new difference. 
    
        Input: coords - global coords that we're going to rearrange. It should
                be a 2D numpy array with the 2nd axis having a length equal
                to the number of number of nodal positions in the mesh.
               
              uorder - a 1D numpy array with the indices corresponding
               to how the crds should be rearranged to the new nodal
               arrangment of the mesh. It's possible that there are repeated
               indices in this array that would correspond to the nodes that
               were lying on the grain boundary.
        
        Output: evtk_x - the rearranged x coords that can now be used
                 in our visualizations using paraview
                evtk_y - the rearranged y coords that can now be used
                 in our visualizations using paraview
                evtk_z - the rearranged z coords that can now be used
                 in our visualizations using paraview
    
    '''
    
    epts_len = uorder.shape[0]
    
    evtk_x = np.zeros((epts_len), dtype='float64')
    evtk_y = np.zeros((epts_len), dtype='float64')
    evtk_z = np.zeros((epts_len), dtype='float64')
    
    evtk_x[:] = coords[0, uorder]
    evtk_y[:] = coords[1, uorder]
    evtk_z[:] = coords[2, uorder]
    
    return(evtk_x, evtk_y, evtk_z)

def evtk_fileCreation(fileLoc, xcrd, ycrd, zcrd, conn, offsets, cell_types, cellData=None, cellKeys=None, ptsData=None, ptsKeys=None):
    '''
       Wrapper around the unstructuredGridToVTK function 
            Export unstructured grid and associated data.

        Inputs:
            fileLoc: name of the file without extension where data should be saved.
            xcrd, ycrd, zcrd: 1D arrays with coordinates of the vertices of cells.
                    It is assumed that each element has diffent number of vertices.
            conn: 1D array that defines the vertices associated to 
                    each element. Together with offset define the connectivity
                    or topology of the grid. It is assumed that vertices in 
                    an element are listed consecutively.
            offsets: 1D array with the index of the last vertex of each element
                    in the connectivity array. It should have length nelem,
                    where nelem is the number of cells or elements in the grid.
            cell_types: 1D array with an integer that defines the cell type
                    of each element in the grid. It should have size nelem.
                    This should be assigned from evtk.vtk.VtkXXXX.tid,
                    where XXXX represent the type of cell. Please check the
                    VTK file format specification for allowed cell types.                       
            cellData: Dictionary with variables associated to each line.
                      Keys should be the names of the variable stored in each array.
                      All arrays must have the same number of elements.        
            pointData: Dictionary with variables associated to each vertex.
                       Keys should be the names of the variable stored in each array.
                       All arrays must have the same number of elements.

        Output:
            fOut: Full path to saved file.
    '''
    fOut = unstructuredGridToVTK(fileLoc, xcrd, ycrd, zcrd, conn, offsets, cell_types, cellData=cellData, cellKeys=cellKeys, pointData=ptsData, pointKeys=ptsKeys)
    
    return fOut
    
def evtk_groupVTKData(fileLoc, fOutList, simTimes):
    '''
        A wrapper function that creates a VTK group to visualize time
        dependent data in Paraview.
        
        Input:
            fileLoc - name of the file without extension where the group
                file should be saved.
            fOutList - a list of all of the vtk files that are to be grouped
                together
            simTimes - a numpy 1D array of the simulation times that each
                vtk corresponds to
    
    
    '''
    
    ltime = simTimes.shape[0]
    
    gclass = evtk.vtk.VtkGroup(fileLoc)
    
    for i in range(ltime):
        gclass.addFile(filepath = fOutList[i], sim_time=simTimes[i])
    
    gclass.save()

            
        
    
    