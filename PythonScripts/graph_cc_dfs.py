#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:53:05 2017

@author: robertcarson
"""

import numpy as np

def graph_DFS(graph, set_true, set_false):
    '''
        Terminology needs to be worked on to be more in line with the
        actual math terms...
        
        A graph traversal to find all connected components in the graph
        using a DFS method. Connected components are determined by if a
        node is true or false, which is determined from the logical array
        passed into the function. If it is true two nodes are said to be
        connected.
        
        It should be noted that this nodes here could be an element or a
        node in a finite element type mesh. The node just refers to a
        vertex in an undirected graph.
        
        The graph inputed is a list of each node and the nodes connected
        to it by a vertex. In a finite element mesh there are a large
        number of ways in which the connected nodes could be determined.
        If one is examining connected elements, the connected codes could
        be other elements that share a common face, edge, or node with the
        element. If one is examining a node this could vary quite a bit
        from nodes that are connected to nodes that share the same element
        as the parent node. It is up to the user to determine how they want
        these graphs to be formed.
        
        The method will require two stacks. One where we traverse through
        the graph which is a list of nodes and their "local graph". The
        second stack is required for the DFS.
        
        Input:
            graph: a list of sets that contain each nodes neighbor in the
                    global graph/mesh
            set_true: a set of nodes that are true
            set_false: a set of nodes that are false
        Output:
            con_comp: a list of sets that has all the various connected
                    components in the global graph/mesh
    '''
    
    #Creating the stack of nodes to go through    
    stack_node = []
    #We already know which ones are good so no need to do extra work
    stack_node.extend(set_true)
    #This stack is initially empty
    stack_DFS = []
    #We want the nodes that have been seen to be a set for fast look ups
    #We are setting it equal to our false set to start off with
    seen = set_false
    #We want to make a con_comp to be an empty list but will have sets
    #in it later as we go through the nodal stack.
    con_comp = []
    
    
    while stack_node:
        #Pop off an element from our nodal stack to examine the data
        i = stack_node.pop()
        #Check to see if that node has already been seen
        if i not in seen:
            #Go ahead and 
            seen.add(i)
            #We now create an empty set for our connected components
            tmp_con_comp = set()
            #Go ahead and add the node we are starting off with
            tmp_con_comp.add(i)
            #Form the stack for our 
            stack_DFS.extend(graph[i])
            while stack_DFS:
                j = stack_DFS.pop()
                #We already know it must be true and it hasn't been seen yet
                if j not in seen: 
                   #Add to j to tmp_con_comp
                   tmp_con_comp.add(j)
                   #Add node j to seen
                   seen.add(j)
                   #Extend our stack_DFS with node Js nodal neighbors
                   stack_DFS.extend(graph[j])
            #We can now add the temporary connected component set to our 
            #connected components list
            con_comp.append(tmp_con_comp)

    #We can now return all of the connected components in the global 
    #graph/mesh
    return con_comp

def tf_sets(logical_array):
    '''
        A simple helper function that takes in a numpy logical array and
        turns it into two sets. One set corresponds to all of the indices
        in the logical array that are true. The other set corresponds to
        all of the indices in the logical array that are false.
        
        Input: logical_array - a numpy logical array
        Output: tr_set - a set of all the indices that correspond to true
                        values in logical_array
                f_set - a set of all the indices that correspond to false
                        values in logical_array
    '''
    
    tr_set = set()
    f_set = set()
    
    i = 0
    
    for log in logical_array:
        if log:
            tr_set.add(i)
        else:
            f_set.add(i)
        i += 1
        
    return (tr_set, f_set)
    
def global_conn_comps(conn_comps):
    '''
        Takes in a list of global conn_comps and then outputs a 
    
    '''