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
            #Go ahead and add that node to seen 
            seen.add(i)
            #We now create an empty set for our connected components
            tmp_con_comp = set()
            #Go ahead and add the node we are starting off with
            tmp_con_comp.add(i)
            #Form the stack for our  stack_DFS with node I's nodal neighbors
            stack_DFS.extend(graph[i])
            while stack_DFS:
                j = stack_DFS.pop()
                #We already know it must be true and it hasn't been seen yet
                if j not in seen: 
                   #Add to j to tmp_con_comp
                   tmp_con_comp.add(j)
                   #Add node j to seen
                   seen.add(j)
                   #Extend our stack_DFS with node J's nodal neighbors
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
    
def global_conn_comps_rss(conn_comps, gb_inter_rss, gb_elem_conn, grains):
    '''
        This file takes in the connected component dictionary created earlier.
        It then using the conditions set by the resolved shear stress slip boundary
        criterions to create the global connected component set. It therefore
        requires the grain boundary connectivity array and the grains associated
        with each element. From this data it will construct a list of sets that contain
        tuples that says what grain number our set is from, the slip system number of that set,
        and finally the set from that slip system that has our connected components.
        Initially, the list contains all of the connected components in it before we've started
        combining sets. We'll go through our gb_inter_rss and use that to determine if any of our sets are
        connected. The nice thing here is we only have to check one node from each grain element to see if
        we have any sets that can be combined. If they are connected we go and find where our
        two tuples are located. We combine the sets that both belong to and replace the first set we came
        across with the new set. We delete the latter set from our list. By constructing everything this way,
        when we finish going through all of the grain boundary interaction list we will have our final list
        of global connected components.
        Input:
            conn_comps - a dictionary of dictionarys that has the following keys:
                        the first key is our grain number and the second key is our slip system number.
                        In our innermost dictionary we store a list of sets of all of the connected components associated
                        with that slip system for a particular grain.
            gb_inter_rss -  A similar structure to gr_inter_list. 
                            A dictionary that contains all of the possible GB element
                            interactions. The keys are a tuple of (GB_e1, GB_e2) where
                            GB_e1 and GB_e2 are the sorted grain elements. 
                            The contents of the dictionary at a specific key are a list
                            with two numpy arrays. The first array contains the permutations
                            of all of the xtal_sn indices. It also contains which slip systems have the
                            highest resolved shear stress for that slip normal. The order goes perms and then
                            corresponds ss num for GB_e1 and GB_e2 respectively. The second array is a numpy
                            boolean array that tells us if slip transferal is even possible there.
                            This dictionary will have to be recreated at each simulation step due to there
                            being new stress values. The nice thing it also will tell us what the 
                            structure of our global connected component list will look like.
            gb_elem_conn - The element connectivity array for each grain boundary
                           element. It allows for the reconstruction of the triangular
                           prism elements. It is given as a numpy array. The last
                           two elements in the array tell us what elements correspond
                           to that particular connectivity array. It has dimensions of 14x#GB elements
            grains - The grain number that each element corresponds to. It is represented as 
                     a 1D numpy int array.
        Output - 
            gl_conn_comp_rss_list - A list of sets that contain tuples with the following info: grain number our
                                    set is from, the slip system number of that set, and finally the set from 
                                    that slip system that has our connected components. It is our global connected
                                    component list that corresponds with the resolved shear stress GB slip transferal
                                    requirements.
    '''
    
    
    gl_conn_comp_rss_list = list()
    #Create the initial structure of gl_conn_comp_list
    for ngrains in conn_comps:
        for nss in conn_comps[ngrains]:
            nelems = len(conn_comps[ngrains][nss])
            if nelems > 0:
                for i in range(nelems):
                    temp_t = tuple([ngrains, nss, i])
                    #Need to create an empty set first and then we can
                    #add our tuple to it. If we try and add this directly
                    #We'll just get the objects in the set.
                    temp_set = set()
                    temp_set.add(temp_t)
                    gl_conn_comp_rss_list.append(temp_set.copy())
                    
    nelems = gb_elem_conn.shape[1]
    
    for ielem in range(nelems):
        elems = np.squeeze(gb_elem_conn[12:14, ielem])       
        elem_list = sorted(elems.tolist())
        #We'll use this for our first index to our conn_comp
        grns = grains[elem_list]
        #The key for gb_inter_rss
        keydict = tuple(elem_list)
        data = gb_inter_rss[keydict]
        #Grab our boolean array that tells us what we need to look at
        bool_arr = data[1]
        #Only need to look at the below so now we know what ss we need
        #We might end up with an empty array here
        interact = data[0][bool_arr, 2:4]
        #We'll be able to from here figure out if we have an empty array or not
        nss_exam = interact.shape[0]
        
        #Grab our nodes to check if our sets are connected or not later on
        if(elems[0] == elem_list[0]):
            #We actually needed all 6 nodes for each element
            nd0 = gb_elem_conn[0, ielem]
            nd1 = gb_elem_conn[1, ielem]
            nd2 = gb_elem_conn[2, ielem]
            nd6 = gb_elem_conn[6, ielem]
            nd7 = gb_elem_conn[7, ielem]
            nd8 = gb_elem_conn[8, ielem]
            
            nd3 = gb_elem_conn[3, ielem]
            nd4 = gb_elem_conn[4, ielem]
            nd5 = gb_elem_conn[5, ielem]
            nd9 = gb_elem_conn[9, ielem]
            nd10 = gb_elem_conn[10, ielem] 
            nd11 = gb_elem_conn[11, ielem]
            
        else:
            #We actually needed all 6 nodes for each element
            nd3 = gb_elem_conn[0, ielem]
            nd4 = gb_elem_conn[1, ielem]
            nd5 = gb_elem_conn[2, ielem]
            nd9 = gb_elem_conn[6, ielem]
            nd10 = gb_elem_conn[7, ielem]
            nd11 = gb_elem_conn[8, ielem]
            
            nd0 = gb_elem_conn[3, ielem]
            nd1 = gb_elem_conn[4, ielem]
            nd2 = gb_elem_conn[5, ielem]
            nd6 = gb_elem_conn[9, ielem]
            nd7 = gb_elem_conn[10, ielem] 
            nd8 = gb_elem_conn[11, ielem]
        
        for i in range(nss_exam):
            #We can now use these to check if they are in our sets or not
            len1 = len(conn_comps[grns[0]][interact[i, 0]])
            len2 = len(conn_comps[grns[1]][interact[i, 1]])
            #Want to make sure that both sets are greater than 0 or we shouldn't bother doing anything
            if (len1 > 0) & (len2 > 0):
                #Setting flags to assume no element is any of our sets
                flag1 = False
                flag2 = False
                #Testing the first set
                tmp = conn_comps[grns[0]][interact[i, 0]]
                nsets = len(tmp)
                for j in range(nsets):
                    #If we found our node in a set we update our flag to true.
                    #Then we create our tuple to search our sets, and finally
                    #we can just exit the loop.
                    surf_elem_test = (nd0 in tmp[j]) | (nd1 in tmp[j]) | (nd2 in tmp[j])
                    surf_elem_test = surf_elem_test | (nd6 in tmp[j]) | (nd7 in tmp[j]) | (nd8 in tmp[j])
                    if surf_elem_test:
                        flag1 = True
                        tup1 = tuple([grns[0], interact[i, 0], j])
                        break
                #Now testing second sets
                tmp = conn_comps[grns[1]][interact[i, 1]]
                nsets = len(tmp)
                for j in range(nsets):
                    #Temporary logical variable
                    surf_elem_test = (nd3 in tmp[j]) | (nd4 in tmp[j]) | (nd5 in tmp[j])
                    surf_elem_test = surf_elem_test | (nd9 in tmp[j]) | (nd10 in tmp[j]) | (nd11 in tmp[j])
                    if surf_elem_test:
                        #If we found our node in a set we update our flag to true.
                        #Then we create our tuple to search our sets, and finally
                        #we can just exit the loop.
                        flag2 = True
                        tup2 = tuple([grns[1], interact[i, 1], j])
                        break
                #If we had the nodes in both sets then we can find where our conn comp are in our
                #large list
                if flag1 & flag2:
                    ind1, ind2 = find_gl_conn_comp_list_loc(gl_conn_comp_rss_list, tup1, tup2)
                    #Need to make sure they aren't in the same set if they are we don't do anything
                    if ind1 != ind2:
                        #Update our set in place
                        gl_conn_comp_rss_list[ind1].update(gl_conn_comp_rss_list[ind2])
                        del gl_conn_comp_rss_list[ind2]
        
        
    
    return gl_conn_comp_rss_list


def find_gl_conn_comp_list_loc(gl_conn_comp_list, tup1, tup2):
    '''
    Helper function to find where our tuples are located in our huge global connected component list.
    The returned indices ind1 and ind2 are sorted so ind1 <= ind2.
    Input:
        gl_conn_comp_list - a list of sets with tuple elements
        tup1 - One tuple to find in our list of sets
        tup2 - Another tuple to find in our list of sets
    Output:
        ind1 - the minimum location of our tuples in the list
        ind2 - the maximum location of our tuples in the list
    '''
    
    ind1 = 0
    ind2 = 0
    
    nelems = len(gl_conn_comp_list)
    
    #We're just going to do one loop
    flag1 = False
    flag2 = False
    
    for i in range(nelems):
        if tup1 in gl_conn_comp_list[i]:
            ind1 = i
            flag2 = True
            
        if tup2 in gl_conn_comp_list[i]:
            ind2 = i
            flag2 = True
            
        if flag1 & flag2:
            indices = sorted([ind1, ind2])
            ind1 = indices[0]
            ind2 = indices[1]
            break            
        
    return (ind1, ind2)
    