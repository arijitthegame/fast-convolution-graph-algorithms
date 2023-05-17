# This is the implementation of Spanberger's algorithm in https://arxiv.org/pdf/2004.01163.pdf

import networkx as nx
import numpy as np
from numpy import linalg

def spanberger_sp_bf(G, i,j):
    """
    G : networkx graph 
    j := starting vertex
    i := end vertex
    """
    L = nx.laplacian_matrix(G).toarray()
    Li = np.delete(np.delete(L, i, 0), i, 1)
    lambda_i, eigh_vector_i = linalg.eigh(Li)
    smallest_entry = lambda_i.argmin()
    eigh_vector_i = eigh_vector_i[:, smallest_entry]
    eigh_vector_i = np.abs(eigh_vector_i)
    eigh_vector_i = eigh_vector_i.flatten()
    extended_vec = np.insert(eigh_vector_i,i, 0)

    # find path 
    vertices_traversed = [j]
    x0 = j 
    while x0 != i :
        all_neighbors = [n for n in G.neighbors(x0)]
        all_neighbors_not_traversed = [x for x in all_neighbors if x not in vertices_traversed] # indices hack added to get rid of self loops
        dict_subset_nodes_eigs = dict(zip(all_neighbors_not_traversed, extended_vec[all_neighbors_not_traversed]))
        # get the relevant eigenvalue and the node 
        # caveat only 1 min
        minval = min(dict_subset_nodes_eigs.values())
        res = [k for k, v in dict_subset_nodes_eigs.items() if v==minval]
        # hack take the first value
        x0 = res[0]
        vertices_traversed.append(x0)
    return vertices_traversed

if __name__ == "__main__":
    import time
    H = nx.random_tree(100, seed=42)
    print("Is H connected?", nx.is_connected(H))
    start = time.time()
    p1 = spanberger_sp_bf(H, 1, 37)
    end = time.time()
    print(f"Total time taken by Spanberger's algorithm is {start-end}")
    # compare with Dijstra
    start = time.time() 
    p0 = nx.shortest_path(H,37,1)
    end = time.time()
    print(f"Total time taken by Dijstra algorithm is {start-end}")
    # compare paths
    print(f"The path for Spanberger is {p1}")
    print(f"The path for Dijstra is {p0}")
    if p1 == p0 : 
        print("The paths match perfectly!")


    
