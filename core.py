import networkx as nx
import numpy as np
from networkx.algorithms import bipartite
from networkx.algorithms.isomorphism import GraphMatcher
from CompilerQC import Graph


def subgraph_is_monomorphic(
        G: nx.classes.graph.Graph,
        H: nx.classes.graph.Graph,
        ):
    """
    G: logischer Graph, where we search for subgraph H
    H: e.g. complete bipartite Graph
    """
    GM = GraphMatcher(G, H)
    return GM.subgraph_is_monomorphic()


def complete_bipartite_graph(
        n1: int,
        n2: int,
        ):
    return nx.complete_bipartite_graph(n1, n2)


def largest_complete_bipartite_graph(
        G: Graph,
        ):
    """
    input: logical graph G
    return: index (x,y) of the largest complete bipartite
    graph K_(x,y)
    """
    tuples = [(i, j) for i in range(1, G.N-1)
            for j in range(1, G.N-1) if i <= j]
    G = G.to_nx_graph()
    largest = 1
    largest_tuple = (1, 1)
    for n1, n2 in tuples:
        K_nn = complete_bipartite_graph(n1, n2)
        if (
                subgraph_is_monomorphic(G, K_nn)
                and n1 * n2 > largest
            ):
            largest = n1 * n2
            largest_tuple = (n1, n2)
    return largest_tuple


def parts_of_complete_bipartite_graph(
        G: nx.classes.graph.Graph,
        K_nn: nx.classes.graph.Graph,
        ):
    """
    input: logical Graph G and largest complete
    bipartite graph K_nn
    return: the two sets (contain the logical bits)
    of the complete bipartite 
    graph in the logical graph
    """
    # relabel nodes in bipartite graph to logical graph nodes
    GM = GraphMatcher(G, K_nn)
    GM.subgraph_is_monomorphic()
    reversed_mapping = {v:k for k,v in GM.mapping.items()}
    K_nn = nx.relabel_nodes(K_nn, reversed_mapping)
    # parts of bipartite graph
    U, V = bipartite.sets(K_nn)
    U, V = list(U), list(V)
    return U, V


def qbits_and_coords_of_core(
        U: list,
        V: list,
        ):
    """
    input: two sets of logical bits, generating the
    complete bipartite graph
    return: list of qbits and their coords
    which build the core of the physical graph
    """
    # coords
    x, y = np.meshgrid(np.arange(len(U)), np.arange(len(V)))
    coords = list(zip(x.flatten(), y.flatten()))
    # qbits
    lbit1, lbit2 = np.meshgrid(U, V)
    qbits = list(zip(lbit1.flatten(), lbit2.flatten()))
    return list(map(tuple, map(sorted, qbits))), coords
