from CompilerQC import Graph, core

def test_find_max_bipartite_core():
    """
    check if the bipartite cores in the even complete
    graphs 4-10 agree with the size 1x1, 2x2,...
    """
    for N in [4, 6]:
        graph = Graph.complete(N)
        assert (
                core.largest_complete_bipartite_graph(graph)
                ==((N - 4) / 2 + 2,  (N - 4) / 2 + 2)  
                )
