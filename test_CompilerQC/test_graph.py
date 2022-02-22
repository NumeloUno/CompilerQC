from CompilerQC import Graph
from scipy.special import binom

def get_3_cycles_for_complete_graph(N: int):
    graph = Graph.complete(N)
    return len(graph.get_cycles(3))

def get_4_cycles_for_complete_graph(N: int):
    graph = Graph.complete(N)
    return len(graph.get_cycles(4))

def test_number_of_cycles_for_complete_graph():
    """
    check if the number of closed 3er and 4er cycles
    agree with expected number
    """
    for N in range(4, 10):
        assert binom(N, 3) == get_3_cycles_for_complete_graph(N)
        assert binom(N, 4) * 3 == get_4_cycles_for_complete_graph(N)

