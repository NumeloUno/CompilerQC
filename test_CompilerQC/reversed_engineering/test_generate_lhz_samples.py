import numpy as np
from CompilerQC import Graph, Polygons
from CompilerQC.reversed_engineering import generate_lhz_samples

def test_generation_of_lhz_graphs():
    for N in range(4, 9):
        adj_matrix, qbit_coord_dict = generate_lhz_samples.generate_LHZ_problem(N)
        graph_ = Graph(adj_matrix=adj_matrix)
        polygon_object_ = Polygons(graph_, qbit_coord_dict=qbit_coord_dict)
        assert set(polygon_object_.qbit_coord_dict.keys()) == set(graph_.qbits_from_graph())

