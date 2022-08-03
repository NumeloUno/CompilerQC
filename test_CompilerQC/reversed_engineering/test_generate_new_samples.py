import numpy as np
import random
from CompilerQC import Graph, Polygons
from CompilerQC.reversed_engineering import generate_new_samples


def test_problem_generation():
    """
    check if problem generation works properly
    """
    for sample in range(10):
        square_plaquette_probability = random.uniform(0.2, 1)
        output = generate_new_samples.create_problem(10, square_plaquette_probability)
        adj_matrix, qbit_coord_dict = generate_new_samples.translate_and_save_problem(
            output, save=False
        )
        graph_ = Graph(adj_matrix=adj_matrix)
        polygon_object_ = Polygons(graph_, qbit_coord_dict=qbit_coord_dict)

        assert set(polygon_object_.qbit_coord_dict.keys()) == set(
            graph_.qbits_from_graph()
        )
