import numpy as np
import argparse
import os
import random
from CompilerQC import Graph, Qbits, Polygons, Energy, paths


def problem_from_file(path_to_problem: str):
    """
    read the path and return adj matrix and the qbit
    to coord assignment
    """
    read_dictionary = np.load(path_to_problem, allow_pickle="TRUE").item()
    return read_dictionary["graph_adj_matrix"], read_dictionary["qbit_coord_dict"]


# TODO: make this function more flexible, if one is for example interested in energy without qbits in core
def energy_from_problem(graph_adj_matrix: np.array, qubit_coord_dict: dict):
    """
    input: graph adj matrix and qubit_to_coord dict
    returns the scopes of nonplaqs3, nonplaqs4, plaqs3, plaqs4 of a graph and its qbit to coord
    translation and the number of logical nodes N, qbits K and constraints C
    """
    graph = Graph(adj_matrix=graph_adj_matrix)
    qbits = Qbits.init_qbits_from_dict(graph, qubit_coord_dict)
    polygon_object = Polygons(qbits)
    energy = Energy(polygon_object=polygon_object)
    return (
        energy.scopes_of_polygons_for_analysis(),
        [graph.N, graph.K, graph.C],
        [graph.number_of_3_cycles, graph.number_of_4_cycles],
    )


def get_files_to_problems(
    problem_folder: str = "training_set", min_C: int = 1, max_C: int = 50
):
    """
    returns all files in problem_folder, if problem has constraints C between min_C and max_C
    """
    filenames = []
    for path in os.listdir(paths.database_path / problem_folder):
        if min_C <= int(path.split("_")[-1]) <= max_C:
            for filename in os.listdir(paths.database_path / problem_folder / path):
                filenames.append(paths.database_path / problem_folder / path / filename)
    return filenames


def get_all_distances(
    problem_folder: str = "training_set", min_C: int = 1, max_C: int = 50
):
    """
    returns energy_from_problem() for all problems in folder problem_folder,
    """
    list_of_distances_to_plaqs = list(
        map(
            lambda filename: energy_from_problem(*problem_from_file(filename)),
            get_files_to_problems(problem_folder),
        )
    )
    return list_of_distances_to_plaqs
