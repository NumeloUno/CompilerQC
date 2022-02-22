import numpy as np
import argparse
import random
from CompilerQC import Graph, paths
from CompilerQC import Polygons
from CompilerQC import Energy
import os 
from scipy.optimize import dual_annealing


def problem_from_file(path_to_problem: str):
    """
    read the path and return adj matrix and the qbit 
    to coord assignment
    """
    read_dictionary = np.load(path_to_problem, allow_pickle='TRUE').item()
    return read_dictionary['graph_adj_matrix'], read_dictionary['qbit_coord_dict']

def energy_from_problem(graph_adj_matrix: np.array, qbit_coord_dict: dict):
    """
    input: graph adj matrix and qbit_to_coord dict
    returns the energy of a graph and its qbit to coord 
    translation and the number of logical nodes N
    """
    graph = Graph(adj_matrix=graph_adj_matrix)
    polygon_object = Polygons(graph, qbit_coord_dict=qbit_coord_dict)
    energy = Energy(polygon_object=polygon_object)
    energy(polygon_object=polygon_object) # just to initialize energy.polygon_coords
    return energy.scopes_of_polygons(), [polygon_object.N, polygon_object.K, polygon_object.C]

def get_files_to_problems(problem_folder: str="training_set", max_C: int=50):
    """
    returns all files in problem_folder, if problem has less constraints C than max_C
    """
    filenames = []
    for path in os.listdir(paths.database_path / problem_folder):
        if int(path.split('_')[-1]) <= max_C:
            for filename in os.listdir(paths.database_path  / problem_folder / path):
                filenames.append(paths.database_path / problem_folder / path / filename)
    return filenames

def get_all_distances(problem_folder: str="training_set"):
    """
    returns energy_from_problem() for all problems in folder problem_folder,
    """
    list_of_distances_to_plaqs = list(map(
        lambda filename: energy_from_problem(*problem_from_file(filename)),
        get_files_to_problems(problem_folder)
    ))
    return list_of_distances_to_plaqs



