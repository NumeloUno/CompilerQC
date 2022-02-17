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
    returns the energy of a given qbit to coord 
    translation for each polygon and the number of logical nodes N
    """
    graph = Graph(adj_matrix=graph_adj_matrix)
    polygon_object = Polygons(graph, qbit_coord_dict=qbit_coord_dict)
    energy = Energy(polygon_object=polygon_object)
    energy(polygon_object=polygon_object, terms=[0,1,0])
    return energy.arbitrary_scaled_distance_to_plaqutte(), energy.polygon.N 

def get_files_to_problems(problem_folder: str="training_set"):
    """
    returns all files in problem_folder
    """
    filenames = []
    for path in os.listdir(paths.database_path / problem_folder):
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


def scale(plaqs_and_N, params):
    plaqs, N = plaqs_and_N
    (
    a_n3,
    b_n3,
    c_n3,
    d_n3,
    a_n4,
    b_n4,
    c_n4,
    d_n4,
    ) = params
    non_plaqs_3, non_plaqs_4, plaqs_3, plaqs_4 = plaqs
    return (
        (np.array(plaqs_3)
          - (a_n3 * N ** 3 * b_n3 * N ** 2 + c_n3 * N ** 1  + d_n3)).sum()
        + (np.array(plaqs_4)
            - (a_n4 * N ** 3 * b_n4 * N ** 2 + c_n4 * N ** 1  + d_n4)).sum()
        + (np.array(non_plaqs_3)).sum()
        + (np.array(non_plaqs_4)).sum()
    ) ** 2

            
#  TODO: description is weak
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit polynom to problems in problem"
    )
    parser.add_argument(
        "-bounds",
        type=list,
        default=[-2,2],
        help="bounds for fitting",
    )    
    parser.add_argument(
        "-path",
        type=str,
        default="training_set",
        help="take samples from folder training_set, validation_set, lhz",
    )    
    args = parser.parse_args()
    list_of_distances_to_plaqs = get_all_distances()
    lw, up = args.bounds
    lw = [lw] * 8
    up = [up] * 8
    func = lambda params: (sum(list(map(lambda x: scale(x, params), list_of_distances_to_plaqs))))
    ret = dual_annealing(func, bounds=list(zip(lw, up)))

    print(func(ret.x), ret.x)
