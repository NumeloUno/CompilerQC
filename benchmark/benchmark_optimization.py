#from functions_for_benchmarking import *
from tqdm import tqdm
from CompilerQC import *
import numpy as np
import pickle
import argparse
import os.path
import pandas as pd

path_to_results = lambda args: (
        paths.benchmark_results_path / f"mc_benchmark_results_{args.id_of_benchmark}_{args.extra_id}.csv"
    )
    
def graphs_to_benchmark(args):
    """
    given a folder with problems,
    return a list of adjacency matrices of these 
    logical problems
    """
    problems = functions_for_database.get_files_to_problems(
                problem_folder=args.problem_folder,
                min_C=args.min_C,
                max_C=args.max_C,
                )
    graphs = []
    for file in problems:
        # read graph and qubit to coord translation from file
        graph_adj_matrix, qubit_coord_dict = (
            functions_for_database.problem_from_file(file))
        graphs.append(graph_adj_matrix)
        if len(graphs) == args.max_size:
            break
    return graphs
        
def benchmark_energy_scaling(args):
    """
    benchmark the search of compiled graphs for 
    fully connected logical graphs,
    scale plaqs by prediction of MLP
    """
    benchmark_df = pd.DataFrame()
        
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=paths.logger_path / f"{args.id_of_benchmark}.log", mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    logger.info("=================================================================")
    logger.info(f"benchmark {args.max_size} problems from {args.problem_folder} folder")
    logger.info("=================================================================")

    for adj_matrix in graphs_to_benchmark(args):
        graph = Graph(adj_matrix = adj_matrix)
        benchmark_df = functions_for_benchmarking.run_benchmark(benchmark_df, graph, args, logger)
    benchmark_df.to_csv(path_to_results(args), mode='a', header=True, index=False)
   
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Benchmark the settings for MC in the mc_parameters.yaml and save evaluation in mc_benchmark_results.txt"
    )
    parser.add_argument(
        "-path",
        "--problem_folder",
        type=str,
        default='training_set',
        help="path of problems to benchmark: lhz or training_set (default)"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=100,
        help="how many times each schedule is evaluated default is 100",
    )
    parser.add_argument(
        "-yp",
        "--yaml_path",
        type=str,
        default="",
        help="get config from yaml_path",
    )
    parser.add_argument(
        "-maxC",
        "--max_C",
        type=int,
        default=10,
        help="set the maximum C, for which benchmarks should be done",
    )
    parser.add_argument(
        "-minC",
        "--min_C",
        type=int,
        default=10,
        help="set the minimum C, for which benchmarks should be done",
    )
    parser.add_argument(
	"-maxN",
        "--max_N",
        type=int,
        default=10,
        help="benchmark_LHZ up to N",
    )
    parser.add_argument(
	"-minN",
        "--min_N",
        type=int,
        default=8,
        help="benchmark_LHZ from minimum N",
    )
    parser.add_argument(
        "-ed",
        "--extra_id",
        type=str,
        default="MLP",
        help="some extra name to add to result filename",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=100,
        help="number of different problems to benchmark on",
    )
    args = parser.parse_args()
    
    # save results using id (which is also in filename of mc_parameters.yaml) in filename
    args.id_of_benchmark = args.yaml_path.split('_')[2]
    benchmark_energy_scaling(args)
        
# def benchmark_problem_folder_with_exact_scaling(args):
#     """
#     calculate the scaling for each 3er and 4er plaq such that 
#     the total energy of the compiled solution is zero,
#     if the same compiled solution is found as in the dataset
#     """
    
#     problems = dataset.get_files_to_problems(
#                 problem_folder=args.problem_folder,
#                 min_C=args.min_C,
#                 max_C=args.max_C,
#                 )
    
#     benchmark_df = pd.DataFrame()
#     for ar in problems:
#         # read graph and qubit to coord translation from file
#         graph_adj_matrix, qubit_coord_dict = (
#             dataset.problem_from_file(file_))
#         # scopes for nonplaqs and plaqs
#         polygon_scopes, NKC, n_cycles = dataset.energy_from_problem(graph_adj_matrix, qubit_coord_dict)
#         n3, n4, p3, p4 = polygon_scopes
#         graph = Graph(adj_matrix=graph_adj_matrix)
#         # contribution of nonplaqs to each constraint 
#         scaling_for_plaq3 = scaling_for_plaq4 = (sum(n4)+sum(n3)) / graph.C          
#         # initialise energy_object
#         qbits = Qbits.init_qbits_from_dict(graph, dict())
#         polygon_object = Polygons(qbits=qbits)
#         energy = Energy(
#             polygon_object,
#             scaling_for_plaq3=scaling_for_plaq3,
#             scaling_for_plaq4=scaling_for_plaq4,
#         )
#         benchmark_df = run_benchmark(benchmark_df, energy, args)
#     benchmark_df.to_csv(path_to_results(args), mode='a', header=True, index=False)
#   def from_graph_to_polygon_object(graph: Graph, with_core: bool=None):
#     """returns a polygon_object"""
#     core_qubits, core_coords = [], []
#     if with_core:
#         #nn = core.largest_complete_bipartite_graph(graph)
#         nn = (graph.N // 2, graph.N - (graph.N // 2))
#         K_nn = core.complete_bipartite_graph(*nn)
#         U, V = core.parts_of_complete_bipartite_graph(graph.to_nx_graph(), K_nn)
#         core_qubits, core_coords = core.qbits_and_coords_of_core(U, V)
#     # initialise energy_object
#     qbits = Qbits.init_qbits_from_dict(graph, dict(zip(core_qubits, core_coords)))
#     if with_core:
#         qbits.assign_core_qbits(core_qubits)
#     polygon_object = Polygons(qbits)
#     return polygon_object

# def load_scaling_from_model(graph, args):
#     """load scaling for plaquettes 
#     from different models"""
#     if args.extra_id == 'MLP':
#         loaded_model = pickle.load(open(paths.energy_scalings / 'MLPregr_model.sav', 'rb'))
#         predicted_energy = loaded_model.predict([[graph.N, graph.K, graph.C, graph.number_of_3_cycles, graph.number_of_4_cycles]])[0]
#         scaling_for_plaq3 = scaling_for_plaq4 = predicted_energy / graph.C  
#         return scaling_for_plaq3, scaling_for_plaq4
    
#     if args.extra_id == 'maxC':
#         poly_coeffs = np.load(paths.energy_scalings / 'energy_max_C_fit.npy')
#         poly_function = np.poly1d(poly_coeffs)
#         scaling_for_plaq3 = scaling_for_plaq4 = poly_function(graph.C) / graph.C  
#         return scaling_for_plaq3, scaling_for_plaq4

#     if args.extra_id == 'LHZ':
#         poly_coeffs = np.load(paths.energy_scalings / 'LHZ_energy_C_fit.npy')
#         poly_function = np.poly1d(poly_coeffs)
#         scaling_for_plaq3 = scaling_for_plaq4 = poly_function(graph.C) / graph.C  
#         return scaling_for_plaq3, scaling_for_plaq4
    
#     if args.extra_id == 'yamlscaling':
#         return None, None # will be set by yaml
    
#     else: print('no model for energy scaling selected')  
