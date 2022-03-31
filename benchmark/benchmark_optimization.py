from functions_for_benchmarking import update_mc, evaluate_optimization, run_benchmark, head
from CompilerQC import functions_for_database as dataset
from tqdm import tqdm
from CompilerQC import Graph, Qbits, Polygons, Energy, MC, paths, core
import numpy as np
import pickle
import argparse
import os.path

def open_file(args):
    """
    open txt where benchmark results will be saved,
    if file does not exist yet, create it and append head
    """
    print(head)
    path_to_results = (
        paths.benchmark_results_path / f"mc_benchmark_results_{args.id_of_benchmark}_{args.extra_id}.txt"
    )
    if os.path.exists(path_to_results):
        file = open(path_to_results, "a")
    else:
        file = open(path_to_results, "a")
        file.write(head)
    return file

def benchmark_problem_folder_with_exact_scaling(args):
    """
    calculate the scaling for each 3er and 4er plaq such that 
    the total energy of the compiled solution is zero,
    if the same compiled solution is found as in the dataset
    """
    problems = dataset.get_files_to_problems(
                problem_folder=args.problem_folder,
                min_C=args.min_C,
                max_C=args.max_C,
                )
    file = open_file(args)
    for file_ in problems:
        # read graph and qubit to coord translation from file
        graph_adj_matrix, qubit_coord_dict = (
            dataset.problem_from_file(file_))
        # scopes for nonplaqs and plaqs
        polygon_scopes, NKC, n_cycles = dataset.energy_from_problem(graph_adj_matrix, qubit_coord_dict)
        n3, n4, p3, p4 = polygon_scopes
        graph = Graph(adj_matrix=graph_adj_matrix)
        # contribution of nonplaqs to each constraint 
        scaling_for_plaq3 = scaling_for_plaq4 = (sum(n4)+sum(n3)) / graph.C          
        # initialise energy_object
        qbits = Qbits.init_qbits_from_dict(graph, dict())
        polygon_object = Polygons(qbits=qbits)
        energy = Energy(
            polygon_object,
            scaling_for_plaq3=scaling_for_plaq3,
            scaling_for_plaq4=scaling_for_plaq4,
        )
        file = run_benchmark(file, energy, args)
    file.close()
    
def benchmark_energy_scaling_by_yaml(args):
    """
    benchmark the search of compiled graphs for 
    fully connected logical graphs, 
    scale plaqs by value in the yaml
    """
    file = open_file(args)
    for N in range(args.min_N, args.max_N +1):
        graph = Graph.complete(N)
        core_qubits, core_coords = [], []
        if args.with_core:
            #nn = core.largest_complete_bipartite_graph(graph)
            nn = (N // 2, N - (N // 2))
            K_nn = core.complete_bipartite_graph(*nn)
            U, V = core.parts_of_complete_bipartite_graph(graph.to_nx_graph(), K_nn)
            core_qubits, core_coords = core.qbits_and_coords_of_core(U, V)
        # initialise energy_object
        qbits = Qbits.init_qbits_from_dict(graph, dict(zip(core_qubits, core_coords)))
        if args.with_core:
            qbits.assign_core_qbits(core_qubits)
        polygon_object = Polygons(qbits)
        energy = Energy(polygon_object)
        file = run_benchmark(file, energy, args)
    file.close()
       
def benchmark_MLP_energy_scaling(args):
    """
    benchmark the search of compiled graphs for 
    fully connected logical graphs,
    scale plaqs by prediction of MLP
    """
    file = open_file(args)
    for N in range(args.min_N, args.max_N +1):
        graph = Graph.complete(N)
        core_qubits, core_coords = [], []
        if args.with_core:
            #nn = core.largest_complete_bipartite_graph(graph)
            nn = (N // 2, N - (N // 2))
            K_nn = core.complete_bipartite_graph(*nn)
            U, V = core.parts_of_complete_bipartite_graph(graph.to_nx_graph(), K_nn)
            core_qubits, core_coords = core.qbits_and_coords_of_core(U, V)
        loaded_model = pickle.load(open(paths.energy_scalings / 'MLPregr_model.sav', 'rb'))
        predicted_energy = loaded_model.predict([[graph.N, graph.K, graph.C, graph.number_of_3_cycles, graph.number_of_4_cycles]])[0]
        scaling_for_plaq3 = scaling_for_plaq4 = predicted_energy / graph.C  
        # initialise energy_object
        qbits = Qbits.init_qbits_from_dict(graph, dict(zip(core_qubits, core_coords)))
        if args.with_core:
            qbits.assign_core_qbits(core_qubits)
        polygon_object = Polygons(qbits)
        energy = Energy(
            polygon_object,
            scaling_for_plaq3=scaling_for_plaq3,
            scaling_for_plaq4=scaling_for_plaq4,
        )
        file = run_benchmark(file, energy, args)
    file.close()
    
def benchmark_energy_scaling_by_max_C(args):
    """
    benchmark the search of compiled graphs for 
    fully connected logical graphs,
    scale by prediction of polynom fittet
    to max_C:energy dataset
    """
    file = open_file(args)
    for N in range(args.min_N, args.max_N +1):
        graph = Graph.complete(N)
        core_qubits, core_coords = [], []
        if args.with_core:
            #nn = core.largest_complete_bipartite_graph(graph)
            nn = (N // 2, N - (N // 2))
            K_nn = core.complete_bipartite_graph(*nn)
            U, V = core.parts_of_complete_bipartite_graph(graph.to_nx_graph(), K_nn)
            core_qubits, core_coords = core.qbits_and_coords_of_core(U, V)

        poly_coeffs = np.load(paths.energy_scalings / 'energy_max_C_fit.npy')
        poly_function = np.poly1d(poly_coeffs)
        scaling_for_plaq3 = scaling_for_plaq4 = poly_function(graph.C) / graph.C  

        # initialise energy_object
        qbits = Qbits.init_qbits_from_dict(graph, dict(zip(core_qubits, core_coords)))
        if args.with_core:
            qbits.assign_core_qbits(core_qubits)
        polygon_object = Polygons(qbits)
        energy = Energy(
            polygon_object,
            scaling_for_plaq3=scaling_for_plaq3,
            scaling_for_plaq4=scaling_for_plaq4,
        )
        file = run_benchmark(file, energy, args)
    file.close()
    
def benchmark_energy_scaling_by_LHZ_C(args):
    """
    benchmark the search of compiled graphs for 
    fully connected logical graphs,
    scale by prediction of polynom fittet
    to LHZ solutions
    """
    file = open_file(args)
    for N in range(args.min_N, args.max_N +1):
        graph = Graph.complete(N)
        core_qubits, core_coords = [], []
        if args.with_core:
            #nn = core.largest_complete_bipartite_graph(graph)
            nn = (N // 2, N - (N // 2))
            K_nn = core.complete_bipartite_graph(*nn)
            U, V = core.parts_of_complete_bipartite_graph(graph.to_nx_graph(), K_nn)
            core_qubits, core_coords = core.qbits_and_coords_of_core(U, V)

        poly_coeffs = np.load(paths.energy_scalings / 'LHZ_energy_C_fit.npy')
        poly_function = np.poly1d(poly_coeffs)
        scaling_for_plaq3 = scaling_for_plaq4 = poly_function(graph.C) / graph.C  

        # initialise energy_object
        qbits = Qbits.init_qbits_from_dict(graph, dict(zip(core_qubits, core_coords)))
        if args.with_core:
            qbits.assign_core_qbits(core_qubits)
        polygon_object = Polygons(qbits)
        energy = Energy(
            polygon_object,
            scaling_for_plaq3=scaling_for_plaq3,
            scaling_for_plaq4=scaling_for_plaq4,
        )
        file = run_benchmark(file, energy, args)
    file.close()
    
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
        "-id",
        "--id_of_benchmark",
        type=str,
        default="",
        help="save results using id (which is also in filename of mc_parameters.yaml) in filename",
    )
    parser.add_argument(
        "-minC",
        "--min_C",
        type=int,
        default=1,
        help="set the minimum C, for which benchmarks should be done",
    )
    parser.add_argument(
	"-maxN",
        "--max_N",
        type=int,
        default=12,
        help="benchmark_LHZ up to N",
    )
    parser.add_argument(
	"-minN",
        "--min_N",
        type=int,
        default=4,
        help="benchmark_LHZ from minimum N",
    )
    parser.add_argument(
        "-maxC",
        "--max_C",
        type=int,
        default=10,
        help="set the maximum C, for which benchmarks should be done",
    )
    parser.add_argument(
        "-core",
        "--with_core",
        action="store_true",
        default=False,
        help="start with bipartite core",
    )
    parser.add_argument(
        "-ed",
        "--extra_id",
        type=str,
        default="",
        help="some extra name to add to result filename",
    )
    args = parser.parse_args()
    if args.extra_id == 'exact_scaling':
        benchmark_problem_folder_with_exact_scaling(args)
    if args.id_of_benchmark == 'maxCpolynom':
        benchmark_energy_scaling_by_max_C(args)
    if args.id_of_benchmark == 'LHZCpolynom':
        benchmark_energy_scaling_by_LHZ_C(args)
    if args.id_of_benchmark == 'energyscaling':
        benchmark_energy_scaling_by_yaml(args)
    if args.id_of_benchmark == 'nmoves':
        args.extra_id = 'MLP_linear_tempschedule_in_moves'
        benchmark_MLP_energy_scaling(args)
    if args.id_of_benchmark == 'temperature_C':
        args.extra_id = 'MLP'
        benchmark_MLP_energy_scaling(args)
    if args.id_of_benchmark == 'temperature_linearC':
        args.extra_id = 'MLP'
        benchmark_MLP_energy_scaling(args)
    if args.id_of_benchmark == 'temperature_kirkpatrick_sigma':
        args.extra_id = 'MLP'
        benchmark_MLP_energy_scaling(args)
    if args.id_of_benchmark == 'MLP':
        benchmark_MLP_energy_scaling(args)
    if args.id_of_benchmark == 'temp_schedule2':
        args.extra_id = 'MLP'
        benchmark_MLP_energy_scaling(args)
    if args.id_of_benchmark == 'delta':
        args.extra_id = 'MLP'
        benchmark_MLP_energy_scaling(args)
    if args.id_of_benchmark == 'chi_0':
        args.extra_id = 'MLP'
        benchmark_MLP_energy_scaling(args)    
    if args.extra_id == 'MLP':
        benchmark_MLP_energy_scaling(args)  
    else:
        print("id not found in parameters folder")
