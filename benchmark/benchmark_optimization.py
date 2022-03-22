from functions_for_benchmarking import update_mc, check_update, evaluate_optimization, run_benchmark, head
from tqdm import tqdm
from CompilerQC import Graph, Qbits, Polygons, Energy, MC, paths, core
import numpy as np
import pickle

def benchmark_problem_folder_with_exact_scaling(args):
    """
    calculate the scaling for each 3er and 4er plaq such that 
    the total energy of the compiled solution is zero,
    if the same compiled solution is found as in the dataset
    """
    problems = get_files_to_problems(
                problem_folder=args.problem_folder,
                min_C=args.min_C,
                max_C=args.max_C,
                )
    print(head)
    for file in problems:
        # read graph and qubit to coord translation from file
        graph_adj_matrix, qubit_coord_dict = (
            problem_from_file(file))
        # scopes for nonplaqs and plaqs
        polygon_scopes, NKC, n_cycles = energy_from_problem(graph_adj_matrix, qubit_coord_dict)
        n3, n4, p3, p4 = polygon_scopes
        graph = Graph(adj_matrix=graph_adj_matrix)
        # contribution of nonplaqs to each constraint 
        scaling_for_plaq3 = sum(n3) / len(p3)
        scaling_for_plaq4 = sum(n4) / len(p4)
        # initialise energy_object
        qbits = Qbits.init_qbits_from_dict(graph,dict())
        polygon_object = Polygons(qbits=qbits)
        energy = Energy(
            polygon_object,
            scaling_for_plaq3=scaling_for_plaq3,
            scaling_for_plaq4=scaling_for_plaq4,
        )
        run_benchmark(energy, args.batchsize, args.id_of_benchmark)
        
def benchmark_energy_scaling_by_yaml(args):
    """
    benchmark the search of compiled graphs for 
    fully connected logical graphs, 
    scale plaqs by value in the yaml
    """
    print(head)
    path_to_results = (
        paths.benchmark_results_path / f"mc_benchmark_results_{args.id_of_benchmark}.txt"
    )
    file = open(path_to_results, "a")
    file.write(head)
    for N in range(4, args.N +1):
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
        file = run_benchmark(file, energy, args.batchsize, args.id_of_benchmark)
    file.close()
       
def benchmark_MLP_energy_scaling(args):
    """
    benchmark the search of compiled graphs for 
    fully connected logical graphs,
    scale plaqs by prediction of MLP
    """
    print(head)
    for N in range(4, args.N +1):
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
        run_benchmark(energy, args.batchsize, args.id_of_benchmark)
        
def benchmark_energy_scaling_by_max_C(args):
    """
    benchmark the search of compiled graphs for 
    fully connected logical graphs,
    scale by prediction of polynom fittet
    to max_C:energy dataset
    """
    print(head)
    for N in range(4, args.N +1):
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
        run_benchmark(energy, args.batchsize, args.id_of_benchmark)

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
        "--batchsize",
        type=int,
        default=100,
        help="how many times each schedule is evaluated default is 100",
    )
    parser.add_argument(
        "-id",
        "--id_of_benchmark",
        type=int,
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
        "-N",
        "--N",
        type=int,
        default=1,
        help="benchmark_LHZ up to N",
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
        type=bool,
        default=False,
        help="start with bipartite core",
    )
    args = parser.parse_args()
    benchmark_problem_folder(args)

