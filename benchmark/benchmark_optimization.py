# from functions_for_benchmarking import *
from tqdm import tqdm
from CompilerQC import *
import numpy as np
import argparse
import pandas as pd
from pathlib import Path

path_to_results = lambda args: (
    paths.benchmark_results_path / f"run_{args.run}"
    / f"benchmark_{(args.problem_folder).replace('/','_')}_with_{args.id_of_benchmark}"
)

def graphs_to_benchmark(args):
    """
    given a folder with problems,
    return a list of adjacency matrices of these
    logical problems
    """
    problems = functions_for_database.uniform_sample_from_folder(
        problem_folder=args.problem_folder,
        min_C=args.min_C,
        max_C=args.max_C,
        min_N=args.min_N,
        max_N=args.max_N,
        max_size=args.max_size,
    )
    graphs, qubit_to_coord_dicts = [], []
    for file in problems:
        # read graph and qubit to coord translation from file
        graph_adj_matrix, qubit_coord_dict = functions_for_database.problem_from_file(
            file
        )
        graphs.append(graph_adj_matrix)
        qubit_to_coord_dicts.append(qubit_coord_dict)
    return (graphs, qubit_to_coord_dicts)


def benchmark_energy_scaling(args):
    """
    benchmark the search of compiled graphs for
    fully connected logical graphs,
    scale plaqs by prediction of MLP
    """
    list_of_graphs_to_benchmark = graphs_to_benchmark(args)[0]
    Path(path_to_results(args) / f'{args.name}').mkdir(parents=True, exist_ok=True)
    for adj_matrix in list_of_graphs_to_benchmark:
        graph = Graph(adj_matrix=adj_matrix)
        functions_for_benchmarking.run_benchmark(graph, args)

def visualize_benchmarks(args):
    """
    visualize the search of compiled graphs for
    fully connected logical graphs,
    scale plaqs by prediction of MLP
    """
    for adj_matrix in graphs_to_benchmark(args)[0]:
        graph = Graph(adj_matrix=adj_matrix)
        functions_for_benchmarking.visualize_settings(graph, args)

def CNOT_ratio_before_compilation(args):
    """
    """
    df = pd.DataFrame()
    list_of_graphs_to_benchmark = graphs_to_benchmark(args)[0]
    for adj_matrix in list_of_graphs_to_benchmark:
        graph = Graph(adj_matrix=adj_matrix)
        for _ in range(args.batch_size):
            qbits = Qbits.init_qbits_from_dict(graph, dict())
            nodes = Nodes(qbits, place_qbits_in_lines=False)
            polygons = Polygons(nodes)
            energy = Energy(polygons)
            mc = MC(energy)
            number_of_CNOTs, number_of_SWAPs,_,_ = mc.number_of_CNOTs()
            CNOT_ratio = number_of_CNOTs / MC.number_of_CNOTs_in_LHZ(graph.N)
            df = df.append(
                {
                    'N':graph.N,
                    'K':graph.K,
                    'C':graph.C,
                    'CNOT_ratio':CNOT_ratio,
                    'qubit_ratio':graph.K / (graph.N / 2 * (graph.N - 1)),
                    'number_of_swap_gates': number_of_SWAPs,
                }
                , ignore_index=True)
    df.to_csv(paths.cwd / f"benchmark/plot_scripts/CNOT_ratio_of_{args.problem_folder}_before_compilation.csv", index=False)

def CNOT_ratio_of_compiled_solutions(args):
    """
    """
    df = pd.DataFrame()
    a, b = graphs_to_benchmark(args)
    for adj_matrix, qubit_to_coord in zip(a, b):
        graph = Graph(adj_matrix=adj_matrix)
        qbits = Qbits.init_qbits_from_dict(graph, qubit_to_coord)
        nodes = Nodes(qbits, place_qbits_in_lines=False)
        polygons = Polygons(nodes)
        energy = Energy(polygons)
        mc = MC(energy)
        number_of_CNOTs, number_of_SWAPs,_,_ = mc.number_of_CNOTs()
        CNOT_ratio = number_of_CNOTs / MC.number_of_CNOTs_in_LHZ(graph.N)
        df = df.append(
            {
                'N':graph.N,
                'K':graph.K,
                'C':graph.C,
                'CNOT_ratio':CNOT_ratio,
                'qubit_ratio':graph.K / (graph.N / 2 * (graph.N - 1)),
                'number_of_swap_gates': number_of_SWAPs,
            }
            , ignore_index=True)
    df.to_csv(paths.cwd / f"benchmark/plot_scripts/CNOT_ratio_of_compiled_solutions.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Benchmark the settings for MC in the mc_parameters.yaml and save evaluation in mc_benchmark_results.txt"
    )
    parser.add_argument(
        "-path",
        "--problem_folder",
        type=str,
        default="training_set",
        help="path of problems to benchmark: lhz or training_set (default)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
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
        default=91,
        help="set the maximum C, for which benchmarks should be done",
    )
    parser.add_argument(
        "-minC",
        "--min_C",
        type=int,
        default=3,
        help="set the minimum C, for which benchmarks should be done",
    )
    parser.add_argument(
        "-maxN",
        "--max_N",
        type=int,
        default=4,
        help="benchmark_LHZ up to N",
    )
    parser.add_argument(
        "-minN",
        "--min_N",
        type=int,
        default=15,
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
    parser.add_argument(
        "--run",
        type=int,
        help="which run to benchmark?",
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        default=False,
        help="visualize",
    )
    parser.add_argument(
        "--visualize_core_search",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--number_of_core_images",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--number_of_images",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--CNOT_ratio",
        type=int,
        default=0,
        help="""
        if 1: creates a csv of ratios of the number of CNOT gates in a non compiled solution vs the LHZ solution
        if 2: creates a csv of ratios of the number of CNOT gates in a compiled solution vs the LHZ solution
        """
    )
    args = parser.parse_args()
    print("yaml_path", args.yaml_path)
    print("batch_size", args.batch_size)
    print("min_N", args.min_N)
    print("max_N", args.max_N)
    print("min_C", args.min_C)
    print("max_C", args.max_C)
    print("max_size", args.max_size)
    print("problem_folder", args.problem_folder)
    if args.CNOT_ratio == 2:
        CNOT_ratio_of_compiled_solutions(args)
    if args.CNOT_ratio == 1:
        CNOT_ratio_before_compilation(args)
    if args.CNOT_ratio == 0:
        # save results using id (which is also in filename of mc_parameters.yaml) in filename
        args.id_of_benchmark = args.yaml_path.split("_")[2]
        names = args.yaml_path.split(".")[0].split("_")
        args.name = f"{names[-2]}_{names[-1]}"
        if args.visualize:
            visualize_benchmarks(args)
        else:
            benchmark_energy_scaling(args)
