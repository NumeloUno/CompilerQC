# from functions_for_benchmarking import *
from tqdm import tqdm
from CompilerQC import *
import numpy as np
import pickle
import argparse
import os.path
import pandas as pd
import logging
import csv

path_to_results = lambda args: (
    paths.benchmark_results_path
    / f"benchmark_results_{args.id_of_benchmark}_{args.problem_folder}.csv"
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
        graph_adj_matrix, qubit_coord_dict = functions_for_database.problem_from_file(
            file
        )
        graphs.append(graph_adj_matrix)
        if len(graphs) == args.max_size:
            break
    return graphs


def benchmark_energy_scaling(args):
    """
    benchmark the search of compiled graphs for
    fully connected logical graphs,
    scale plaqs by prediction of MLP
    logger note: duplicate-log-output-when-using-python-logging-module
    """
    benchmark_df = pd.DataFrame()

    logger = logging.getLogger()
    fhandler = logging.FileHandler(
        filename=paths.logger_path / f"{args.id_of_benchmark}.log", mode="a"
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fhandler.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    logger.info("=================================================================")
    logger.info(f"benchmark {args.max_size} problems from {args.problem_folder} folder")
    logger.info("=================================================================")

    for adj_matrix in graphs_to_benchmark(args):
        graph = Graph(adj_matrix=adj_matrix)
        benchmark_df = functions_for_benchmarking.run_benchmark(
            benchmark_df, graph, args, logger
        )

    csv_path = path_to_results(args)
    if (csv_path).exists():
        with open(csv_path, "r") as file:
            fieldnames = csv.DictReader(file).fieldnames
        benchmark_df[fieldnames].to_csv(csv_path, mode="a", header=False, index=False)
    else:
        benchmark_df.to_csv(csv_path, mode="a", header=True, index=False)


def visualize_benchmarks(args):
    """
    visualize the search of compiled graphs for
    fully connected logical graphs,
    scale plaqs by prediction of MLP
    """
    for adj_matrix in graphs_to_benchmark(args):
        graph = Graph(adj_matrix=adj_matrix)
        functions_for_benchmarking.visualize_settings(graph, args)


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
    parser.add_argument(
        "--visualize",
        type=bool,
        default=False,
        help="visualize",
    )
    args = parser.parse_args()
    # save results using id (which is also in filename of mc_parameters.yaml) in filename
    args.id_of_benchmark = args.yaml_path.split("_")[2]
    if args.visualize:
        visualize_benchmarks(args)
    else:
        benchmark_energy_scaling(args)
