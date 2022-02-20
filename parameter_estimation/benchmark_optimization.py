import numpy as np
import random
import matplotlib.pyplot as plt
import yaml
import argparse
from tqdm import tqdm
from copy import deepcopy #actually deepcopy isnt needed here
from CompilerQC import Graph, Polygons, core, Energy, MC, paths
import os 
from CompilerQC.reversed_engineering.functions_for_database import (
    problem_from_file, energy_from_problem, get_files_to_problems
)

def update_mc(mc, mc_schedule) -> MC:
    for k, v in mc_schedule.items():
        mc.__setattr__(k, v)
    return mc


def evaluate_optimization(
    energy_object: Energy, mc_schedule: dict, batch_size: int, visualize: bool = False
):
    """
    evaluate monte carlo/ simulated annealing schedule for batch_size timed and returns
    a success probability -> measure of how good the choosen schedule (temperature, 
    configuration distribution, ...) is
    """
    success_rate = []
    for iteration in range(batch_size):
        mc = MC(deepcopy(energy_object))
        mc = update_mc(mc, mc_schedule)
        for repetition in range(mc.n_moves):
            mc.optimization_schedule()
            if mc.polygon.n_found_plaqs() == mc.polygon.C:
                break
        success_rate.append((mc.polygon.C - mc.polygon.n_found_plaqs()) == 0)
        if visualize:
            fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
            for coord in mc.free_neighbour_coords(mc.polygon.core_coords):
                ax.annotate(".", coord)
                ax.scatter(*coord, color="green")
            mc.polygon.visualize(ax, mc.polygon.get_all_polygon_coords(), zoom=1)
            plt.show()
    return sum(success_rate) / batch_size


def run_benchmark(
    energy_object: Energy,
    batch_size: int = 100,
    id_of_benchmark: str = "",
):
    """
    evaluate simulated annealing schedules from mc_parameters yaml
    and save success rate in corresponding txt
    """

    path_to_config = paths.parameters_path / f"mc_parameters_{id_of_benchmark}.yaml"
    path_to_results = (
        paths.benchmark_results_path / f"mc_benchmark_results_{id_of_benchmark}.txt"
    )
    with open(path_to_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    file = open(path_to_results, "a")

    for name, mc_schedule in config.items():
        succes_rate = evaluate_optimization(energy_object, mc_schedule, batch_size)
        file.write(
            name + " " + str(succes_rate) + " " + str(energy_object.polygon.N) + "\n"
        )
        print(name, succes_rate)
    file.close()

def benchmark_problem_folder(args):
    for file in tqdm(get_files_to_problems(args.problem_folder), desc=f"Evaluate problems in folder {args.problem_folder}"):
        if int(str(file).split('_')[-3]) > args.max_C:
            continue
        graph_adj_matrix, qbit_coord_dict = (
            problem_from_file(file))
        polygon_scopes, NKC = energy_from_problem(graph_adj_matrix, qbit_coord_dict)
        n3, n4, p3, p4 = polygon_scopes
        scaling_for_plaquette = (sum(n3) + sum(n4)) / NKC[2]

        graph = Graph(adj_matrix=graph_adj_matrix)
        U, V = [], []
        if args.with_core:
            nn = core.largest_complete_bipartite_graph(graph)
            K_nn = core.complete_bipartite_graph(*nn)
            U, V = core.parts_of_complete_bipartite_graph(graph.to_nx_graph(), K_nn)
        polygon_object = Polygons.from_max_core_bipartite_sets(graph, [U, V])
        energy_object = Energy(polygon_object)
        energy_object.scaling_for_plaq3 = scaling_for_plaquette
        energy_object.scaling_for_plaq4 = scaling_for_plaquette
        run_benchmark(energy_object, args.batchsize, args.id_of_benchmark)

        break
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Benchmark the settings for MC in the mc_parameters.yaml and save evaluation in mc_benchmark_results.txt"
    )
    parser.add_argument(
        "-path",
        "--problem_folder",
        type=str,
        default='training_set',
        help="path of problems to benchmark"
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=100,
        help="how many times each schedule is evaluated",
    )
    parser.add_argument(
        "-id",
        "--id_of_benchmark",
        type=int,
        default="",
        help="save results using id (which is also in filename of mc_parameters.yaml) in filename",
    )
    parser.add_argument(
        "-C",
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


