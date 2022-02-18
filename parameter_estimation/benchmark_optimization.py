import numpy as np
import random
import matplotlib.pyplot as plt
import yaml
import argparse
from copy import deepcopy #actually deepcopy isnt needed here
from CompilerQC import Graph, Polygons, core, Energy, MC, paths

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Benchmark the settings for MC in the mc_parameters.yaml and save evaluation in mc_benchmark_results.txt"
    )
    parser.add_argument(
        "-N", type=int, default=4, help="size of complete connected logical graph"
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
    args = parser.parse_args()
    graph = Graph.complete(args.N)
    nn = core.largest_complete_bipartite_graph(graph)
    K_nn = core.complete_bipartite_graph(*nn)
    U, V = core.parts_of_complete_bipartite_graph(graph.to_nx_graph(), K_nn)
    polygon_object = Polygons(graph, core_bipartite_sets=[U, V])
    energy_object = Energy(polygon_object)
    run_benchmark(energy_object, args.batchsize, args.id_of_benchmark)
