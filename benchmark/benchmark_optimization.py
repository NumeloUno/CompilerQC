from CompilerQC import Graph, paths
from CompilerQC.reversed_engineering.functions_for_database import (
    problem_from_file, energy_from_problem, get_files_to_problems
)
import numpy as np
import yaml
import argparse
import pickle
from tqdm import tqdm
from copy import deepcopy 
from CompilerQC import Graph, Qbits, Polygons, Energy, MC, paths

def update_mc(mc, mc_schedule) -> MC:
    for k, v in mc_schedule.items():
        if k == 'n_moves':
            continue
        if k == 'energy_schedule':
            mc.energy.__setattr__(k, v)
            continue
        if k == 'energy_scaling':
            for name, scaling in v.items():
                mc.energy.__setattr__(name, scaling)
            continue
        mc.__setattr__(k, v)
    return mc

def evaluate_optimization(
    energy: Energy, mc_schedule: dict, batch_size: int,
):
    """
    evaluate monte carlo/ simulated annealing schedule for batch_size timed and returns
    a success probability -> measure of how good the choosen schedule (temperature, 
    configuration distribution, ...) is
    """
    success_rate = []
    for iteration in range(batch_size):
        mc = MC(deepcopy(energy))
        mc = update_mc(mc, mc_schedule)
        if mc.current_temperature == 0:
            initial_temperature = mc.initial_temperature()
            mc.current_temperature = initial_temperature
        for repetition in range(mc_schedule['n_moves']):
            mc.optimization_schedule()
        success_rate.append((mc.energy.polygon_object.qbits.graph.C
                             - mc.energy.polygon_object.number_of_plaqs
                            ) == 0)
    return sum(success_rate) / batch_size, [mc.chi_0, mc.delta, mc.repetition_rate, mc.n_total_steps, round(initial_temperature)]

def check_update(energy: Energy, mc_schedule: dict):
    """check if upate from yaml is working"""
    mc = MC(deepcopy(energy))
    mc = update_mc(mc, mc_schedule)
    if 'energy_scaling' in list(mc_schedule.keys()):
        assert mc.energy.scaling_for_plaq3 ==  mc_schedule['energy_scaling']['scaling_for_plaq3'], (
        'scaling for plaquette 3 didnt work')
        assert mc.energy.scaling_for_plaq4 ==  mc_schedule['energy_scaling']['scaling_for_plaq4'], (
        'scaling for plaquette 4 didnt work')
            
def run_benchmark(
    energy: Energy,
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
        
    # evaluate schedule from yaml and save it
    file = open(path_to_results, "a")
    for name, mc_schedule in config.items():
#         check_update(energy, mc_schedule)
        succes_rate,  [chi_0, delta, repetition_rate, n_total_steps, initial_temperature] = evaluate_optimization(energy, mc_schedule, batch_size)
        N, C = energy.polygon_object.qbits.graph.N, energy.polygon_object.qbits.graph.C
        file.write("{} {} {} {} {} {} {} {} {}\n".format(name,
                                                str(succes_rate),
                                                str(N),
                                                str(C),
                                                str(chi_0),
                                                str(delta),
                                                str(repetition_rate),
                                                str(n_total_steps),
                                                str(initial_temperature),
                                               ))
        print("{} {} {} {} {} {} {} {} {}\n".format(name,
                                                str(succes_rate),
                                                str(N),
                                                str(C),
                                                str(chi_0),
                                                str(delta),
                                                str(repetition_rate),
                                                str(n_total_steps),
                                                str(initial_temperature),
                                               ))
    file.close()

def benchmark_problem_folder(args):
    problems = get_files_to_problems(
                problem_folder=args.problem_folder,
                min_C=args.min_C,
                max_C=args.max_C,
                )
    for file in tqdm(problems, desc=f"Evaluate problems in folder {args.problem_folder}"):
        
        # read graph and qubit to coord translation from file
        graph_adj_matrix, qubit_coord_dict = (
            problem_from_file(file))
        # scopes for nonplaqs and plaqs
        polygon_scopes, NKC, n_cycles = energy_from_problem(graph_adj_matrix, qubit_coord_dict)
        n3, n4, p3, p4 = polygon_scopes

        graph = Graph(adj_matrix=graph_adj_matrix)
        # contribution of nonplaqs to each constraint 
        #scaling_for_plaquette = (sum(n3) + sum(n4)) / NKC[2]
        loaded_model = pickle.load(open(paths.MLP_energy_path / 'MLPregr_model.sav', 'rb'))
        predicted_energy = loaded_model.predict([[graph.N, graph.K, graph.C, graph.number_of_3_cycles, graph.number_of_4_cycles]]) 
        scaling_for_plaquette = predicted_energy / graph.C  
        
        # initialise energy_object
        qbits = Qbits.init_qbits_from_dict(graph,dict())
        polygon_object = Polygons(qbits=qbits)
        energy = Energy(
            polygon_object,
            scaling_for_plaq3=scaling_for_plaquette,
            scaling_for_plaq4=scaling_for_plaquette,
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


