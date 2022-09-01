from CompilerQC.reversed_engineering.functions_for_database import (
    problem_from_file,
    energy_from_problem,
    get_files_to_problems,
)
import numpy as np
import yaml
import argparse
import pickle
from copy import deepcopy
from CompilerQC import *
import logging

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd


def update_mc(mc, mc_schedule, core: bool = False):
    """
    update attributes in MC or Energy
    according to the values in the yaml
    if core is True, update only the core attributes in the yaml,
    to update mc_core instead of mc
    """
    if core:  # if core is True, set core_ attributes in yaml to mc object
        for k, v in mc_schedule.items():
            if k.split("_")[0] != "core":
                continue
            k = k[5:]  # ignore 'core_' in k
            if k.split(".")[0] == "energy":
                if k.split(".")[1] == "scaling_model":
                    if v is not None:
                        mc.energy.set_scaling_from_model(v)
                else:
                    mc.energy.__setattr__(k.split(".")[1], v)
            else:
                mc.__setattr__(k, v)
        return mc
    else:  # if core is False, set all attributes in yaml to mc object excpet those starting with core_
        for k, v in mc_schedule.items():
            if k.split("_")[0] == "core":
                continue
            if k.split(".")[0] == "energy":
                if k.split(".")[1] == "scaling_model":
                    if v is not None:
                        mc.energy.set_scaling_from_model(v)
                else:
                    mc.energy.__setattr__(k.split(".")[1], v)
            else:
                mc.__setattr__(k, v)
        return mc


def init_energy_core(graph: Graph):
    graph_for_core = Graph.init_without_short_nodes(np.copy(graph.adj_matrix))
    qbits_for_core = Qbits.init_qbits_from_dict(graph_for_core, dict())
    nodes_object_for_core = Nodes(qbits_for_core)
    polygon_object_for_core = Polygons(
        nodes_object_for_core,
        polygons=Polygons.create_polygons(graph_for_core.get_cycles(4)),
    )
    energy_for_core = Energy_core(polygon_object_for_core)
    return energy_for_core


def init_energy(graph: Graph):
    qbits = Qbits.init_qbits_from_dict(graph, dict())
    nodes_object = Nodes(qbits, place_qbits_in_lines=False)
    polygon_object = Polygons(nodes_object)
    energy = Energy(polygon_object)
    return energy


# we could combine init_energy_core and intit_energy by removing Energy_core class
# and in evaluate_optimization() we would have to add graph.get_cycles(3) to polygon_object


def initialize_MC_object(graph: Graph, mc_schedule: dict, core: bool = False):
    """
    core is used in update_mc()
    """
    # initialise MC search
    if core:
        energy_core = init_energy_core(graph)
        mc = MC_core(energy_core)
    else:
        energy = init_energy(graph)
        mc = MC(energy)
    mc = update_mc(mc, mc_schedule, core)
    mc.n_moves = int(mc.n_moves * mc.repetition_rate)
    # initialize temperature
    initial_temperature = mc.current_temperature
    if initial_temperature == 0:
        initial_temperature = mc.initial_temperature()
    mc.T_0 = initial_temperature
    mc.current_temperature = initial_temperature
    if core:
        # remove_ancillas False since there shouldnt be any ancillas, so we dont have to remove them
        mc.reset(current_temperature=initial_temperature, remove_ancillas=False)
    else:
        # remove_ancillas False since there shouldnt be any ancillas, so we dont have to remove them
        mc.reset(
            current_temperature=initial_temperature,
            remove_ancillas=False,
            keep_core=False,
        )

    return mc


def evaluate_optimization(
    graph: Graph,
    name: str,
    mc_schedule: dict,
    dataframe: pd.DataFrame,
    batch_size: int,
    logger: "logging object",
):
    """
    evaluate monte carlo/ simulated annealing schedule for batch_size timed and returns
    a success probability -> measure of how good the choosen schedule (temperature,
    configuration distribution, ...) is
    """
    logger.info("Initialize mc object")
    mc = initialize_MC_object(graph, mc_schedule)
    mc.test1, mc.test2 = None, None
    if mc_schedule["with_core"]:
        mc_core = initialize_MC_object(graph, mc_schedule, core=True)
    else:
        mc.swap_probability = 0
    # benchmark
    success_rate = np.zeros(batch_size)
    record_n_total_steps = np.zeros(batch_size)
    record_n_missing_C = np.zeros(batch_size)
    record_core_size = np.zeros(batch_size)
    record_n_core_qbits = np.zeros(batch_size)
    number_of_ancillas = np.zeros(batch_size)
    for iteration in range(batch_size):
        # search for core
        if mc_schedule["with_core"]:
            logger.info(f"search core in iteration {iteration}")
            (
                qubit_coord_dict,
                mc.energy.polygon_object.core_corner,
                ancillas_in_core,
            ) = search_max_core(mc_core, graph.K)
            logger.info(
                f"found core with {mc_core.number_of_plaquettes} / {graph.C} plaquettes"
            )
            record_n_core_qbits[iteration] = len(qubit_coord_dict)
            record_core_size[iteration] = mc_core.number_of_plaquettes
            # if core is empty, dont allow swaps
            if len(qubit_coord_dict) == 0:
                mc.swap_probability = 0
                mc.finite_grid_size = True
                mc.random_qbit = True
            # reset mc core
            mc_core.reset(mc_core.T_0, remove_ancillas=True)
            # update core (reset is part of update)
            # remove ancillas which are already in graph, but not recognized due to removing of short lines in graph
            # add ancillas first since updates_qbits_from_dict doesnt know them otherwise
            ancillas_in_core = {qubit:coord for qubit, coord in ancillas_in_core.items() if qubit not in graph.qubits}
            mc.add_ancillas(ancillas_in_core, update_energy=False)
            mc.update_qbits_from_dict(qubit_coord_dict, assign_to_core=True)
            mc.total_energy, mc.number_of_plaquettes = mc.energy(
                mc.energy.polygon_object.nodes_object.qbits
            )      
        # check if there are still some qbits left to place
        remaining_qbits = len(mc.energy.polygon_object.nodes_object.qbits.qubits) - len(
            mc.energy.polygon_object.nodes_object.qbits.core_qbits
        )
        if remaining_qbits > 0:
            logger.info(f"place {remaining_qbits} remaining qubits")
            mc.apply(mc.n_moves)
        # remove ancillas which dont reduce d.o.f
        mc.remove_ancillas(
            mc.energy.polygon_object.nodes_object.propose_ancillas_to_remove()
        )
        # save results in arrays
        n_missing_C = graph.C - mc.number_of_plaquettes
        record_n_total_steps[iteration] = mc.n_total_steps
        record_n_missing_C[iteration] = n_missing_C
        success_rate[iteration] = n_missing_C == 0
        number_of_ancillas[iteration] = len(
            [
                qbit
                for qbit in mc.energy.polygon_object.nodes_object.qbits
                if qbit.ancilla == True
            ]
        )

        # reset mc object
        mc.reset(current_temperature=mc.T_0, remove_ancillas=True, keep_core=False)
    # save resutls in dataframe
    mc_schedule.update(
        {
            "batchsize": batch_size,
            "success_rate": np.mean(success_rate),
            "avg_n_missing_C": np.mean(record_n_missing_C),
            "var_n_missing_C": np.std(record_n_missing_C),
            "avg_n_total_steps": np.mean(record_n_total_steps),
            "var_n_total_steps": np.std(record_n_total_steps),
            "avg_core_size": np.mean(record_core_size),
            "std_core_size": np.std(record_core_size),
            "avg_n_core_qbits": np.mean(record_n_core_qbits),
            "std_n_core_qbits": np.std(record_n_core_qbits),
            "N": graph.N,
            "C": graph.C,
            "energy.scaling_for_plaq3": mc.energy.scaling_for_plaq3,
            "energy.scaling_for_plaq4": mc.energy.scaling_for_plaq4,
            "init_temperature": mc.T_0,
            "core_energy.scaling_for_plaq3": None,
            "core_energy.scaling_for_plaq4": None,
            "core_init_temperature": None,
            "core_N": None,
            "core_C": None,
            "name": name,
            "number_of_ancillas": np.mean(number_of_ancillas),
        }
    )
    if mc_schedule["with_core"]:
        mc_schedule.update(
            {
                "core_energy.scaling_for_plaq3": mc_core.energy.scaling_for_plaq3,
                "core_energy.scaling_for_plaq4": mc_core.energy.scaling_for_plaq4,
                "core_init_temperature": mc_core.T_0,
                "core_N": mc_core.energy.polygon_object.nodes_object.qbits.graph.N,
                "core_C": mc_core.energy.polygon_object.nodes_object.qbits.graph.C,
            }
        )
    dataframe = dataframe.append(mc_schedule, ignore_index=True, sort=False)
    return dataframe


def run_benchmark(
    benchmark_df: pd.DataFrame,
    graph: Graph,
    args,
    logger: "logging object",
):
    """
    evaluate simulated annealing schedules from mc_parameters yaml
    and save success rate in corresponding txt
    args contain args.batch_size, args.id_of_benchmarks
    """

    # load yaml
    path_to_config = paths.parameters_path / args.id_of_benchmark / args.yaml_path
    with open(path_to_config) as f:
        mc_schedule = yaml.load(f, Loader=yaml.FullLoader)
    # evaluate schedule from yaml and save it
    logger.info("================== benchmark ==================")
    logger.info(
        f"start benchmarking {args.id_of_benchmark, args.name} with problem size {graph.N}"
    )
    benchmark_df = evaluate_optimization(
        graph, args.name, mc_schedule, benchmark_df, args.batch_size, logger
    )
    return benchmark_df


def visualize_search(
    graph: Graph,
    name: str,
    mc_schedule: dict,
    visualize_core_search: bool,
    number_of_core_images: int=50,
    number_of_images: int=1000,
):
    """
    create gif of search according to settings in mc_schedule
    """
    mc = initialize_MC_object(graph, mc_schedule)
    if mc_schedule["with_core"]:
        mc_core = initialize_MC_object(graph, mc_schedule, core=True)
    else:
        mc.swap_probability = 0
    if mc_schedule["with_core"]:
        if visualize_core_search:
            search.visualize_search_process(mc_core, f"core_{name}", number_of_core_images)
        (
            qubit_coord_dict,
            mc.energy.polygon_object.core_corner,
            ancillas_in_core,
        ) = search_max_core(mc_core, graph.K)
        if len(qubit_coord_dict) == 0:
            mc.swap_probability = 0
            mc.finite_grid_size = True
            mc.random_qbit = True
        mc_core.reset(mc_core.T_0, remove_ancillas=True)
        ancillas_in_core = {qubit:coord for qubit, coord in ancillas_in_core.items() if qubit not in graph.qubits}
        mc.add_ancillas(ancillas_in_core, update_energy=False)
        mc.update_qbits_from_dict(qubit_coord_dict, assign_to_core=True)
    search.visualize_search_process(mc, name, number_of_images)
    mc.remove_ancillas(
        mc.energy.polygon_object.nodes_object.propose_ancillas_to_remove()
    )
    mc.reset(current_temperature=mc.T_0, remove_ancillas=True, keep_core=False)

        
def visualize_settings(
    graph: Graph,
    args,
):
    """
    """
    path_to_config = paths.parameters_path / args.id_of_benchmark / args.yaml_path
    with open(path_to_config) as f:
        mc_schedule = yaml.load(f, Loader=yaml.FullLoader)
    visualize_search(graph=graph,
                     name=f"gif_of_{args.name}",
                     mc_schedule=mc_schedule,
                     visualize_core_search=args.visualize_core_search,
                     number_of_core_images=args.number_of_core_images,
                     number_of_images=args.number_of_images)


def search_max_core(mc_core, K):
    mc_core.apply(mc_core.n_moves)
    (
        core_qbit_to_core_dict,
        core_corner,
        ancillas_in_core,
    ) = mc_core.energy.qbits_in_max_core(K)
    return core_qbit_to_core_dict, core_corner, ancillas_in_core


def update(origin_dict, new_dict):
    updated_dict = origin_dict.copy()
    updated_dict.update(new_dict)
    return updated_dict


def create_and_save_settings(name, new_dicts, new_config):
    """update default yaml by newdict and save them in mc_parameters_name.yaml
    note: there are absolute paths in use!"""
    (paths.parameters_path / "csvs").mkdir(parents=True, exist_ok=True)
    filenames, log_names = [], []
    for idx, new_dict in enumerate(new_dicts):
        # create folder if it doesnt exists yet
        filenames.append(f"mc_parameters_{name}_{idx}.yaml")
        log_names.append(paths.logger_path / name / f"{name}_{idx}")
        (paths.parameters_path / name).mkdir(parents=True, exist_ok=True)
        (paths.logger_path / name).mkdir(parents=True, exist_ok=True)
        dict_to_save = update(deepcopy(new_config), new_dict)
        with open(
            paths.parameters_path / name / f"mc_parameters_{name}_{idx}.yaml", "w"
        ) as f:
            print(name, idx)
            yaml.dump(dict_to_save, f, default_flow_style=False)
    pd.DataFrame(data={"filenames":filenames, "log_names":log_names}).to_csv(paths.parameters_path / "csvs" / f"{name}.csv", index=False, index_label=False)        
    with open(paths.parameters_path / "csvs" / f"run_{name}.sh", 'w') as sh:
        sh.write(('''\
#!/bin/bash 
NUM_PARALLEL_JOBS=10 
CSV_FILE=%s.csv
tail -n +2 ${CSV_FILE} | parallel --progress --colsep ',' -j${NUM_PARALLEL_JOBS} \
python "../../benchmark_optimization.py --yaml_path={1} --batch_size=20 \
--min_N=4 --max_N=15 --min_C=3 --max_C=91 --max_size=50 --problem_folder=training_set 2>&1 > {2}.log" 
        ''')%(name))
    with open(paths.plots / "scripts_to_create_gifs" / f"create_gif_for_{name}.sh", 'w') as sh:
        sh.write(('''\
#!/bin/bash 
NUM_PARALLEL_JOBS=10 
CSV_FILE=../../parameters/csvs/%s.csv
tail -n +2 ${CSV_FILE} | parallel --progress --colsep ',' -j${NUM_PARALLEL_JOBS} \
python "../../benchmark_optimization.py --yaml_path={1} \
--min_N=4 --max_N=15 --min_C=3 --max_C=91 --max_size=1 --problem_folder=training_set  --visualize=True --number_of_core_images=1000 --number_of_images=50" 
        ''')%(name))