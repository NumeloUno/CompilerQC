from CompilerQC.reversed_engineering.functions_for_database import (
    problem_from_file, energy_from_problem, get_files_to_problems
)
import numpy as np
import yaml
import argparse
import pickle
from copy import deepcopy 
from CompilerQC import *
import logging

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

def update_mc(mc, mc_schedule, core: bool=False) -> MC:
    """
    update attributes in MC or Energy
    according to the values in the yaml
    if core is True, update only the core attributes in the yaml,
    to update mc_core instead of mc
    """
    if core: #if core is True, set core_ attributes in yaml to mc object
        for k, v in mc_schedule.items():
            if k.split('_')[0] != 'core':
                continue
            k = k[5:] #ignore 'core_' in k
            if k.split('.')[0] == 'energy':
                if k.split('.')[1] == 'scaling_model':
                    mc.energy.set_scaling_from_model(v)
                mc.energy.__setattr__(k.split('.')[1], v)
            else:
                mc.__setattr__(k, v)
        return mc
    else: # if core is False, set all attributes in yaml to mc object excpet those starting with core_
        for k, v in mc_schedule.items():
            if k.split('_')[0] == 'core':
                continue
            if k.split('.')[0] == 'energy':
                if k.split('.')[1] == 'scaling_model':
                    mc.energy.set_scaling_from_model(v)
                mc.energy.__setattr__(k.split('.')[1], v)
            else:
                mc.__setattr__(k, v)
        return mc

def init_energy_core(graph: Graph):
    graph_for_core = Graph.init_without_short_nodes(np.copy(graph.adj_matrix))
    qbits_for_core = Qbits.init_qbits_from_dict(graph_for_core, dict())
    nodes_object_for_core = Nodes(qbits_for_core)
    polygon_object_for_core = Polygons(nodes_object_for_core,
                                       polygons=Polygons.create_polygons(graph_for_core.get_cycles(4))
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

def initialize_MC_object(graph: Graph, mc_schedule: dict, core: bool=False):
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
        mc.reset(current_temperature=initial_temperature, remove_ancillas=False, keep_core=False)
        
    return mc

def evaluate_optimization(
    graph: Graph, name: str, mc_schedule: dict, dataframe: pd.DataFrame,  batch_size: int, logger: 'logging object',
):
    """
    evaluate monte carlo/ simulated annealing schedule for batch_size timed and returns
    a success probability -> measure of how good the choosen schedule (temperature, 
    configuration distribution, ...) is
    """ 
    logger.info('Initialize mc object')
    mc = initialize_MC_object(graph, mc_schedule)
    if mc_schedule['with_core']:
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
        if mc_schedule['with_core']:
            logger.info(f"search core in iteration {iteration}")
            qubit_coord_dict, mc.energy.polygon_object.core_corner, ancillas_in_core = search_max_core(mc_core, graph.K)
            logger.info(f"found core with {mc_core.number_of_plaquettes} / {graph.C} plaquettes")
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
            mc.update_qbits_from_dict(qubit_coord_dict, assign_to_core=True) 
            print(ancillas_in_core)
            mc.add_ancillas(ancillas_in_core)


        # check if there are still some qbits left to place
        remaining_qbits = (
            len(mc.energy.polygon_object.nodes_object.qbits.qubits)
            - len(mc.energy.polygon_object.nodes_object.qbits.core_qbits)
        )
        if remaining_qbits > 0:
            logger.info(f"place {remaining_qbits} remaining qubits")
            mc.apply(mc.n_moves)
        # remove ancillas which dont reduce d.o.f
        mc.remove_ancillas(mc.energy.polygon_object.nodes_object.propose_ancillas_to_remove())
        # save results in arrays
        n_missing_C = (graph.C - mc.number_of_plaquettes)
        record_n_total_steps[iteration] = mc.n_total_steps
        record_n_missing_C[iteration] = n_missing_C
        success_rate[iteration] = (n_missing_C == 0)
        number_of_ancillas[iteration] = len([qbit for qbit in mc.energy.polygon_object.nodes_object.qbits if qbit.ancilla==True])

        #reset mc object
        mc.reset(current_temperature=mc.T_0, remove_ancillas=True, keep_core=False)
    # save resutls in dataframe 
    print(record_n_missing_C)
    mc_schedule.update(
        {'success_rate': np.mean(success_rate),
         'avg_n_missing_C': np.mean(record_n_missing_C),
         'var_n_missing_C': np.std(record_n_missing_C),
         'avg_n_total_steps': np.mean(record_n_total_steps),
         'var_n_total_steps': np.std(record_n_total_steps),
         'avg_core_size': np.mean(record_core_size),
         'std_core_size': np.std(record_core_size), 
         'avg_n_core_qbits': np.mean(record_n_core_qbits),
         'std_n_core_qbits': np.std(record_n_core_qbits), 
         'N': graph.N,
         'C': graph.C,
         'energy.scaling_for_plaq3': mc.energy.scaling_for_plaq3,
         'energy.scaling_for_plaq4': mc.energy.scaling_for_plaq4,
         'init_temperature': mc.T_0,
         'name': name,
         'number_of_ancillas':np.mean(number_of_ancillas),
        })
    dataframe = dataframe.append(mc_schedule, ignore_index=True)
    return dataframe

def run_benchmark(
    benchmark_df: pd.DataFrame,
    graph: Graph,
    args,
    logger: 'logging object',
):
    """
    evaluate simulated annealing schedules from mc_parameters yaml
    and save success rate in corresponding txt
    args contain args.batch_size, args.id_of_benchmarks
    """

    # load yaml
    path_to_config = paths.parameters_path / args.id_of_benchmark / args.yaml_path
    with open(path_to_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # evaluate schedule from yaml and save it
    for name, mc_schedule in config.items():
        logger.info("================== benchmark ==================")
        logger.info(f"start benchmarking {args.id_of_benchmark, name} with problem size {graph.N}")
        benchmark_df = evaluate_optimization(graph, name, mc_schedule, benchmark_df, args.batch_size, logger)
    return benchmark_df

def visualize_search(
    graph: Graph, name: str, mc_schedule: dict,
):
    """
    create gif of search according to settings in mc_schedule
    """
    mc = initialize_MC_object(graph, mc_schedule)
    if mc_schedule['with_core']:
        mc_core = initialize_MC_object(graph, mc_schedule, core=True)
    else:
        mc.swap_probability = 0
    if mc_schedule['with_core']:
        qubit_coord_dict, mc.energy.polygon_object.core_corner, ancillas_in_core = search_max_core(mc_core, graph.K)
        if len(qubit_coord_dict) == 0:
            mc.swap_probability = 0
            mc.finite_grid_size = True
            mc.random_qbit = True
        mc_core.reset(mc_core.T_0, remove_ancillas=True)
        mc.update_qbits_from_dict(qubit_coord_dict, assign_to_core=True) 
        mc.add_ancillas(ancillas_in_core)
    remaining_qbits = (
        len(mc.energy.polygon_object.nodes_object.qbits.qubits)
        - len(mc.energy.polygon_object.nodes_object.qbits.core_qbits)
    )
    search.visualize_search_process(mc, name, 1000)
    mc.reset(current_temperature=mc.T_0, remove_ancillas=True, keep_core=False)

def visualize_settings(
    graph: Graph,
    args,
):
    path_to_config = paths.parameters_path / args.id_of_benchmark / args.yaml_path
    with open(path_to_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for name, mc_schedule in config.items():
        visualize_search(graph, f'{args.yaml_path}_{name}', mc_schedule)

def search_max_core(mc_core, K): 
    mc_core.apply(mc_core.n_moves)
    core_qbit_to_core_dict, core_corner, ancillas_in_core = mc_core.energy.qbits_in_max_core(K)
    return core_qbit_to_core_dict, core_corner, ancillas_in_core