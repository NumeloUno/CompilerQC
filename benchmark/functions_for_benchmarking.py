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
    qbits = Qbits.init_qbits_from_dict(graph, dict())
    nodes_object = Nodes(qbits)
    polygons_4 = Polygons.create_polygons(graph.get_cycles(4))
    polygon_object = Polygons(nodes_object.qbits, polygons=polygons_4)
    polygon_object.nodes_object = nodes_object
    energy_core = Energy_core(polygon_object)
    return energy_core

def init_energy(graph: Graph):
    qbits = Qbits.init_qbits_from_dict(graph, dict())
    polygon_object = Polygons(qbits)
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
        mc.reset(current_temperature=initial_temperature)
    else:
        mc.reset(current_temperature=initial_temperature, with_core=False)
        
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
        mc.operation_schedule.pop('swap', None)
    # benchmark    
    success_rate = np.zeros(batch_size)
    record_n_total_steps = np.zeros(batch_size)
    record_n_missing_C = np.zeros(batch_size)
    record_core_size = np.zeros(batch_size)
    record_n_core_qbits = np.zeros(batch_size)
    for iteration in range(batch_size):
        if mc_schedule['with_core']:
            logger.info(f"search core in iteration {iteration}")
            qubit_coord_dict, mc.energy.polygon_object.core_corner = search_max_core(mc_core)
            logger.info(f"found core with {mc_core.number_of_plaquettes} / {graph.C} plaquettes")
            record_n_core_qbits[iteration] = len(qubit_coord_dict)
            record_core_size[iteration] = mc_core.number_of_plaquettes
            # if core is empty, dont allow swaps
            if len(qubit_coord_dict) == 0:
                mc.operation_schedule.pop('swap', None)
            # reset mc core
            mc_core.reset(mc_core.T_0)
            # update core (reset is part of update)
            mc.energy.polygon_object.qbits.update_qbits_from_dict(qubit_coord_dict)
            mc.reset(mc.T_0, with_core=True)      
        
        # check if there are still some qbits left to place
        remaining_qbits = (
            len(mc.energy.polygon_object.qbits.qubits)
            - len(mc.energy.polygon_object.qbits.core_qbits)
        )
        if remaining_qbits > 0:
            logger.info(f"place {remaining_qbits} remaining qubits")
            # apply search
            for repetition in range(mc.n_moves):
                mc.optimization_schedule()
        # save results in arrays
        n_missing_C = (graph.C - mc.number_of_plaquettes)
        record_n_total_steps[iteration] = mc.n_total_steps
        record_n_missing_C[iteration] = n_missing_C
        success_rate[iteration] = (n_missing_C == 0)
        #reset mc object
        mc.reset(current_temperature=mc.T_0, with_core=False)
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
         'initial_temperature': mc.T_0,
         'name': name,
        })
    dataframe = dataframe.append(mc_schedule, ignore_index=True)
    return dataframe, mc
            
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

def search_max_core(mc_core): 
    for repetition in range(mc_core.n_moves):
        mc_core.optimization_schedule()
    core_qubit_coord_dict, core_corner = mc_core.energy.qbits_in_max_core()
    return core_qubit_coord_dict, core_corner

# def evaluate_optimization_with_core(
#     graph: Graph, name: str,mc_schedule: dict, dataframe: pd.DataFrame,  batch_size: int,
# ):
#     """
#     evaluate monte carlo/ simulated annealing schedule for batch_size timed and returns
#     a success probability -> measure of how good the choosen schedule (temperature, 
#     configuration distribution, ...) is
#     """ 
    
#     energy_core = init_energy_core(graph)

#     # benchmark    
#     success_rate = []
#     record_n_total_steps = []
#     record_n_missing_C = []
#     for iteration in range(batch_size):
#         core_qubit_coord_dict, core_initial_temperature = search_max_core(deepcopy(energy_core), mc_schedule)
#         energy = init_energy(graph, core_qubit_coord_dict)
#         mc, initial_temperature = initialize_MC_object(energy, mc_schedule)
#         for repetition in range(mc.n_moves):
#             mc.optimization_schedule()
#         C = mc.energy.polygon_object.qbits.graph.C
#         n_missing_C = (C - mc.number_of_plaquettes)
#         if n_missing_C == 0:
#             avg_n_total_steps += mc.n_total_steps
#         avg_n_missing_C += n_missing_C
#         success_rate.append(n_missing_C == 0)
#         #reset mc object
#         mc.reset(current_temperature=initial_temperature, with_core=mc.with_core)
#         # append to dataframe   


#     #if len(record_n_total_steps) == 0:
#     #record_n_total_steps.append(mc.n_total_steps)
    
#     record_n_total_steps = np.array(record_n_total_steps)

#     #if len(record_n_missing_C) == 0:
#     #record_n_missing_C.append(0)    
        
#     record_n_missing_C = np.array(record_n_missing_C)
    
#     if not mc_schedule['with_core']:
#         record_core_size.append(0)
#     record_core_size = np.array(record_core_size)
        
    
    
    
#         mc_schedule.update(
#             {'success_rate':sum(success_rate) / batch_size,
#              'avg_n_missing_C': avg_n_missing_C / batch_size, 
#              'core_size': len(core_qubit_coord_dict),
#              'N':mc.energy.polygon_object.qbits.graph.N,
#              'C':mc.energy.polygon_object.qbits.graph.C,
#              'avg_n_total_steps':avg_n_total_steps / batch_size,
#              'energy.scaling_for_plaq3':mc.energy.scaling_for_plaq3,
#              'energy.scaling_for_plaq4':mc.energy.scaling_for_plaq4,
#              'initial_temperature': initial_temperature,
#              'initial_temperature': core_initial_temperature,
#              'name': name,
#             })
#         dataframe = dataframe.append(mc_schedule, ignore_index=True)
#         return dataframe

#             # append mc_core_schedule also to df, run_benchmark_with_core...