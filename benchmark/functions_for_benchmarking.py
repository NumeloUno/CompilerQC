from CompilerQC.reversed_engineering.functions_for_database import (
    problem_from_file, energy_from_problem, get_files_to_problems
)
import numpy as np
import yaml
import argparse
import pickle
from copy import deepcopy 
from CompilerQC import Graph, Qbits, Polygons, Energy, MC, paths, core

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

def update_mc(mc, mc_schedule) -> MC:
    """
    update attributes in MC or Energy
    according to the values in the yaml
    """
    for k, v in mc_schedule.items():
        if k.split('.')[0] == 'energy':
            mc.energy.__setattr__(k.split('.')[1], v)
        else:
            mc.__setattr__(k, v)
    return mc


def evaluate_optimization(
    energy: Energy, name: str, mc_schedule: dict, dataframe: pd.DataFrame,  batch_size: int,
):
    """
    evaluate monte carlo/ simulated annealing schedule for batch_size timed and returns
    a success probability -> measure of how good the choosen schedule (temperature, 
    configuration distribution, ...) is
    """ 
    # initialise MC search
    mc = MC(deepcopy(energy))
    mc = update_mc(mc, mc_schedule)  
    mc.n_moves = int(mc.n_moves * mc.repetition_rate)
    # initialize temperature
    initial_temperature = mc.current_temperature
    if initial_temperature == 0:
        initial_temperature = mc.initial_temperature()
    mc.T_0 = initial_temperature
    mc.current_temperature = initial_temperature
    # delete core (if True) and reset MC search
    if not mc.with_core:
        mc.reset(current_temperature=initial_temperature, with_core=mc.with_core)
    else:
        assert len(mc.energy.polygon_object.qbits.core_qbits) > 0, "a core is expected, but the core is empty"
    # benchmark    
    success_rate = []
    avg_n_total_steps = 0          
    for iteration in range(batch_size):
        # apply search
        for repetition in range(mc.n_moves):
            mc.optimization_schedule()
        avg_n_total_steps += mc.n_total_steps
        C = mc.energy.polygon_object.qbits.graph.C
        success_rate.append((C - mc.number_of_plaquettes) == 0)
        #reset mc object
        mc.reset(current_temperature=initial_temperature, with_core=mc.with_core)
    # append to dataframe     
    mc_schedule.update(
        {'success_rate':sum(success_rate) / batch_size,
         'N':mc.energy.polygon_object.qbits.graph.N,
         'C':mc.energy.polygon_object.qbits.graph.C,
         'avg_n_total_steps':avg_n_total_steps / batch_size,
         'energy.scaling_for_plaq3':mc.energy.scaling_for_plaq3,
         'energy.scaling_for_plaq4':mc.energy.scaling_for_plaq4,
         'initial_temperature': initial_temperature,
         'name': name,
        })
    dataframe = dataframe.append(mc_schedule, ignore_index=True)
    return dataframe
            
def run_benchmark(
    benchmark_df: pd.DataFrame,
    energy: Energy,
    args,
):
    """
    evaluate simulated annealing schedules from mc_parameters yaml
    and save success rate in corresponding txt
    args contain args.batch_size, args.id_of_benchmarks
    """
    # load yaml
    path_to_config = paths.parameters_path / f"mc_parameters_{args.id_of_benchmark}.yaml"
    with open(path_to_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # evaluate schedule from yaml and save it
    for name, mc_schedule in config.items():
        benchmark_df = evaluate_optimization(energy, name, mc_schedule, benchmark_df, args.batch_size)
        print(energy.polygon_object.qbits.graph.N, name, args.id_of_benchmark, args.extra_id)
    return benchmark_df


# def evaluate_optimization(
#     energy: Energy, mc_schedule: dict, dataframe: pd.DataFrame,  batch_size: int,
# ):
#     """
#     evaluate monte carlo/ simulated annealing schedule for batch_size timed and returns
#     a success probability -> measure of how good the choosen schedule (temperature, 
#     configuration distribution, ...) is
#     """ # TODO: think how to make this faster, e.g. remove_core...
#     success_rate = []
#     for iteration in range(batch_size):
#         # initialise MC search
#         mc = MC(deepcopy(energy))
#         mc = update_mc(mc, mc_schedule)
#         if not mc.with_core:
#             Qbits.remove_core(mc.energy.polygon_object.qbits)
#         initial_temperature = mc.current_temperature
#         if initial_temperature == 0:
#             # initial tempertature is set since mc_schedule is updatet by it!! fix benchmark"""
#             initial_temperature = mc.initial_temperature()
#         mc.T_0 = initial_temperature
#         mc.current_temperature = initial_temperature
#         # apply search
#         for repetition in range(mc.n_moves):
#             mc.optimization_schedule()
#         success_rate.append((mc.energy.polygon_object.qbits.graph.C
#                              - mc.energy.polygon_object.number_of_plaqs #replace with mc.number_of_pl...
#                             ) == 0)
#         mc_schedule.update(
#             {'success_rate':sum(success_rate) / batch_size,
#              'N':mc.energy.polygon_object.qbits.graph.N,
#              'C':mc.energy.polygon_object.qbits.graph.C,
#              'n_total_steps':mc.n_total_steps,
#              'energy.scaling_for_plaq3':mc.energy.scaling_for_plaq3,
#              'energy.scaling_for_plaq4':mc.energy.scaling_for_plaq4,
#              'initial_temperature': initial_temperature,
#             })
#         dataframe = dataframe.append(mc_schedule, ignore_index=True)
