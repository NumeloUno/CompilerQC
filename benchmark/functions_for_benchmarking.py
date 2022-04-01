from CompilerQC.reversed_engineering.functions_for_database import (
    problem_from_file, energy_from_problem, get_files_to_problems
)
import numpy as np
import yaml
import argparse
import pickle
from copy import deepcopy 
from CompilerQC import Graph, Qbits, Polygons, Energy, MC, paths, core
import pandas as pd

def update_mc(mc, mc_schedule) -> MC:
    """
    update attributes in MC or Energy
    according to the values in the yaml
    """
    for k, v in mc_schedule.items():
        if k.split('.')[0] == 'energy':
            mc.energy.__setattr__(k.split('.')[1], v)
            continue
        mc.__setattr__(k, v)
    return mc


# def evaluate_optimization(
#     energy: Energy, mc_schedule: dict, dataframe: pd.DataFrame,  batch_size: int,
# ):
#     """
#     evaluate monte carlo/ simulated annealing schedule for batch_size timed and returns
#     a success probability -> measure of how good the choosen schedule (temperature, 
#     configuration distribution, ...) is
#     """ # TODO: think how to make this faster, e.g. remove_core...
#     success_rate = []
    
#     # initialise MC search
#     mc = MC(deepcopy(energy))
#     mc = update_mc(mc, mc_schedule)
#     if not mc.with_core:
#         Qbits.remove_core(mc.energy.polygon_object.qbits)
#     initial_temperature = mc.current_temperature
#     if initial_temperature == 0:
#         initial_temperature = mc.initial_temperature()
#     mc.T_0 = initial_temperature
#     mc.current_temperature = initial_temperature
        
#     for iteration in range(batch_size):
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
#         #reset mc object
#         mc.reset(current_temperature=initial_temperature, with_core=mc.with_core)
#         return dataframe
def evaluate_optimization(
    energy: Energy, mc_schedule: dict, dataframe: pd.DataFrame,  batch_size: int,
):
    """
    evaluate monte carlo/ simulated annealing schedule for batch_size timed and returns
    a success probability -> measure of how good the choosen schedule (temperature, 
    configuration distribution, ...) is
    """ # TODO: think how to make this faster, e.g. remove_core...
    success_rate = []
    for iteration in range(batch_size):
        # initialise MC search
        mc = MC(deepcopy(energy))
        mc = update_mc(mc, mc_schedule)
        if not mc.with_core:
            Qbits.remove_core(mc.energy.polygon_object.qbits)
        initial_temperature = mc.current_temperature
        if initial_temperature == 0:
            # initial tempertature is set since mc_schedule is updatet by it!! fix benchmark"""
            initial_temperature = mc.initial_temperature()
        mc.T_0 = initial_temperature
        mc.current_temperature = initial_temperature
        # apply search
        for repetition in range(mc.n_moves):
            mc.optimization_schedule()
        success_rate.append((mc.energy.polygon_object.qbits.graph.C
                             - mc.energy.polygon_object.number_of_plaqs #replace with mc.number_of_pl...
                            ) == 0)
        mc_schedule.update(
            {'success_rate':sum(success_rate) / batch_size,
             'N':mc.energy.polygon_object.qbits.graph.N,
             'C':mc.energy.polygon_object.qbits.graph.C,
             'n_total_steps':mc.n_total_steps,
             'energy.scaling_for_plaq3':mc.energy.scaling_for_plaq3,
             'energy.scaling_for_plaq4':mc.energy.scaling_for_plaq4,
             'initial_temperature': initial_temperature,
            })
        dataframe = dataframe.append(mc_schedule, ignore_index=True)

            
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
        benchmark_df = evaluate_optimization(energy, mc_schedule, benchmark_df, args.batch_size)
    return benchmark_df
