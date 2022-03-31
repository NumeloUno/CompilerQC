from CompilerQC.reversed_engineering.functions_for_database import (
    problem_from_file, energy_from_problem, get_files_to_problems
)
import numpy as np
import yaml
import argparse
import pickle
from copy import deepcopy 
from CompilerQC import Graph, Qbits, Polygons, Energy, MC, paths, core


head = "{:9} {:12} {:3} {:3} {:6} {:6} {:16} {:15} {:18} {:15} {:15} {:6} \n".format(
    'name', 'succes_rate', 'N', 'C', 'chi_0', 'delta',
    'repetition_rate', 'n_total_steps', 'initial_temperature',
    'scaling_plaq3', 'scaling_plaq4', 'with_core')

def update_mc(mc, mc_schedule) -> MC:
    """
    update attributes in MC or Energy
    according to the values in the yaml
    """
    for k, v in mc_schedule.items():
        if k == 'n_moves':
            continue
        if k.split('.')[0] == 'energy':
            mc.energy.__setattr__(k.split('.')[1], v)
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
            mc.T_0 = initial_temperature
            mc.current_temperature = initial_temperature
            mc.n_moves = mc_schedule['n_moves']
        for repetition in range(mc_schedule['n_moves']):
            mc.optimization_schedule()
        success_rate.append((mc.energy.polygon_object.qbits.graph.C
                             - mc.energy.polygon_object.number_of_plaqs
                            ) == 0)
    return (sum(success_rate) / batch_size,
            mc.energy.polygon_object.qbits.graph.N,
            mc.energy.polygon_object.qbits.graph.C,
            mc.chi_0, mc.delta, mc.repetition_rate,
            mc.n_total_steps, initial_temperature,
            mc.energy.scaling_for_plaq3,
            mc.energy.scaling_for_plaq4)
            
def run_benchmark(
    file,
    energy: Energy,
    args,
):
    """
    evaluate simulated annealing schedules from mc_parameters yaml
    and save success rate in corresponding txt
    args contain args.batch_size, args.id_of_benchmarks
    """
    path_to_config = paths.parameters_path / f"mc_parameters_{args.id_of_benchmark}.yaml"
    with open(path_to_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # evaluate schedule from yaml and save it
    for name, mc_schedule in config.items():
        write_to_file = evaluate_optimization(energy, mc_schedule, args.batch_size)
        result = "{:15} {:4} {:4} {:4} {:4} {:6} {:4} {:9} {:8.0f} {:8.0f} {:8.0f} {:6} \n".format(
            name, *write_to_file, args.with_core)
        file.write(result)
        print(result)
    return file 
    
# def check_update(energy: Energy, mc_schedule: dict):
#     """check if upate from yaml is working"""
#     mc = MC(deepcopy(energy))
#     mc = update_mc(mc, mc_schedule)
#     if 'energy_scaling' in list(mc_schedule.keys()):
#         assert mc.energy.scaling_for_plaq3 ==  mc_schedule['energy_scaling']['scaling_for_plaq3'], (
#         'scaling for plaquette 3 didnt work')
#         assert mc.energy.scaling_for_plaq4 ==  mc_schedule['energy_scaling']['scaling_for_plaq4'], (
#         'scaling for plaquette 4 didnt work')

