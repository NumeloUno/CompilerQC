#!/usr/bin/env python
# coding: utf-8

# Here you can create the yaml files, which can be benchmarked afterwards.
# Therefore, the default.yaml in the parameters folder is updatet by the dictionary you will specify.
# Dont use _ in file names
#
# Move to folder and run e.g.:find -name "*yaml" | parallel python ../../benchmark_optimization.py --yaml_path={} -b=10 -minN=10 -maxN=10   or run sh ../run_....sh
# or move to csvs folder in parameters and run sh files, the csvs are created in this script, by calling the function create_... in functions_for_benchmarking, there you can also change settings for all csvs, thats not ideal i know

# delete all folder
from pathlib import Path
import shutil
import os
import itertools
print("===================================")
print("==== Delete old sh_scripts =====")
print("===================================")
for p in Path("sh_scripts/").glob("*.sh"):
    shutil.rmtree(f"parameters/{p.name}", ignore_errors=True)
print("===================================")
print("==== Delete old settings/csvs =====")
print("===================================")
for p in Path("parameters/").glob("*"):
    if p.name not in ["Default"]:
        print(p.name)
        shutil.rmtree(f"parameters/{p.name}", ignore_errors=True)
print("===================================")
print("=Delete old scripts to create gifs=")
print("===================================")
for p in Path("plots/scripts_to_create_gifs").glob("*"):
    print(p.name)
    os.remove(p)
    
import yaml
from CompilerQC import *
print("=============================================")
print("==== Create new settings (yamls and csvs) ===")
print("=============================================")
# open default yaml and update it by general settings
with open(paths.parameters_path / "Default/default.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
default_update = {}
new_config = functions_for_benchmarking.update(config, default_update)

########################################################################################
number = ""
########################################################################################
name = f"CoreEnergyForDatabase{number}"
new_dicts = [
    {"core_energy.scaling_for_plaq4": 0, "core_energy.scaling_model": None},
    {"core_energy.scaling_for_plaq4": 1000000, "core_energy.scaling_model": None},
    {"core_energy.only_squares_in_core": True},
    {"core_energy.only_squares_in_core": False},
    {"core_energy.polygon_object.exponent": 1},
    {"core_energy.polygon_object.exponent": 2},
    {"core_energy.polygon_object.exponent": 3},
    {"core_energy.all_constraints": True},
    {"core_energy.only_number_of_plaquettes": True},
]
functions_for_benchmarking.create_and_save_settings(name, new_dicts, new_config, problem_folder='training_set')

########################################################################################
    
name = f"CoreMcForDatabase{number}"
new_dicts = [
    {"core_cluster_shuffling_probability": 0},
    {"core_cluster_shuffling_probability": 0.05},
    {"core_cluster_shuffling_probability": 0.1},
    {"core_cluster_shuffling_probability": 0.15},
    {
        "core_only_four_cycles_for_ancillas": False,
        "core_energy.only_squares_in_core": False,
    },
    {"core_ancilla_insertion_probability": 0.01, "core_ancilla_deletion_probability": 0},
    {"core_ancilla_insertion_probability": 0.02, "core_ancilla_deletion_probability": 0},
    {"core_ancilla_insertion_probability": 0.04, "core_ancilla_deletion_probability": 0},
    {"core_ancilla_insertion_probability": 0.08, "core_ancilla_deletion_probability": 0},
    {"core_ancilla_insertion_probability": 0.01, "core_ancilla_deletion_probability": 1},
    {"core_ancilla_insertion_probability": 0.02, "core_ancilla_deletion_probability": 1},
    {"core_ancilla_insertion_probability": 0.04, "core_ancilla_deletion_probability": 1},
    {"core_ancilla_insertion_probability": 0.08, "core_ancilla_deletion_probability": 1},

]
functions_for_benchmarking.create_and_save_settings(name, new_dicts, new_config, problem_folder='training_set')

########################################################################################

name = f"McForDatabase{number}"
new_dicts = [
    {"swap_only_core_qbits_in_line_swaps": False},
    {"swap_only_core_qbits_in_line_swaps": True},
    {"swap_probability": 0.05, "decay_rate_of_swap_probability": 0.95},
    {"swap_probability": 0.1, "decay_rate_of_swap_probability": 0.95},
    {"swap_probability": 0.15, "decay_rate_of_swap_probability": 0.95},
    {
        "shell_time": 50,
        "envelop_shell_search": True,
        "finite_grid_size": False,
    },
    {
        "shell_time": 200,
        "envelop_shell_search": True,
        "finite_grid_size": False,
    },
    {
        "random_qbit": True,
    },
    {
        "number_of_plaquettes_weight": True,
        "random_qbit": False,
    },
    {
        "sparse_plaquette_density_weight": True,
        "random_qbit": False,
    },
    {
        "length_of_node_weight": 0.05,
        "random_qbit": False,
    },
    {
        "length_of_node_weight": 0.05,
        "envelop_shell_search": True,
        "random_qbit": False,
        "finite_grid_size": False,
    },
    {
        "qbit_with_same_node": True,
        "random_qbit": False,
    },
]
functions_for_benchmarking.create_and_save_settings(name, new_dicts, new_config, problem_folder='training_set')

########################################################################################
name = f"EnergyForLHZGraphs{number}"
individual_settings1 = [
    [        
        {"energy.polygon_object.scope_measure": False},
        {"energy.polygon_object.scope_measure": True},

    ],
    [
        {"energy.scaling_model": 'INIT'},
    ],
    [
        {"energy.polygon_object.exponent":0.5},
        {"energy.polygon_object.exponent":2},
        {"energy.polygon_object.exponent":3},
    ],
]

individual_settings2 = [

    [
        {"energy.scaling_for_plaq3": 0, "energy.scaling_for_plaq4": 0, "energy.scaling_model":None},
        {"energy.scaling_for_plaq3": 1000000, "energy.scaling_for_plaq4": 1000000, "energy.scaling_model":None},
        {"energy.scaling_model": 'LHZ'},
        {"energy.scaling_model": 'INIT'},
    ]
]

individual_settings3 = [
    [
        {"energy.decay_weight": True, "energy.decay_rate": 0.5,
        "energy.line": False},
        {"energy.decay_weight": True, "energy.decay_rate": 1,
        "energy.line": False},
        {"energy.decay_weight": True, "energy.decay_rate": 1.5,
        "energy.line": False},
        
        {"energy.decay_weight": True, "energy.decay_rate": 1,
        "energy.line": True, "energy.line_exponent": 1},
        {"energy.decay_weight": True, "energy.decay_rate": 1,
        "energy.line": True, "energy.line_exponent": 2},
        
        {"energy.line": True, "energy.line_exponent": 1,
        "energy.basic_energy": True,},
        {"energy.line": True, "energy.line_exponent": 2,
        "energy.basic_energy": True,},
        
        {"energy.line": True, "energy.line_exponent": 1,
        "energy.basic_energy": False,},
        {"energy.line": True, "energy.line_exponent": 2,
        "energy.basic_energy": False,},
    ]
]

individual_settings4 = [
    [
        {"energy.all_constraints":True,
         "energy.scaling_for_plaq3": 0,
         "energy.scaling_for_plaq4": 0,
         "energy.scaling_model":None},
        {"energy.all_constraints": True,
         "energy.scaling_model": 'LHZ'},
        {"energy.count_constraints": True,
         "energy.scaling_model":None,
         "chi_0": 0.001,}
    ]
]
individual_settings1 = list(itertools.product(*individual_settings1))
individual_settings2 = list(itertools.product(*individual_settings2))
individual_settings3 = list(itertools.product(*individual_settings3))
individual_settings4 = list(itertools.product(*individual_settings4))

individual_settings = (individual_settings1 + individual_settings2 + individual_settings3 + individual_settings4)
new_dicts = [{k: v for d in L for k, v in d.items()} for L in individual_settings]
functions_for_benchmarking.create_and_save_settings(
    f"{name}", new_dicts, new_config, problem_folder='lhz'
)
########################################################################################
name = f"McForLHZGraphs{number}"
individual_settings = [
    [     
        {"repetition_rate_factor": 2},
        {"repetition_rate_factor": 4},
        {"repetition_rate_factor": 8},
        {"temperature_adaptive": True, "temperature_kirkpatrick": False},
        {"temperature_kirkpatrick_sigma": True, "temperature_kirkpatrick": False},
        {"linear_in_moves": True, "temperature_kirkpatrick": False},
        {"temperature_C": True, "temperature_kirkpatrick": False},
        {"temperature_linearC": True, "temperature_kirkpatrick": False},
        {"alpha":0.8},
        {"alpha":0.9},
        {"alpha":0.95},
        {"with_core":True},
    ],
]
individual_settings = list(itertools.product(*individual_settings))
new_dicts = [{k: v for d in L for k, v in d.items()} for L in individual_settings]
functions_for_benchmarking.create_and_save_settings(
    f"{name}", new_dicts, new_config, problem_folder='lhz'
)
########################################################################################
name = f"AdvancedMcForLHZGraphsWithCore{number}"
individual_settings = [
    [
        {'with_core':True},
    ],
    [
        {"swap_only_core_qbits_in_line_swaps": False},
        {"swap_only_core_qbits_in_line_swaps": True},
        {"swap_probability": 0.05, "decay_rate_of_swap_probability": 0.95},
        {"swap_probability": 0.1, "decay_rate_of_swap_probability": 0.95},
        {"swap_probability": 0.15, "decay_rate_of_swap_probability": 0.95},
        {
            "shell_time": 50,
            "envelop_shell_search": True,
            "finite_grid_size": False,
        },
        {
            "shell_time": 200,
            "envelop_shell_search": True,
            "finite_grid_size": False,
        },
        {
            "random_qbit": True,
        },
        {
            "number_of_plaquettes_weight": True,
            "random_qbit": False,
        },
        {
            "sparse_plaquette_density_weight": True,
            "random_qbit": False,
        },
        {
            "length_of_node_weight": 0.05,
            "random_qbit": False,
        },
        {
            "length_of_node_weight": 0.05,
            "envelop_shell_search": True,
            "random_qbit": False,
            "finite_grid_size": False,
        },
        {
            "qbit_with_same_node": True,
            "random_qbit": False,
        },
    ]
]
individual_settings = list(itertools.product(*individual_settings))
new_dicts = [{k: v for d in L for k, v in d.items()} for L in individual_settings]
functions_for_benchmarking.create_and_save_settings(name, new_dicts, new_config, problem_folder='lhz')
########################################################################################
name = f"McForDatabaseWithCore{number}"
individual_settings = [
    [
        {'with_core':True},
    ],
    [
        {"swap_only_core_qbits_in_line_swaps": False},
        {"swap_only_core_qbits_in_line_swaps": True},
        {"swap_probability": 0.05, "decay_rate_of_swap_probability": 0.95},
        {"swap_probability": 0.1, "decay_rate_of_swap_probability": 0.95},
        {"swap_probability": 0.15, "decay_rate_of_swap_probability": 0.95},
        {
            "shell_time": 50,
            "envelop_shell_search": True,
            "finite_grid_size": False,
        },
        {
            "shell_time": 200,
            "envelop_shell_search": True,
            "finite_grid_size": False,
        },
        {
            "random_qbit": True,
        },
        {
            "number_of_plaquettes_weight": True,
            "random_qbit": False,
        },
        {
            "sparse_plaquette_density_weight": True,
            "random_qbit": False,
        },
        {
            "length_of_node_weight": 0.05,
            "random_qbit": False,
        },
        {
            "length_of_node_weight": 0.05,
            "envelop_shell_search": True,
            "random_qbit": False,
            "finite_grid_size": False,
        },
        {
            "qbit_with_same_node": True,
            "random_qbit": False,
        },
    ]
]
individual_settings = list(itertools.product(*individual_settings))
new_dicts = [{k: v for d in L for k, v in d.items()} for L in individual_settings]
functions_for_benchmarking.create_and_save_settings(name, new_dicts, new_config, problem_folder='training_set')
