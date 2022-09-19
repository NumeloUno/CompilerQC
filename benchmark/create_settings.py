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
print("===================================")
print("==== Delete old settings/csvs =====")
print("===================================")
for p in Path("parameters/").glob("*"):
    if p.name not in ["run_script.sh", "run_parameters.sh", "Default"]:
        print(p.name)
        shutil.rmtree(f"parameters/{p.name}", ignore_errors=True)
print("===================================")
print("======== Delete old logs ==========")
print("===================================")
for p in Path("logs/").glob("*"):
    print(p.name)
    shutil.rmtree(f"logs/{p.name}", ignore_errors=True)
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
# open default yaml and update it by default settings
with open(paths.parameters_path / "Default/default.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
default_update = {
    "with_core": True,
    "random_qbit": True,
    "finite_grid_size": True,
    "padding_finite_grid": 2,
    "repetition_rate_factor": 5,
    "temperature_kirkpatrick": True,
    "core_temperature_kirkpatrick": True,
    "core_alpha": 0.99,
    "alpha": 0.99,
    "init_T_by_std": True,
    "core_energy.only_largest_core": True,
    "core_energy.only_squares_in_core": True,
    "core_energy.scaling_model": "LHZ",
    "energy.scaling_model": "INIT",
    "core_only_four_cycles_for_ancillas": True,
}
new_config = functions_for_benchmarking.update(config, default_update)


name = "CoreEnergy"
new_dicts = [
    {"core_energy.scaling_model": "MLP"},
    {"core_energy.scaling_model": "INIT"},
    {"core_energy.scaling_model": "LHZ"},
    {"core_energy.scaling_for_plaq4": 0, "core_energy.scaling_model": None},
    {"core_energy.scaling_for_plaq4": 1000000, "core_energy.scaling_model": None},
    {"core_energy.only_squares_in_core": True},
    {"core_energy.only_squares_in_core": False},
    {"core_energy.spring_energy": True},
    {"core_energy.only_number_of_plaquettes": True},
]
functions_for_benchmarking.create_and_save_settings(name, new_dicts, new_config)


name = "CoreSearch"
new_dicts = [
    {"core_n_moves": 10},
    {"core_n_moves": 100},
    {"core_n_moves": 1000},
    {"core_repetition_rate_factor": 2},
    {"core_repetition_rate_factor": 4},
    {"core_temperature_adaptive": True, "core_temperature_kirkpatrick": False},
    {"core_temperature_kirkpatrick_sigma": True, "core_temperature_kirkpatrick": False},
    {"core_linear_in_moves": True, "core_temperature_kirkpatrick": False},
    {"core_temperature_C": True, "core_temperature_kirkpatrick": False},
    {"core_temperature_linearC": True, "core_temperature_kirkpatrick": False},
    {"core_cluster_shuffling_probability": 0},
    {"core_cluster_shuffling_probability": 0.1},
    {"core_cluster_shuffling_probability": 0.2},
    {"core_cluster_shuffling_probability": 0.3},
    {
        "core_only_four_cycles_for_ancillas": False,
        "core_energy.only_squares_in_core": False,
    },
    {"core_only_four_cycles_for_ancillas": True},
    {"core_ancilla_insertion_probability": 0.1, "core_ancilla_deletion_probability": 0},
    {"core_ancilla_insertion_probability": 0.2, "core_ancilla_deletion_probability": 0},
    {"core_ancilla_insertion_probability": 0.4, "core_ancilla_deletion_probability": 0},
    {"core_ancilla_insertion_probability": 0.5, "core_ancilla_deletion_probability": 0},
    {
        "core_ancilla_insertion_probability": 0.1,
        "core_ancilla_deletion_probability": 0.1,
    },
    {
        "core_ancilla_insertion_probability": 0.2,
        "core_ancilla_deletion_probability": 0.2,
    },
    {
        "core_ancilla_insertion_probability": 0.3,
        "core_ancilla_deletion_probability": 0.3,
    },
]
functions_for_benchmarking.create_and_save_settings(name, new_dicts, new_config)


name = "Energy"
new_dicts = [
    {"energy.scaling_model": "MLP"},
    {"energy.scaling_model": "INIT"},
    {"energy.scaling_model": "LHZ"},
    {"energy.scaling_model": None, "energy.scaling_for_plaq3": 0, "energy.scaling_for_plaq4": 0},
    {"energy.scaling_model": None, "energy.scaling_for_plaq3": 100, "energy.scaling_for_plaq4": 100},
    {"energy.scaling_model": None, "energy.scaling_for_plaq3": 1000, "energy.scaling_for_plaq4": 1000},
    {"energy.decay_weight": True, "energy.decay_rate": 1},
    {"energy.decay_weight": True, "energy.decay_rate": 2},
    {"energy.decay_weight": True, "energy.decay_rate": 4},
    {"energy.line": True, "energy.line_exponent": 1},
    {"energy.line": True, "energy.line_exponent": 2},
    {"energy.line": True, "energy.line_exponent": 4},
    {"energy.line": True, "energy.bad_line_penalty": 0},
    {"energy.line": True, "energy.bad_line_penalty": 5},
    {"energy.line": True, "energy.bad_line_penalty": 10},
    {
        "energy.decay_weight": True,
        "energy.decay_rate": 1,
        "energy.line": True,
        "energy.line_exponent": 1,
        "energy.bad_line_penalty": 0,
    },
    {
        "energy.decay_weight": True,
        "energy.decay_rate": 1,
        "energy.line": True,
        "energy.line_exponent": 2,
        "energy.bad_line_penalty": 1,
    },
    {
        "energy.decay_weight": True,
        "energy.decay_rate": 1,
        "energy.line": True,
        "energy.line_exponent": 4,
        "energy.bad_line_penalty": 2,
    },
    {"energy.subset_weight": False},
    {"energy.subset_weight": True},
    {"energy.polygon_object.scope_measure": True},
    {"energy.polygon_object.scope_measure": False},
    {"energy.polygon_object.exponent": 0.5},
    {"energy.polygon_object.exponent": 1},
    {"energy.polygon_object.exponent": 2},
    {"energy.polygon_object.exponent": 3},
    {"energy.polygon_object.exponent": 0.5, "scaling_for_plaq3": 1000, "scaling_for_plaq4": 1000},
    {"energy.polygon_object.exponent": 2, "scaling_for_plaq3": 1000, "scaling_for_plaq4": 1000},
    {"energy.sparse_density_penalty": False},
    {"energy.sparse_density_penalty": True, "energy.sparse_density_factor": 5},
]
functions_for_benchmarking.create_and_save_settings(
    f"{name}WithCore", new_dicts, new_config
)
new_config_without_core = functions_for_benchmarking.update(
    new_config, {"with_core": False}
)
functions_for_benchmarking.create_and_save_settings(
    name, new_dicts, new_config_without_core
)


name = "Search"
new_dicts = [
    {"repetition_rate_factor": 2},
    {"repetition_rate_factor": 4},
    {"repetition_rate_factor": 8},
    {"temperature_adaptive": True, "temperature_kirkpatrick": False},
    {"temperature_kirkpatrick_sigma": True, "temperature_kirkpatrick": False},
    {"linear_in_moves": True, "temperature_kirkpatrick": False},
    {"temperature_C": True, "temperature_kirkpatrick": False},
    {"temperature_linearC": True, "temperature_kirkpatrick": False},
    {"swap_only_core_qbits_in_line_swaps": False},
    {"swap_only_core_qbits_in_line_swaps": True},
    {"swap_probability": 0},
    {"swap_probability": 0.05},
    {"swap_probability": 0.1},
    {"swap_probability": 0.2},
    {"swap_probability": 0.1, "decay_rate_of_swap_probability": 0.9},
    {"swap_probability": 0.1, "decay_rate_of_swap_probability": 0.6},
    {"swap_probability": 0.1, "decay_rate_of_swap_probability": 0.3},
    {"finite_grid_size": True, "padding_finite_grid": 2},
    {
        "min_plaquette_density_in_softcore": 0.75,
        "shell_time": 50,
        "envelop_shell_search": True,
        "finite_grid_size": False,
    },
    {
        "min_plaquette_density_in_softcore": 0.75,
        "envelop_shell_search": True,
        "finite_grid_size": False,
    },
    {
        "min_plaquette_density_in_softcore": 0.85,
        "envelop_shell_search": True,
        "finite_grid_size": False,
    },
    {
        "min_plaquette_density_in_softcore": 0.95,
        "envelop_shell_search": True,
        "finite_grid_size": False,
    },
    {
        "min_plaquette_density_in_softcore": 0.75,
        "shell_time": 50,
        "shell_search": True,
        "finite_grid_size": False,
    },
    {
        "min_plaquette_density_in_softcore": 0.75,
        "shell_search": True,
        "finite_grid_size": False,
    },
    {
        "min_plaquette_density_in_softcore": 0.85,
        "shell_search": True,
        "finite_grid_size": False,
    },
    {
        "min_plaquette_density_in_softcore": 0.95,
        "shell_search": True,
        "finite_grid_size": False,
    },
    {
        "random_qbit": True,
    },
    {
        "number_of_plaquettes_weight": 0.2,
        "random_qbit": False,
    },
    {
        "number_of_plaquettes_weight": 0.4,
        "random_qbit": False,
    },
    {
        "number_of_plaquettes_weight": 0.6,
        "random_qbit": False,
    },
    {
        "number_of_plaquettes_weight": 0.8,
        "random_qbit": False,
    },
    {
        "sparse_plaquette_density_weight": 0.2,
        "random_qbit": False,
    },
    {
        "sparse_plaquette_density_weight": 0.4,
        "random_qbit": False,
    },
    {
        "sparse_plaquette_density_weight": 0.6,
        "random_qbit": False,
    },
    {
        "sparse_plaquette_density_weight": 0.8,
        "random_qbit": False,
    },
    {
        "length_of_node_weight": 0.2,
        "random_qbit": False,
    },
    {
        "length_of_node_weight": 0.4,
        "random_qbit": False,
    },
    {
        "length_of_node_weight": 0.6,
        "random_qbit": False,
    },
    {
        "length_of_node_weight": 0.8,
        "random_qbit": False,
    },
    {
        "length_of_node_weight": 0.6,
        "shell_search": True,
        "random_qbit": False,
        "finite_grid_size": False,
    },
    {
        "length_of_node_weight": 0.6,
        "envelop_shell_search": True,
        "random_qbit": False,
        "finite_grid_size": False,
    },
    {
        "qbit_with_same_node": True,
        "random_qbit": False,
    },
    {
        "same_node_coords": True,
        "shell_search": True,
        "random_qbit": True,
        "finite_grid_size": False,
    },
    {
        "same_node_coords": True,
        "envelop_shell_search": True,
        "random_qbit": True,
        "finite_grid_size": False,
    },
]
functions_for_benchmarking.create_and_save_settings(
    f"{name}WithCore", new_dicts, new_config
)


name = "Search"
new_dicts = [
    {"repetition_rate_factor": 2},
    {"repetition_rate_factor": 4},
    {"repetition_rate_factor": 8},
    {"temperature_adaptive": True, "temperature_kirkpatrick": False},
    {"temperature_kirkpatrick_sigma": True, "temperature_kirkpatrick": False},
    {"linear_in_moves": True, "temperature_kirkpatrick": False},
    {"temperature_C": True, "temperature_kirkpatrick": False},
    {"temperature_linearC": True, "temperature_kirkpatrick": False},
    {"finite_grid_size": True, "padding_finite_grid": 2},
    {
        "random_qbit": True,
    },
    {
        "number_of_plaquettes_weight": 0.2,
        "random_qbit": False,
    },
    {
        "number_of_plaquettes_weight": 0.4,
        "random_qbit": False,
    },
    {
        "number_of_plaquettes_weight": 0.6,
        "random_qbit": False,
    },
    {
        "number_of_plaquettes_weight": 0.8,
        "random_qbit": False,
    },
    {
        "sparse_plaquette_density_weight": 0.2,
        "random_qbit": False,
    },
    {
        "sparse_plaquette_density_weight": 0.4,
        "random_qbit": False,
    },
    {
        "sparse_plaquette_density_weight": 0.6,
        "random_qbit": False,
    },
    {
        "sparse_plaquette_density_weight": 0.8,
        "random_qbit": False,
    },
    {
        "length_of_node_weight": 0.2,
        "random_qbit": False,
    },
    {
        "length_of_node_weight": 0.4,
        "random_qbit": False,
    },
    {
        "length_of_node_weight": 0.6,
        "random_qbit": False,
    },
    {
        "length_of_node_weight": 0.8,
        "random_qbit": False,
    },
]
new_config_without_core = functions_for_benchmarking.update(
    new_config, {"with_core": False}
)
functions_for_benchmarking.create_and_save_settings(
    name, new_dicts, new_config_without_core
)


name = "CompleteSearch"
import itertools

# open default yaml and update it by default settings
with open(paths.parameters_path / "Default/default.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
default_update = {
    "with_core": True,
    "random_qbit": True,
    "finite_grid_size": True,
    "padding_finite_grid": 2,
    "repetition_rate_factor": 5,
    "temperature_kirkpatrick": True,
    "core_temperature_kirkpatrick": True,
    "core_alpha": 0.99,
    "alpha": 0.99,
    "init_T_by_std": True,
    "core_energy.only_largest_core": True,
    "core_energy.only_squares_in_core": True,
    "core_energy.scaling_model": "MLP",
    "energy.scaling_model": "MLP",
    "core_only_four_cycles_for_ancillas": True,
    "core_repetition_rate_factor": 4,
    "core_cluster_shuffling_probability": 0.1,
    "repetition_rate_factor": 6,
    "swap_probability": 0.1,
    "decay_rate_of_swap_probability": 0.5,
}
new_config = functions_for_benchmarking.update(config, default_update)

individual_settings = [
    [
        {"core_energy.scaling_for_plaq4": 1000, "core_energy.scaling_model": None},
        {"core_energy.spring_energy": True},
        {"core_energy.only_number_of_plaquettes": True},
    ],
    [
        {
            "core_ancilla_insertion_probability": 0.1,
            "core_ancilla_deletion_probability": 0,
        },
        {
            "core_ancilla_insertion_probability": 0.5,
            "core_ancilla_deletion_probability": 0,
        },
    ],
    [
        {"energy.scaling_model": "MLP"},
        {"energy.scaling_for_plaq3": 1000, "energy.scaling_for_plaq4": 1000},
    ],
    [{}, {"energy.decay_weight": True, "energy.decay_rate": 1}],
    [
        {},
        {"energy.line": True, "energy.bad_line_penalty": 5, "energy.line_exponent": 2},
    ],
    [
        {"swap_only_core_qbits_in_line_swaps": False},
        {"swap_only_core_qbits_in_line_swaps": True},
    ],
    [
        {
            "min_plaquette_density_in_softcore": 0.75,
            "shell_time": 50,
            "envelop_shell_search": True,
            "finite_grid_size": False,
        },
        {
            "min_plaquette_density_in_softcore": 0.95,
            "shell_search": True,
            "finite_grid_size": False,
        },
        {
            "number_of_plaquettes_weight": 0.6,
            "random_qbit": False,
        },
        {
            "length_of_node_weight": 0.6,
            "shell_search": True,
            "random_qbit": False,
            "finite_grid_size": False,
        },
        {
            "length_of_node_weight": 0.6,
            "envelop_shell_search": True,
            "random_qbit": False,
            "finite_grid_size": False,
        },
    ],
]
individual_settings = list(itertools.product(*individual_settings))
new_dicts = [{k: v for d in L for k, v in d.items()} for L in individual_settings]
functions_for_benchmarking.create_and_save_settings(
    f"{name}WithCore", new_dicts, new_config
)


name = "EnergyForLHZGraphs"
# open default yaml and update it by default settings
with open(paths.parameters_path / "Default/default.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
default_update = {
    "random_qbit": True,
    "finite_grid_size": True,
    "padding_finite_grid": 1,
    "temperature_kirkpatrick": True,
    "alpha": 0.99,
    "init_T_by_std": True,
    "repetition_rate_factor": 6,
}
new_config = functions_for_benchmarking.update(config, default_update)

individual_settings = [
    [
        {"with_core": True},
        {"with_core": False}
    ],
    [
        {"energy.polygon_object.scope_measure": True},
        {"energy.polygon_object.scope_measure": False}
    ],
    [
        {"energy.scaling_for_plaq3": 0, "energy.scaling_for_plaq4": 0},
        {"energy.scaling_for_plaq3": 1000000, "energy.scaling_for_plaq4": 1000000},
        {"energy.scaling_model": 'LHZ'},
        {"energy.scaling_model": 'INIT'}
    ],
    [
#         {"energy.polygon_object.exponent":0.5},
        {"energy.polygon_object.exponent":1},
        {"energy.polygon_object.exponent":2},
#         {"energy.polygon_object.exponent":3},
#         {"energy.polygon_object.exponent":4},
#         {"energy.polygon_object.exponent":5},
#         {"energy.polygon_object.exponent":6},
    ],
    [
        {"energy.decay_weight": False},
        {"energy.decay_weight": True, "energy.decay_rate": 1},
#         {"energy.decay_weight": True, "energy.decay_rate": 1.5},        
    ],
    [
        {"energy.line": False},
        {"energy.line": True, "energy.line_exponent": 2},
#         {"energy.line": True, "energy.line_exponent": 3}
    ],
#     [    
#         {"energy.subset_weight": True},       
#         {"energy.subset_weight": False},
#     ],
#     [
#         {"energy.sparse_density_penalty": True},
#         {"energy.sparse_density_penalty": False},
#     ],
#     [
#         {"energy.low_noplaqs_penalty": True},
#         {"energy.low_noplaqs_penalty": False},
#     ]
]
individual_settings = list(itertools.product(*individual_settings))
new_dicts = [{k: v for d in L for k, v in d.items()} for L in individual_settings]
functions_for_benchmarking.create_and_save_settings(
    f"{name}", new_dicts, new_config
)

