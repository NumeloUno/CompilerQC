#!/usr/bin/env python
# coding: utf-8

# Here you can create the yaml files, which can be benchmarked afterwards.
# Therefore, the default.yaml in the parameters folder is updatet by the dictionary you will specify.
# Dont use _ in file names
#
# Move to folder and run e.g.:find -name "*yaml" | parallel python ../../benchmark_optimization.py --yaml_path={} -b=10 -minN=10 -maxN=10   or run sh ../run_....sh
# or move to csvs folder in parameters and run sh files, the csvs are created in this script, by calling the function create_... in functions_for_benchmarking, there you can also change settings for all csvs, thats not ideal i know

########################################################################################
number = "5"
delete = True
########################################################################################

from pathlib import Path
import shutil
import os
import itertools
import pickle
import yaml
from CompilerQC import *
# delete all folder
if delete:
    print("===========================================================")
    print("==== Delete old settings/sh_scripts/csvs/dictionaties =====")
    print("===========================================================")
    shutil.rmtree(paths.parameters_path / f"run_{number}", ignore_errors=True)

print("===========================================================")
print("========== Create new settings (yamls and csvs) ===========")
print("===========================================================")
Path(paths.parameters_path / f"run_{number}").mkdir(parents=True, exist_ok=True)
Path(paths.parameters_path / f"run_{number}/csvs").mkdir(parents=True, exist_ok=True)
Path(paths.parameters_path / f"run_{number}/sh_scripts").mkdir(parents=True, exist_ok=True)
Path(paths.parameters_path / f"run_{number}/dictionaries").mkdir(parents=True, exist_ok=True)
    
# open default yaml and update it by general settings
with open(paths.parameters_path / "Default/default.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
default_update = {
    "energy.polygon_object.scope_measure": False,
    "energy.scaling_model": "INIT",
    "energy.polygon_object.exponent": 1,
    "with_core": True,
    "envelop_shell_search": True,
    "shell_time": 20,
    "finite_grid_size": False,
    "core_energy.only_squares_in_core": False,
    "min_plaquette_density_in_softcore": 0.75,
    "core_only_four_cycles_for_ancillas": False,

}
new_config = functions_for_benchmarking.update(config, default_update)

########################################################################################
    
name = f"CoreMcForDatabase{number}"
new_dicts = [
    {"core_ancilla_insertion_probability": 0.01, "core_ancilla_deletion_probability": 1},
    {"core_ancilla_insertion_probability": 0.02, "core_ancilla_deletion_probability": 1},
    {"core_ancilla_insertion_probability": 0.04, "core_ancilla_deletion_probability": 1},
    {"core_ancilla_insertion_probability": 0.08, "core_ancilla_deletion_probability": 1},
    {"core_ancilla_insertion_probability": 0.16, "core_ancilla_deletion_probability": 1},
    {"core_ancilla_insertion_probability": 0.32, "core_ancilla_deletion_probability": 1},

]
functions_for_benchmarking.create_and_save_settings(name, number, new_dicts, new_config, problem_folder='training_set')
with open(paths.parameters_path / f"run_{number}/dictionaries/{name}.pkl", "wb") as f:
    pickle.dump(new_dicts, f)
f.close()