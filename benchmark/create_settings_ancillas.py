#!/usr/bin/env python
# coding: utf-8

# Here you can create the yaml files, which can be benchmarked afterwards.
# Therefore, the default.yaml in the parameters folder is updatet by the dictionary you will specify.
# Dont use _ in file names
#
# Move to folder and run e.g.:find -name "*yaml" | parallel python ../../benchmark_optimization.py --yaml_path={} -b=10 -minN=10 -maxN=10   or run sh ../run_....sh
# or move to csvs folder in parameters and run sh files, the csvs are created in this script, by calling the function create_... in functions_for_benchmarking, there you can also change settings for all csvs, thats not ideal i know

########################################################################################
number = "4"
delete = True
number_of_files_in_one_csv = 15
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
Path(paths.parameters_path / f"run_{number}/sh_scripts").mkdir(
    parents=True, exist_ok=True
)
Path(paths.parameters_path / f"run_{number}/dictionaries").mkdir(
    parents=True, exist_ok=True
)

# open default yaml and update it by general settings
with open(paths.parameters_path / "Default/default.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

default_update = {
    "energy.scaling_model": "INIT",
    "with_core": True,
    "envelop_shell_search": True,
    "finite_grid_size": False,
}
new_config = functions_for_benchmarking.update(config, default_update)
name = f"EnergyForDatabase{number}"
individual_settings1 = [
    [        
        {"energy.polygon_object.scope_measure": False,
        "energy.polygon_object.exponent":2},
        {"energy.polygon_object.scope_measure": True,
        "energy.polygon_object.exponent":2},
 
    ],
]
 
individual_settings2 = [
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
 
individual_settings = (individual_settings1 + individual_settings2)
new_dicts = [{k: v for d in L for k, v in d.items()} for L in individual_settings]
functions_for_benchmarking.create_and_save_settings(
    f"{name}", number, new_dicts, new_config, problem_folder='training_set'
)
with open(paths.parameters_path / f"run_{number}/dictionaries/{name}.pkl", "wb") as f:
    pickle.dump(new_dicts, f)
f.close()
########################################################################################