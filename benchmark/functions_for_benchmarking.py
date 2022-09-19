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
from uuid import uuid4

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
path_to_results = lambda args: (
    paths.benchmark_results_path
    / f"benchmark_{(args.problem_folder).replace('/','_')}_with_{args.id_of_benchmark}"
)
        
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
                if k.split(".")[1] == "polygon_object":
                    mc.energy.polygon_object.__setattr__(k.split(".")[2], v)
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
                if k.split(".")[1] == "polygon_object":
                    mc.energy.polygon_object.__setattr__(k.split(".")[2], v)
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
        mc.with_core = mc_schedule["with_core"]

    mc = update_mc(mc, mc_schedule, core)
    mc.energy.set_scaling_from_model()
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
    mc_schedule: dict,
    args: "arguments from parser",
    _id: str=str(uuid4()),
):
    """
    evaluate monte carlo/ simulated annealing schedule for batch_size timed and returns
    a success probability -> measure of how good the choosen schedule (temperature,
    configuration distribution, ...) is
    """
    mc = initialize_MC_object(graph, mc_schedule)
    if mc.with_core:
        if not graph.is_complete:
            mc_core = initialize_MC_object(graph, mc_schedule, core=True)
    else:
        mc.swap_probability = 0
    for iteration in range(args.batch_size):
        # search for core
        if mc.with_core:
            (
                qubit_coord_dict,
                mc.energy.polygon_object.core_corner,
                ancillas_in_core,
            ) = search_max_core(mc_core, graph.K)
            if not graph.is_complete:
                # if core is empty, dont allow swaps
                if len(qubit_coord_dict) == 0:
                    mc.swap_probability = 0
                    mc.finite_grid_size = True
                    mc.random_qbit = True
                else:
                    save_object(mc_core, path_to_results(args) / f'{args.name}/{_id}_CORE_N_K_C_{graph.N}_{graph.K}_{graph.C}_.pkl')

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
            mc.apply(mc.n_moves)
        # remove ancillas which dont reduce d.o.f
        mc.remove_ancillas(
            mc.energy.polygon_object.nodes_object.propose_ancillas_to_remove()
        )
        mc.name, mc.batch_size = args.name, args.batch_size
        save_object(mc, path_to_results(args) / f'{args.name}/{_id}_N_K_C_{graph.N}_{graph.K}_{graph.C}_.pkl')
        # reset mc object
        mc.reset(current_temperature=mc.T_0, remove_ancillas=True, keep_core=False)

def run_benchmark(
    graph: Graph,
    args,
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
    evaluate_optimization(
        graph, mc_schedule, args
    )


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
    search.visualize_search_process(mc, name, number_of_images)# before or after remove ancillas?
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


def search_max_core(mc_core, graph):
    """
    is graph is complete connected graph, do not search for the maximim merged graph,
    but simply take the maximum complete bipartite core
    """
    if graph.is_complete:
        core_qbit_to_core_dict = Polygons.move_to_center(graph.max_complete_bipartite_core_for_LHZ(), graph.K)
        core_corner = Polygons.corner_of_coords(list(core_qbit_to_core_dict.values()))
        ancillas_in_core = []
    else:
        mc_core.apply(mc_core.n_moves)
        (
            core_qbit_to_core_dict,
            core_corner,
            ancillas_in_core,
        ) = mc_core.energy.qbits_in_max_core(graph.K)
    return core_qbit_to_core_dict, core_corner, ancillas_in_core


def update(origin_dict, new_dict):
    updated_dict = origin_dict.copy()
    updated_dict.update(new_dict)
    return updated_dict


def create_and_save_settings(name, new_dicts, new_config):
    """update default yaml by newdict and save them in mc_parameters_name.yaml
    note: there are absolute paths in use!"""
    (paths.parameters_path / "csvs").mkdir(parents=True, exist_ok=True)
    filenames = []
    for idx, new_dict in enumerate(new_dicts):
        # create folder if it doesnt exists yet
        filenames.append(f"mc_parameters_{name}_{idx}.yaml")
        (paths.parameters_path / name).mkdir(parents=True, exist_ok=True)
        dict_to_save = update(deepcopy(new_config), new_dict)
        with open(
            paths.parameters_path / name / f"mc_parameters_{name}_{idx}.yaml", "w"
        ) as f:
            print(name, idx)
            yaml.dump(dict_to_save, f, default_flow_style=False)
    pd.DataFrame(data={"filenames":filenames}).to_csv(paths.parameters_path / "csvs" / f"{name}.csv", index=False, index_label=False)        
    with open(paths.parameters_path / "csvs" / f"run_{name}.sh", 'w') as sh:
        sh.write(('''\
#!/bin/bash 
NUM_PARALLEL_JOBS=10 
CSV_FILE=%s.csv
tail -n +2 ${CSV_FILE} | parallel --progress --colsep ',' -j${NUM_PARALLEL_JOBS} \
python "../../benchmark_optimization.py --yaml_path={1} --batch_size=20 \
--min_N=4 --max_N=40 --min_C=3 --max_C=91 --max_size=50 \
--problem_folder=problems_by_square_density/fsquare_density_of_1.0" 
        ''')%(name))
    with open(paths.plots / "scripts_to_create_gifs" / f"create_gif_for_{name}.sh", 'w') as sh:
        sh.write(('''\
#!/bin/bash 
NUM_PARALLEL_JOBS=1
CSV_FILE=../../parameters/csvs/%s.csv
tail -n +2 ${CSV_FILE} | parallel --progress --colsep ',' -j${NUM_PARALLEL_JOBS} \
python "../../benchmark_optimization.py --yaml_path={1} \
--min_N=4 --max_N=40 --min_C=3 --max_C=91 --max_size=1 \
--problem_folder=problems_by_square_density/fsquare_density_of_1.0  \
--visualize=True --number_of_core_images=50 --number_of_images=1000" 
        ''')%(name))