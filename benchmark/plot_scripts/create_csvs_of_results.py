from CompilerQC import *
import pickle
import os
import pandas as pd
import multiprocessing
from pathlib import Path

def load_object(path):
    try:
        file = open(path, "rb")
        mc = pickle.load(file)
        file.close()
        return mc
    except:
        print('empty input')
        return None

def create_csv_from_results(folder_and_number):
    folder, number = folder_and_number
    df = pd.DataFrame()
    for setting in os.listdir(paths.benchmark_results_path / f"run_{number}" / folder):
        number_of_setting = int(setting.split("_")[1])
        for object_path in os.listdir(paths.benchmark_results_path / f"run_{number}" / f"{folder}/{setting}"):
            N, K, C = [int(object_path.split("_")[-i]) for i in [4, 3, 2]]
            mc_object = load_object(paths.benchmark_results_path / f"run_{number}" / f"{folder}/{setting}/{object_path}")
            if mc_object is None or type(mc_object) is MC_core:
                # in this case, the object is empty or we dont evalute MC_core objects
                continue                
            number_of_CNOTs, number_of_swap_gates, C_3, C_4 = mc_object.number_of_CNOTs()
            number_of_ancillas = sum([qbit.ancilla for qbit in mc_object.energy.polygon_object.nodes_object.qbits])
            schedule = {'N': N,
                        'K':K,
                        'C':C,
                        'original_C':int(C - number_of_ancillas),
                        'C_density': (C - mc_object.number_of_plaquettes) / C,
                        'number_of_setting': number_of_setting,
                        'number_of_plaquettes':mc_object.number_of_plaquettes,
                        'number_of_3er_plaquettes': C_3,
                        'number_of_4er_plaquettes': C_4,
                        'core': mc_object.with_core,
                        'number_of_ancillas': number_of_ancillas,
                        'scaling': mc_object.energy.scaling_model,
                        'steps': mc_object.n_total_steps,
                        'moves': mc_object.n_moves,
                        'number_of_swap_gates': number_of_swap_gates, 
                        'number_of_CNOTs': number_of_CNOTs, 
                        'number_of_CNOTs_in_LHZ': mc_object.number_of_CNOTs_in_LHZ(N),
                        'scope_measure': mc_object.energy.polygon_object.scope_measure,
                        'polygon_exponent': mc_object.energy.polygon_object.exponent,
                        'path': f"run_{number}/{folder}/{setting}/{object_path}",
                        'complete': mc_object.energy.polygon_object.nodes_object.qbits.graph.complete,
                       }
            df = df.append(schedule, ignore_index=True)
    df.to_csv(paths.cwd / f"benchmark/plot_scripts/results_in_csvs/run_{number}/{folder.split('_')[-1]}.csv", index=False)
    print(f"saved {folder.split('_')[-1]}")

def create_csv_from_results_(folder, number):
    return create_csv_from_results(folder, number)
# Look at docs to see why "if __name__ == '__main__'" is necessary
if __name__ == "__main__":
    number = 2
    result_folders = os.listdir(paths.benchmark_results_path / f"run_{number}")
    Path(paths.cwd / f"benchmark/plot_scripts/results_in_csvs/run_{number}").mkdir(parents=True, exist_ok=True)
    # Create pool with 4 processors
    pool = multiprocessing.Pool(len(result_folders))
    # Create jobs
    jobs = []
    for folder in result_folders:
        print(folder, number)
        folder_and_number = (folder, number)
        # Create asynchronous jobs that will be submitted once a processor is ready
        jobs.append(pool.apply_async(create_csv_from_results, (folder_and_number, )))
    # Submit jobs
    results = [job.get() for job in jobs]
# Combine results
