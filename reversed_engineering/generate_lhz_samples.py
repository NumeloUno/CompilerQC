import numpy as np
from CompilerQC import Graph
from CompilerQC import Polygons
from CompilerQC import paths
from datetime import datetime
from uuid import uuid4
import os
from tqdm import tqdm

def generate_LHZ_problem(N: int):
    """
    return the adj matrix for a complete connected graph 
    of size N and its compiled solution for the 
    physical graph as qbit coord dictionary
    """
    graph = Graph.complete(N)
    polygon_object = Polygons(graph)
    lhz_coords = qbits = polygon_object.qbits
    polygon_object.update_qbits_coords(qbits, lhz_coords)
    qbit_coord_dict = polygon_object.qbit_coord_dict
    adj_matrix = graph.adj_matrix

    C = polygon_object.C
    K = len(qbits)
    N = np.abs(C - K - 1)
    return adj_matrix, qbit_coord_dict

if __name__ == "__main__":
    
    for N in tqdm(range(4, 18), desc="Generate lhz samples"):
        adj_matrix, qbit_coord_dict = generate_LHZ_problem(N)
        eventid = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
        # Save
        save_in_folder = "lhz"
        os.makedirs(paths.database_path / save_in_folder / f"problem_N_{N}_K_{K}_C_{C}", exist_ok=True)
        dictionary = {"qbit_coord_dict": qbit_coord_dict, "graph_adj_matrix": adj_matrix}
        np.save(
            paths.database_path / save_in_folder / f"problem_N_{N}_K_{K}_C_{C}" / f"LHZ_N_{N}.npy",
            dictionary,
        )
