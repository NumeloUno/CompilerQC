import numpy as np
from CompilerQC import Graph, Qbits, paths
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
    qbits = Qbits.init_qbits_from_dict(graph, dict(zip(graph.qbits, graph.qbits)))
    qubit_to_coord_dict = qbits.qubit_to_coord_dict
    adj_matrix = graph.adj_matrix
    
    return adj_matrix, qubit_to_coord_dict, [graph.N, graph.K, graph.C]


    
for N_ in tqdm(range(4, 18), desc="Generate lhz samples"):
    adj_matrix, qubit_to_coord_dict, NKC = generate_LHZ_problem(N_)
    N, K, C = NKC
    assert N_ == N, "the expected N is not seen, some error in generate_LHZ_problem"
    eventid = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
    # Save
    save_in_folder = "lhz"
    os.makedirs(paths.database_path / save_in_folder / f"problem_N_{N}_K_{K}_C_{C}", exist_ok=True)
    dictionary = {"qbit_coord_dict": qubit_to_coord_dict, "graph_adj_matrix": adj_matrix}
    np.save(
        paths.database_path / save_in_folder / f"problem_N_{N}_K_{K}_C_{C}" / f"LHZ_N_{N}.npy",
        dictionary,
    )
        
        
  