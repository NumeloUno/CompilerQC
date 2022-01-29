import numpy as np
import random
import argparse
from CompilerQC import paths
from datetime import datetime
from uuid import uuid4
import os
from parityos.visualize_results import visualize_parityos_output
from parityos import DeviceModel, Client
from tqdm import tqdm

# TODO: hide this information
user = "kerem"
password = "4k9ZF8k!w99geb"

client = Client(user, password)


def create_problem(client, num_constraints: int, square_plaquette_probability: int):
    """
    create logical problem and its compiled solution
    on a 8 x 8 grid, with num_constraints
    """
    # Set up the device
    x, y = 8, 8
    device_model = DeviceModel.rectangular_analog(x, y)

    settings = {
        # Maximum total number of constraints that will be put in the reverse compiled problem
        "num_constraints": num_constraints,
        # The probability of a square constraint (1 means only squares, 0 means only triangles)
        "square_plaquette_probability": square_plaquette_probability,
        # The maximum label length, so if 2, the logical problem will only have quadratic terms.
        "maximum_label_length": 2,
    }
    output, problem = client.reverse_engineer_problem(device_model, settings)

    return output


def save_problem(output, save_in_folder: str):
    """
    brings output from create_problem() in form
    for Graph class (adj_matrix) and Polygon class
    (qbit_coord_dict) and save them in .npy
    """

    coord_qbit_list = output.to_json()["mappings"]["encoding_map"]
    coords = [coord_qbit[0] for coord_qbit in coord_qbit_list]
    qbits = [coord_qbit[1][0] for coord_qbit in coord_qbit_list]

    # add new node to each single number qbit
    max_node = max([node for edge in qbits for node in edge])
    for i, l in enumerate(qbits):
        if len(l) == 1:
            qbits[i] += [max_node + 1]
    qbits = np.array([sorted(edge) for edge in qbits])
    qbit_coord_dict = dict(zip(map(tuple, qbits), map(tuple, coords)))

    adj_matrix = np.zeros((max_node + 2, max_node + 2))
    adj_matrix[qbits[:, 0], qbits[:, 1]] = 1
    adj_matrix = np.transpose(adj_matrix) + adj_matrix

    C = len(output.compiled_problem.constraints)
    K = len(qbits)
    N = np.abs(C - K - 1)

    eventid = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
    # Save
    os.makedirs(paths.database_path / save_in_folder / f"problem_N_{N}_K_{K}_C_{C}", exist_ok=True)
    dictionary = {"qbit_coord_dict": qbit_coord_dict, "graph_adj_matrix": adj_matrix}
    np.save(
        paths.database_path / save_in_folder / f"problem_N_{N}_K_{K}_C_{C}" / f"{eventid}.npy",
        dictionary,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate new samples for the database, containing graphs and their compiled solutions"
    )
    parser.add_argument(
        "-n",
        "--number_of_new_samples",
        type=int,
        default=100,
        help="how many new samples you want to generate",
    )
    parser.add_argument(
        "-C",
        "--min_max_num_consraints",
        type=list,
        default=[2, 20],
        help="generate samples with constraints C between [min, max]",
    )
    parser.add_argument(
        "-path",
        type=str,
        default="training_set",
        help="save samples in folder training_set, validation_set, lhz_set",
    )
    args = parser.parse_args()
    for sample in tqdm(range(args.number_of_new_samples), desc="Generate new samples"):
        for C in range(*args.min_max_num_consraints):
            square_plaquette_probability = random.uniform(0.2, 1)
            output = create_problem(client, C, square_plaquette_probability)
            # visualize_parityos_output(output)
            save_problem(output, save_in_folder=args.path)

# read_dictionary = np.load(paths.database_path / f'problem_N_{N}_K_{K}_C_{C}' / filename,allow_pickle='TRUE').item()
