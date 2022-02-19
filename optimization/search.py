import random
import itertools
import numpy as np
from scipy.special import binom
from copy import deepcopy
from CompilerQC import Polygons, core
from CompilerQC.objective_function import Energy
from shapely.geometry import LineString


class MC:
    def __init__(
        self,
        energy_object: Energy,
        temperatur_schedule: str = "1 / x",
        acceptance: str = "np.exp(- x / y)",
        n_moves: int = 100,
        schedule: dict = {},
        energy_schedule: dict = {},
    ):
        self.energy = energy_object
        self.polygon = energy_object.polygon
        self.temperatur_schedule = temperatur_schedule
        self.acceptance = acceptance
        self.schedule = schedule
        self.energy_schedule = energy_schedule
        self.n_total_steps = 0
        self.n_moves = n_moves


    def free_neighbour_coords(self, qbit_coords: list):
        """
        return list of all free neighbours for qbits_coords
        """
        neighbour_list = []
        for qbit_coord in qbit_coords:
            neighbour_list += Polygons.neighbours(qbit_coord)
        neighbour_list = list(set(neighbour_list) - set(qbit_coords))
        return neighbour_list

    # TODO: add radius to polygon.neighbour function, what if core is not in the middle and only a small part of the graph
    def random_coord_next_to_core(
        self,
        qbit: tuple = None,
        radius: float = None,
    ):
        """
        generate random coord next to core (fixed qbits)
        for random movable qbit
        """
        if qbit is None:
            qbit = random.choice(self.polygon.movable_qbits)
        new_coord = random.choice(self.free_neighbour_coords(self.polygon.core_coords))
        return [qbit], [new_coord]

    def swap_qbits(self):
        """
        swap qbits which are next to each other
        if no qbits are next to each other, swap any qbits
        """
        qbit_coords = self.polygon.movable_coords.copy()
        random.shuffle(qbit_coords)
        for coord in qbit_coords:
            coords_neighbour_qbits = list(set(
                    Polygons.neighbours(coord)
                    ).intersection(
                    self.polygon.movable_coords))
            if len(coords_neighbour_qbits):
                random.shuffle(coords_neighbour_qbits)
                coord_to_swap_with = coords_neighbour_qbits[0]
                coord_qbit_dict = {v:k for k,v in 
                        self.polygon.qbit_coord_dict.items()}
                qbit1 = coord_qbit_dict[coord]
                qbit2 = coord_qbit_dict[coord_to_swap_with]
                return [qbit1, qbit2], [coord_to_swap_with, coord]
        else:
            qbit1, qbit2 = random.sample(self.polygon.movable_qbits, 2)
            return [qbit1, qbit2], [qbit_coords[qbit2], qbit_coords[qbit1]]

    def swap_lines_in_core(
        self,
    ):
        """
        swap two logical nodes(from set U or V) only in core -> swap lines
        """
        try:
            U_or_V = random.sample([self.polygon.U, self.polygon.V], 1)[0]
            a, b = random.sample(U_or_V, 2)
            n = self.polygon.core_qbits
            n = [(-1, k) if i == a else (i, k) for i, k in n]
            n = [(i, -1) if k == a else (i, k) for i, k in n]
            n = [(a, k) if i == b else (i, k) for i, k in n]
            n = [(i, a) if k == b else (i, k) for i, k in n]
            n = [(b, k) if i == -1 else (i, k) for i, k in n]
            swapped_core_qbits = [(i, b) if k == -1 else (i, k) for i, k in n]
            sorted_swapped_core_qbits = list(
                map(lambda x: tuple(sorted(x)), swapped_core_qbits)
            )
            # detach core_coords from class
            core_coords = [coord for coord in self.polygon.core_coords]
            return sorted_swapped_core_qbits, core_coords
        except AttributeError:
            print('sets U and V arent defined yet, please define them to call this function')

    def most_distant_qbit_from_core(self):
        core_center = self.polygon.center_of_convex_hull(self.polygon.core_coords)
        distances = list(
            map(
                lambda coord: LineString([core_center, coord]).length,
                self.polygon.movable_coords,
            )
        )
        most_distand_qbit = self.polygon.movable_coords[distances.index(max(distances))]
        # TODO: add variance of distances, if small --> return None
        return [
            qbit
            for qbit, coord in self.polygon.qbit_coord_dict.items()
            if coord == most_distand_qbit
        ]

    def step(
        self,
        operation: str,
    ):
        self.current_polygon = deepcopy(self.polygon)
        current_energy = self.energy(self.polygon, self.energy_schedule)
        if operation == "contract":
            qbits, coords = self.random_coord_around_core_center()
        if operation == "swap":
            qbits, coords = self.swap_qbits()
        if operation == "swap_lines_in_core":
            qbits, coords = self.swap_lines_in_core()
        if operation == "grow_core":
            qbits, coords = self.random_coord_next_to_core()
        self.polygon.update_qbits_coords(qbits, coords)
        new_energy = self.energy(self.polygon, self.energy_schedule)
        return current_energy, new_energy

    def temperature(self, k: int=None):
        if k is None:
            k = self.n_total_steps
        temperatur = lambda x: eval(str(self.temperatur_schedule))
        return temperatur(k) + 1e-5

    def metropolis(self, current_energy, new_energy):
        self.n_total_steps += 1
        delta_energy = new_energy - current_energy
        acceptance_probability = lambda x, y: eval(str(self.acceptance))
        if random.random() < min([1, acceptance_probability(delta_energy, self.temperature())]):
            pass
        else:
            self.polygon = self.current_polygon
        return delta_energy

    def apply(
        self,
        operation,
        n_steps,
    ):
        for i in range(n_steps):
            current_energy, new_energy = self.step(operation)
            delta_energy = self.metropolis(current_energy, new_energy)
        # update polygon in energy
        self.energy(self.polygon, self.energy_schedule)

    def optimization_schedule(
            self,
            ):
        """
        schedule is a dict = {operation:n_steps}
        set schedule for the optimization
        """
        for operation, n_steps in self.schedule.items():
            self.apply(operation, n_steps)
