import random
import itertools
import numpy as np
from copy import deepcopy
from polygons import Polygons
from energy import Energy

class MC():

    def __init__(self, polygon_object: Polygons):
        self.energy = Energy(polygon_object)
        self.polygon = polygon_object

    def generate_new_coord_for_qbit(self, qbit: tuple=None, radius: float=None):
        if qbit is None:
            qbit = random.choice(list(self.polygon.qbit_coord_dict.keys()))
        center_x, center_y = map(int, self.polygon.center_of_convex_hull())
        if radius is None:
            radius = round(np.sqrt(self.polygon.C))
        x = np.arange(center_x - radius, center_x + radius)
        y = np.arange(center_y - radius, center_y + radius)
        x, y = np.meshgrid(x[x>=0], y[y>=0])
        random_coords_around_center = list(zip(x.flatten(), y.flatten()))
        np.random.shuffle(random_coords_around_center)
        qbit_coords = self.polygon.qbit_coord_dict.values()
        new_random_coord = None
        for coord in random_coords_around_center:
            if coord not in qbit_coords:
                new_random_coord = coord
                break
        if new_random_coord is None:
            qbit, new_random_coord = self.generate_new_coord_for_qbit(radius = radius ** 2) 
        return qbit, new_random_coord
        
    def swap_qbits(self):
        qbit_coords = self.polygon.qbit_coord_dict
        qbit1, qbit2 = random.sample(qbit_coords.keys(), 2)
        return [qbit1, qbit2], [qbit_coords[qbit1], qbit_coords[qbit2]]    

    def step(self, operation: str):
        self.current_qbit_coords = deepcopy(self.polygon.qbit_coord_dict)
        current_energy = self.energy(self.polygon, factors=self.factors)
        if operation == 'contract':
            qbits, new_coords = self.generate_new_coord_for_qbit()
        if operation == 'swap':
            qbits, new_coords = self.swap_qbits()
        self.polygon.update_qbits_coords(qbits, new_coords)
        new_energy = self.energy(self.polygon, factors=self.factors)
        return current_energy, new_energy

    # TODO: thats not metropolis, here you may stuck in local minima
    def metropolis(self, current_energy, new_energy):
        delta_energy = new_energy - current_energy
        if delta_energy < 0: 
            pass
        else:
            self.polygon.qbit_coord_dict = self.current_qbit_coords

    def apply_contract(self, n_steps):
        self.factors = [1, 1, 1, 1]
        for i in range(n_steps):
            current_energy, new_energy = self.step('contract')
            self.metropolis(current_energy, new_energy)


    def apply_swap(self, n_steps):
        self.factors = [1, 1, 1, 1]
        for i in range(n_steps):
            current_energy, new_energy = self.step('swap')
            self.metropolis(current_energy, new_energy)
