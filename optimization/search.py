import random
import itertools
import numpy as np
from scipy.special import binom
from copy import deepcopy
from CompilerQC import Polygons, core
from CompilerQC.objective_function import Energy
from shapely.geometry import LineString

class MC():

    def __init__(
            self,
            polygon_object: Polygons,
            temperature: float,
            ):
        self.energy = Energy(polygon_object)
        self.polygon = polygon_object
        self.temperature = temperature
        self.factors = [1, 1, 1, 1]


    def random_coord_around_core_center(
            self,
            qbit: tuple=None,
            radius: float=None,
            ):
        """
        return random coord around center of core(! not convex hull), if this coord is free
        search in range of radius, if everything is occupied, search again with 
        increased radius
        """
        if qbit is None:
            qbit = random.choice(self.polygon.movable_qbits)
        core_qbit_coords = [self.polygon.qbit_coord_dict[qbit]
                for qbit in self.polygon.core_qbits]
        center_x, center_y = map(int, 
                self.polygon.center_of_convex_hull(
                    core_qbit_coords)
                )
        if radius is None:
            radius = round(np.sqrt(self.polygon.C))
        x = np.arange(center_x - radius, center_x + radius)
        y = np.arange(center_y - radius, center_y + radius)
        x, y = np.meshgrid(x, y)
        random_coords_around_center = list(zip(x.flatten(), y.flatten()))
        np.random.shuffle(random_coords_around_center)
        qbit_coords = self.polygon.qbit_coord_dict.values()
        free_random_coords_around_center = list(
                set(random_coords_around_center) - set(qbit_coords)
                )
        if len(free_random_coords_around_center) > 0:
            new_random_coord = random.choice(free_random_coords_around_center)
        else:
            qbit, new_random_coord = [x[0] for x in 
                    self.random_coord_around_core_center(radius = int(radius + 1))] 
        return [qbit], [new_random_coord]
    
    
    def free_neighbour_coords(
        self,
        qbit_coords: list
    ):
        """
        return list of all free neighbours for qbits_coords
        """
        neighbour_list = []
        for qbit_coord in qbit_coords: 
            neighbour_list += Polygons.neighbours(qbit_coord)
        neighbour_list = list(
            set(neighbour_list) - set(qbit_coords)) 
        return neighbour_list
    
    # TODO: add radius to polygon.neighbour function
    def random_coord_next_to_core(
            self,
            qbit: tuple=None,
            radius: float=None,
            ):
        """
        generate random coord next to core (fixed qbits)
        for random movable qbit
        """
        core_coords = [c for q, c in
                self.polygon.qbit_coord_dict.items() 
                if q in self.polygon.core_qbits]
        if qbit is None:
            qbit = random.choice(self.polygon.movable_qbits)
        new_coord = random.choice(self.free_neighbour_coords(
            core_coords))
        return [qbit], [new_coord]
         

    def swap_qbits(self):
        qbit_coords = self.polygon.qbit_coord_dict
        qbit1, qbit2 = random.sample(self.polygon.movable_qbits, 2)
        return [qbit1, qbit2], [qbit_coords[qbit2], qbit_coords[qbit1]]    

    # TODO: could be that a, b not in core if not LHZ!
    def swap_lines_in_core(
            self,
            ):
        """
        swap two logical nodes or only in core -> swap lines
        """
        a, b = random.sample(range(0, self.polygon.N), 2)
        n = self.polygon.core_qbits
        n = [(-1,  k) if i== a else (i, k) for i, k in n]
        n = [( i, -1) if k== a else (i, k) for i, k in n]
        n = [( a,  k) if i== b else (i, k) for i, k in n]
        n = [( i,  a) if k== b else (i, k) for i, k in n]
        n = [( b,  k) if i==-1 else (i, k) for i, k in n]
        swapped_core_qbits = [( i,  b) if k==-1 else (i, k) for i, k in n]
        sorted_swapped_core_qbits = list(
                map(lambda x: tuple(sorted(x)), new_qbits))
        core_coords = [c for q, c in self.polygon.qbit_coord_dict.items()
                if q in self.polygon.core_qbits]
        return sorted_swapped_core_qbits, core_coords
                      
    def most_distant_qbit_from_core(self):
        core_coords = [c for q, c in self.polygon.qbit_coord_dict.items()
                if q in self.polygon.core_qbits]
        core_center = self.polygon.center_of_convex_hull(core_coords)
        movable_coords = [coord for qbit, coord in 
                self.polygon.qbit_coord_dict.items()]
        distances = list(map(lambda coord: 
                        LineString([core_center, coord]).length,
                        movable_coords))
        most_distand_qbit = movable_coords[distances.index(max(distances))]
        # TODO: add variance of distances, if small --> return None
        return [qbit for qbit, coord in self.polygon.qbit_coord_dict.items()
            if coord == most_distand_qbit]


    def step(
            self,
            operation: str,
            ):
        self.current_qbit_coords = deepcopy(self.polygon.qbit_coord_dict)
        current_energy = self.energy(self.polygon, factors=self.factors)
        qbit = self.most_distant_qbit_from_core()[0]
        if operation == 'contract':
            qbits, coords = self.random_coord_around_core_center(qbit)
        if operation == 'swap':
            qbits, coords = self.swap_qbits()
        if operation == 'swap_lines_in_core':
            qbits, coords = self.swap_lines_in_core()
        if operation == 'grow_core':
            qbits, coords = self.random_coord_next_to_core(qbit)
        self.polygon.update_qbits_coords(qbits, coords)
        new_energy = self.energy(self.polygon, factors=self.factors)
        return current_energy, new_energy

    
    def update_temperature(
            self,
            temperature:float=None):
        if temperature is None:
            num_of_found_plaqs = self.energy.distance_to_plaquette().count(0)
            still_to_find = self.polygon.C - num_of_found_plaqs
            temperature = still_to_find * 0.001 # this factor seems to have an effect
        self.temperature = temperature
    
    def metropolis(self, current_energy, new_energy):
        delta_energy = new_energy - current_energy
        if random.random() < min([1, np.exp(- delta_energy / self.temperature)]):
            pass
        else:
            self.polygon.qbit_coord_dict = self.current_qbit_coords
        return delta_energy

            
    def apply(
            self,
            operation,
            n_steps,
            ):
        for i in range(n_steps):
            current_energy, new_energy = self.step(operation)
            delta_energy = self.metropolis(current_energy, new_energy)
            #if i % 100 == 0:
            #    self.polygon.move_center_to_middle()
            return delta_energy
                
                      
