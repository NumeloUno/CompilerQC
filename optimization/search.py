import random
import itertools
import numpy as np
from scipy.special import binom
from copy import deepcopy
from CompilerQC import Polygons, core
from CompilerQC.objective_function import Energy

class MC():

    def __init__(
            self,
            polygon_object: Polygons,
            ):
        self.energy = Energy(polygon_object)
        self.polygon = polygon_object
        self.factors = [1, 1, 1, 1]
#         self.movable_qbits = list(self.polygon.qbit_coord_dict.keys())
#         self.core_qbits = list()

#     def update_core_and_movable_qbits(
#             self,
#             fixed_qbits: list,
#             ):
#         """
#         fixed_coords: qbit coords which belong to e.g the core obtained 
#         by the bipartite graph ansatz 
#         """
#         all_qbits = set(self.polygon.qbit_coord_dict.keys())
#         self.
#         = fixed_qbits
#         self.movable_qbits = list(all_qbits - set(fixed_qbits))


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
    ):
        """
        return list of all free neighbours for all qbits
        """
        neighbour_list = []
        qbit_coords = self.polygon.qbit_coord_dict.values()

        for qbit_coord in qbit_coords: 
            neighbour_list += Polygons.neighbours(qbit_coord)
        neighbour_list = list(
            set(neighbour_list) - set(qbit_coords)) 
        return neighbour_list
    
    def random_coord_next_to_core(
            self,
            qbit: tuple=None,
            ):
        """
        generate random coord next to core (fixed qbits)
        for random movable qbit
        """
        if qbit is None:
            qbit = random.choice(self.polygon.movable_qbits)
        new_coord = random.choice(self.free_neighbour_coords())
        return [qbit], [new_coord]
         
    def swap_qbits(self):
        qbit_coords = self.polygon.qbit_coord_dict
        qbit1, qbit2 = random.sample(self.polygon.movable_qbits, 2)
        return [qbit1, qbit2], [qbit_coords[qbit2], qbit_coords[qbit1]]    

    # TODO: check if this is working 
    def swap_lines_in_core(
            self,
            U: list,
            V: list,
            ):
        """
        U, V: list of qbits 
        U and V are the parts of the complete bipartite graph
        """
        if random.choice([0, 1]):
            i, j = random.sample(range(len(U)), 2)
            U[i], U[j] = U[j], U[i]
        else:
            i, j = random.sample(range(len(V)), 2)
            V[i], V[j] = V[j], V[i]
        #swap two random elements in part
        core_qbits, core_coords = core.qbits_and_coords_of_core(U, V)
        return core_qbits, core_coords

    # check if code is working, unit test
    def relabel_lbits(self):
        a, b = random.sample(range(0, self.polygon.N), 2)
        n = self.polygon.qbit_coord_dict.keys()
        n = [(-1,  k) if i== a else (i, k) for i, k in n]
        n = [( i, -1) if k== a else (i, k) for i, k in n]
        n = [( a,  k) if i== b else (i, k) for i, k in n]
        n = [( i,  a) if k== b else (i, k) for i, k in n]
        n = [( b,  k) if i==-1 else (i, k) for i, k in n]
        new_qbits = [( i,  b) if k==-1 else (i, k) for i, k in n]
        sorted_new_qbits = list(map(lambda x: tuple(sorted(x)), new_qbits))
        coords = list(self.polygon.qbit_coord_dict.values())
        return sorted_new_qbits, coords
                      
                      
    def step(
            self,
            operation: str,
            U: list=None,
            V: list=None,
            ):
        self.current_qbit_coords = deepcopy(self.polygon.qbit_coord_dict)
        current_energy = self.energy(self.polygon, factors=self.factors)
        if operation == 'contract':
            qbits, coords = self.random_coord_around_core_center()
        if operation == 'swap':
            qbits, coords = self.swap_qbits()
        if operation == 'relable':
            qbits, coords = self.relabel_lbits()
        if operation == 'swap_lines_in_core':
            qbits, coords = self.swap_lines_in_core(U, V)
        if operation == 'grow_core':
            qbits, coords = self.random_coord_next_to_core()
        self.polygon.update_qbits_coords(qbits, coords)
        new_energy = self.energy(self.polygon, factors=self.factors)
        return current_energy, new_energy

    
    def temperature(self):
        num_of_found_plaqs = self.energy.distance_to_plaquette().count(0)
        still_to_find = self.polygon.C - num_of_found_plaqs
        return still_to_find * 0.001 # this factor seems to have an effect

    
    def metropolis(self, current_energy, new_energy, temperature=None):
        if temperature is None:
            temperature = self.temperature()
        delta_energy = new_energy - current_energy
        if random.random() < min([1, np.exp(- delta_energy / temperature)]):
            pass
        else:
            self.polygon.qbit_coord_dict = self.current_qbit_coords
        return delta_energy

            
    def apply(
            self,
            operation,
            n_steps,
            temperature=None,
            U: list=None,
            V: list=None,
            ):
        for i in range(n_steps):
            current_energy, new_energy = self.step(operation, U, V)
            delta_energy = self.metropolis(current_energy, new_energy, temperature)
            #if i % 100 == 0:
            #    self.polygon.move_center_to_middle()
            return delta_energy
                
                      
