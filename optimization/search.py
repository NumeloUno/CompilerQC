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
        self.movable_qbits = list(self.polygon.qbit_coord_dict.keys())

    def update_movable_qbits(
            self,
            fixed_qbits: list,
            ):
        """
        fixed_coords: qbit coords which belong to e.g the core obtained 
        by the bipartite graph ansatz 
        """
        all_qbits = set(self.polygon.qbit_coord_dict.keys())
        self.movable_qbits = list(all_qbits - set(fixed_qbits))


    def generate_new_coord_for_qbit(
            self,
            qbit: tuple=None,
            radius: float=None,
            ):
        if qbit is None:
            qbit = random.choice(self.movable_qbits)
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
        return [qbit], [new_random_coord]
        
        
    def swap_qbits(self):
        qbit_coords = self.polygon.qbit_coord_dict
        qbit1, qbit2 = random.sample(self.movable_qbits, 2)
        return [qbit1, qbit2], [qbit_coords[qbit1], qbit_coords[qbit2]]    

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
            qbits, coords = self.generate_new_coord_for_qbit()
        if operation == 'swap':
            qbits, coords = self.swap_qbits()
        if operation == 'relable':
            qbits, coords = self.relabel_lbits()
        if operation == 'swap_lines_in_core':
            qbits, coords = self.swap_lines_in_core(U, V)
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
            if i % 100 == 0:
                self.polygon.move_center_to_middle()
            return delta_energy
                
                      

class Encode():

    def __call__(self, polygon_object: Polygons):
        self.polygon_object = polygon_object
        self.qbit_coord_dict = polygon_object.qbit_coord_dict
        self.qbits_to_numbers = self.qbits_to_numbers_()
        return self.from_grid_to_intrinsic()


    def qbits_to_numbers_(self):
        qbits = list(self.qbit_coord_dict.keys())
        numbers = np.arange(1, len(qbits)+1)
        return dict(zip(qbits, numbers))


    def neighbours(self, x, y):
        return ((x, y+1), (x+1, y+1), (x+1, y), (x+1, y-1), (x, y-1), (x-1, y-1), (x-1, y), (x-1, y+1))


    def get_neighbours(self, qbit):
        x, y = self.qbit_coord_dict[qbit] #center
        revers = {v: k for k, v in self.qbit_coord_dict.items()}
        return [self.qbits_to_numbers[revers[coord]] if coord in revers.keys() else 0
                for coord in self.neighbours(x, y)]


    def from_grid_to_intrinsic(self):
        """ from grid representation to intrinsic neighbour representation"""
        return np.array(list(map(self.get_neighbours, self.qbit_coord_dict.keys())))


    def from_intrinsic_to_grid(self, intrinsic_representation):
        clusters = self.get_intrinsic_cluster(intrinsic_representation)
        new_coords, radius = {}, 0
        for j, cluster in enumerate(sorted(clusters, key=len)):
            coords = dict.fromkeys(cluster)
            radius += len(cluster) 
            coords[cluster[0]], i = (radius, radius), 0
            while list(coords.values()).count(None) > 0:
                qbit = int(cluster[i%len(cluster)])
                i += 1
                if coords[qbit] is None:
                    continue
                coords_of_neigh_qbits = [self.neighbours(*coords[qbit])[i] 
                                         for i in np.where(intrinsic_representation[qbit-1])[0]]
                nonzero_neigh_qbits = list(filter(lambda x: x!=0, intrinsic_representation[qbit-1]))
                for qbit_, coord in zip(nonzero_neigh_qbits, coords_of_neigh_qbits):
                    coords[qbit_] = coord
                if i > 100:
                    return -99999
            radius += len(cluster) 
            new_coords.update(coords)
        return {k: new_coords[v] for k, v in self.qbits_to_numbers.items()}


    def get_intrinsic_cluster(self, intrinsic_representation):
        intrinsic_neig_qbits = [list(filter(lambda x: x!=0, l)) for l in intrinsic_representation]
        clusters_of_qbits = [l + [i + 1 ] for i, l in enumerate(intrinsic_neig_qbits)]
        for qbit in self.qbits_to_numbers.values():
            components = [cluster for cluster in clusters_of_qbits if qbit in cluster]
            for i in components:
                clusters_of_qbits.remove(i)
            clusters_of_qbits += [list(set(itertools.chain.from_iterable(components)))]
        return clusters_of_qbits
    

    # TODO: if we have cluster, polygons between cluster enlarge energy, also this deepcopy solution is ugly
    def fitness(self, flattend_intrinsic_representation, factors=[1,1,1,1]):
        intrinsic_representation = np.reshape(flattend_intrinsic_representation,
                (len(flattend_intrinsic_representation) // 8, 8))
        new_qbit_coord_dict = self.from_intrinsic_to_grid(intrinsic_representation)
        polygon_object = deepcopy(self.polygon_object)
        polygon_object.qbit_coord_dict = new_qbit_coord_dict
        return Energy(polygon_object)(polygon_object, factors=factors)
    

    def fitness_per_qbit(self, flattend_intrinsic_representation):
        intrinsic_representation = np.reshape(flattend_intrinsic_representation,
                (len(flattend_intrinsic_representation) // 8, 8))
        new_qbit_coord_dict = self.from_intrinsic_to_grid(intrinsic_representation)
        #check if valid grid layout was found
        if type(new_qbit_coord_dict) == int:
            return [-99999]
        #check if coords are unique:
        if len(new_qbit_coord_dict) > len(set(new_qbit_coord_dict.values())):
            return [-99999]
        polygon_object = deepcopy(self.polygon_object)
        polygon_object.qbit_coord_dict = new_qbit_coord_dict
        coords_qbit_dict = {v: k for k, v in 
                polygon_object.qbit_coord_dict.items()}
        qbits_in_polygon = [list(map(lambda x: coords_qbit_dict[x],
            polygon)) for polygon in polygon_object.get_all_polyg_coords()]
        qbits_in_plaqs = np.array(qbits_in_polygon,
                dtype=object)[
                np.array(Energy(polygon_object).is_plaquette()) == 0]
        if not qbits_in_plaqs.size:
            return [-99999]
        qbits_in_plaqs = list(map(tuple, np.concatenate(qbits_in_plaqs)))
        return [qbits_in_plaqs.count(qbit) for qbit in polygon_object.qbits]


    # TODO: combine good genes
    def cross_over(self):
        pass

    def binary_list_from_indices(self, indices):
        indices_set = set(indices)
        return np.array([i in indices_set for i in list(self.qbits_to_numbers.values())]).astype(int)


    def indices_from_binary_list(self, binary_list):
        return (np.where(binary_list)[0] + 1).tolist()
        
        
    def old_encode(self, binary_list):
        binomial_tuples = zip(np.arange(len(binary_list)),
                np.cumsum(binary_list))
        binomial_coeffs = np.array(list(map(
            lambda x: binom(x[0], x[1]), binomial_tuples)))
        return np.dot(binary_list, binomial_coeffs)

    
    def decode(self):
        pass


               
