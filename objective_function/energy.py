import numpy as np
from CompilerQC import Polygons
from shapely.geometry import Polygon, MultiPoint
from scipy.ndimage import convolve
from itertools import combinations, permutations
from CompilerQC import Graph
import os.path
homedir = os.path.expanduser("~/UniInnsbruck/")
path_to_scaling_factors = os.path.join(homedir,
        "CompilerQC/parameter_estimation/parameters")

class Energy(Polygons):


    def __init__(
            self,
            polygon_object: Polygons,
            ):
        self.N, self.K, self.C = (
                polygon_object.N,
                polygon_object.K,
                polygon_object.C,
                )
        #self.qbit_coords = list(polygon_object.qbit_coord_dict.keys())
        self.scaling_for_plaq3 = self.scaling_factors_LHZ(3)
        self.scaling_for_plaq4 = self.scaling_factors_LHZ(4)

    @staticmethod
    def scaling_factors_LHZ(number: int):
        """
        load the array of scaling factors for 
        3plaqs or 4plaqs.
        these scaling factors compensate for the 
        higher number of non plaqs,
        number: 3 or 4
        """
        return np.load(os.path.join(path_to_scaling_factors,
           f"scaling_factors_LHZ{number}.npy"))


    def scopes_of_polygons(self):
        return list(map(self.polygon_length, self.polygon_coords))


    def area_of_polygons(self):
        return list(map(self.polygon_area, self.polygon_coords))


    def distance_to_plaquette(self):
        """
        return: list, each entry represents the closeness of a polygon 
        to plaquette (square or triangle), which is zero if the plaquette is found
        """
        return [self.is_unit_square(coord) if len(coord)==4 
                else self.is_unit_triangle(coord) for coord in self.polygon_coords]

    def scaled_distance_to_plaquette(self):
        """
        return: list, each entry represents the closeness of a polygon 
        to plaquette (square or triangle), which is minus the the sacling factor
        LHZ (Wolfgangs Idee)
        """
        distances = []
        for coord in self.polygon_coords:
            if len(coord) == 4:
                distance = self.is_unit_square(coord)
                if distance == 0:
                    distance -= self.scaling_for_plaq4[self.N-4]
            if len(coord) == 3:
                distance = self.is_unit_triangle(coord)
                if distance == 0:
                    distance -= self.scaling_for_plaq3[self.N-4]
            distances.append(distance)
        return distances


    def exp_factor(self, constant: float):
        """
        if all C plaqs has been found, energy is scaled down to 0
        """
        number_of_found_plaqs = self.distance_to_plaquette().count(0)
        return (1 - np.exp(constant * (number_of_found_plaqs - self.C)))
    

    def intersection_array(self):
        """
        return: unit square polygon which intersects with qbit layout
        """
        unit_square_grid = self.get_grid_of_unit_squares(self.K)
        intersects = list(map(lambda x: Polygon(self.envelop_polygon).intersects(Polygon(x)),
            unit_square_grid))
        touches = list(map(lambda x: Polygon(self.envelop_polygon).touches(Polygon(x)),
            unit_square_grid))
        intersected_unit_square = np.logical_xor(intersects, touches)
        return np.reshape(intersected_unit_square, (self.K - 1, self.K - 1))


    def number_of_non_plaquette_neighbours(self):
        """ from 69962789 stackoverflow """
        matrix = self.intersection_array().astype(int)
        matrix = np.pad(matrix, 1, mode='constant')
        kernel = np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]])
        return np.sum(np.where(convolve(matrix, kernel) < 0, 1, 0))


    # TODO: complete tis function, actually this term is automatically fullfiled by the algo
    def distance_btw_pqbits(self):
        pass

        
    def __call__(
            self,
            polygon_object,
            constant_for_exp: float=1.,
            factors: list=[1., 1., 1., 1.]):
        area, scope, compact, n_plaq = factors
#        self.polygon_coords = polygon_object.get_all_polygon_coords()
        self.polygon_coords = polygon_object.polygons_outside_core()
        list_of_plaquettes = self.distance_to_plaquette()
        polygon_weights = self.polygon_weights(list_of_plaquettes)
#         list_of_plaquettes =self.scaled_distance_to_plaquette()
        energy = (
              + n_plaq * np.dot(np.array(list_of_plaquettes), polygon_weights) 
#                 self.exp_factor(constant_for_exp) * sum(list_of_plaquettes)
              )
        return energy


class Energy_landscape():

    def __init__(
            self,
            logical_graph: Graph,
            polygon_object: Polygons,
            ):
        """
        This class calulcates the energie landscape for different combinations (where are the qbits on the grid)
        and for different permutations (certain positions on the grid, but different qbits on it)
        grid: dict of coords as keys, ans numbers as values
        """
        self.r = logical_graph.K
        self.grid = self.init_grid(logical_graph.N) 
        self.fit = Energy(polygon_object)
        self.qbits = polygon_object.qbits
        self.polygon_object = polygon_object


    def init_grid(
            self,
            N: int
            ):
        """
        N: number of logical qbits
        return: dict with key:value <-> tuple of grid point coordinate: grid point number
        """
        x, y = np.meshgrid(np.arange(N-1), np.arange(1, N))
        grid_points = list(zip(x.flatten(), y.flatten()))
        grid_point_numbers = np.arange(len(grid_points))
        return dict(zip(grid_points, grid_point_numbers))


    def get_all_combinations(
            self, 
            n: list=None
            ):
        """
        n: grid point numbers  we consider, default is the whole square of (N-1)² points
        r: number of qbits
        return: list of tuples of all combinations, without order
        """
        if n is None:
            n = list(self.grid.values())
        return list(combinations(n, self.r))


    def get_all_permutations(
            self,
            to_permute: list,
            ):
        """
        to_permute: list of list(s) to permute
        return: list of tuples of all permutations
        """
        return [list(permutations(i)) for i in to_permute]


    def grid_num_to_coords(
            self,
            list_of_grid_nums: list,
            ):
        """
        list_of_grid_nums: list of grid numbers for each qbit
        return: list of coords for each qbit
        """
        return [list(self.grid.keys()) [i] for i in list_of_grid_nums]


    def energy_landscape(
            self,
            list_of_coords: list,
            constant_for_exp: float=1.,
            ):
        """
        list_of_coords: list of list(s) of coords
        return: array of energy values
        """
        energy_list = []
        for coords in list_of_coords:
            self.polygon_object.update_qbits_coords(self.qbits, coords)
            energy_list.append(self.fit(self.polygon_object, constant_for_exp=constant_for_exp))
        return np.array(energy_list)
