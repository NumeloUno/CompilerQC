import numpy as np
from CompilerQC import Polygons
from shapely.geometry import Polygon, MultiPoint
from scipy.ndimage import convolve
from itertools import combinations, permutations
from CompilerQC import Graph
import os.path

homedir = os.path.expanduser("~/UniInnsbruck/")
path_to_scaling_factors = os.path.join(
    homedir, "CompilerQC/parameter_estimation/parameters"
)


class Energy(Polygons):
    def __init__(
        self,
        polygon_object: Polygons,
    ):
        self.N = polygon_object.N
        self.K = polygon_object.K
        self.C = polygon_object.C
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
        return np.load(
            os.path.join(path_to_scaling_factors, f"scaling_factors_LHZ{number}.npy")
        )

    def distance_to_plaquette(self):
        """
        return: list, each entry represents the closeness of a polygon
        to plaquette (square or triangle), which is zero if the plaquette is found
        """
        return [
            self.is_unit_square(coord)
            if len(coord) == 4
            else self.is_unit_triangle(coord)
            for coord in self.polygon_coords
        ]

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
                    distance -= self.scaling_for_plaq4[self.N - 4]
            if len(coord) == 3:
                distance = self.is_unit_triangle(coord)
                if distance == 0:
                    distance -= self.scaling_for_plaq3[self.N - 4]
            distances.append(distance)
        return distances

    def exp_factor(self, constant: float):
        """
        if all C plaqs has been found, energy is scaled down to 0
        """
        number_of_found_plaqs = self.distance_to_plaquette().count(0)
        return 1 - np.exp(constant * (number_of_found_plaqs - self.C))

    # TODO: complete tis function
    def same_lbits_on_line(self):
        """
        reward bipartite lines are found
        """

    def __call__(self, polygon_object, terms: list = [1.0, 0.0, 1.0]):
        (
            ignore_inside_polygons,
            scaled_wolfgang,
            polygon_weigth,
        ) = terms
        if ignore_inside_polygons:
            self.polygon_coords = polygon_object.polygons_outside_core()
        else:
            self.polygon_coords = polygon_object.get_all_polygon_coords()
        if scaled_wolfgang:
            list_of_plaquettes = self.scaled_distance_to_plaquette()
            return sum(list_of_plaquettes)
        if polygon_weigth:
            list_of_plaquettes = self.distance_to_plaquette()
            polygon_weights = self.polygon_weights(list_of_plaquettes)
            return np.dot(np.array(list_of_plaquettes), polygon_weights)


class Energy_landscape:
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

    def init_grid(self, N: int):
        """
        N: number of logical qbits
        return: dict with key:value <-> tuple of grid point coordinate: grid point number
        """
        x, y = np.meshgrid(np.arange(N - 1), np.arange(1, N))
        grid_points = list(zip(x.flatten(), y.flatten()))
        grid_point_numbers = np.arange(len(grid_points))
        return dict(zip(grid_points, grid_point_numbers))

    def get_all_combinations(self, n: list = None):
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
        return [list(self.grid.keys())[i] for i in list_of_grid_nums]

    def energy_landscape(
        self,
        list_of_coords: list,
        constant_for_exp: float = 1.0,
    ):
        """
        list_of_coords: list of list(s) of coords
        return: array of energy values
        """
        energy_list = []
        for coords in list_of_coords:
            self.polygon_object.update_qbits_coords(self.qbits, coords)
            energy_list.append(
                self.fit(self.polygon_object, constant_for_exp=constant_for_exp)
            )
        return np.array(energy_list)
