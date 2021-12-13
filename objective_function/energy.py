import numpy as np
from CompilerQC import Polygons
from shapely.geometry import Polygon, MultiPoint
from scipy.ndimage import convolve
from itertools import combinations, permutations
from CompilerQC import Graph

class Energy(Polygons):


    def __init__(
            self,
            polygon_object: Polygons,
            ):
        self.polygon_coords = polygon_object.get_all_polygon_coords()
        self.envelop_polygon = polygon_object.convex_hull()
        self.N = polygon_object.N
        self.K, self.C = polygon_object.K, polygon_object.C
        #self.qbit_coords = list(polygon_object.qbit_coord_dict.keys())
    
    scaling_for_plaq3 = ([  1.82184071,   6.06593094,  13.63807311,  25.44413237,
        42.38993543,  65.381298  ,  95.32403166, 133.12394619,
       179.68685042, 235.9185527 , 302.72486109, 381.01158345,
       471.68452755, 575.64950108, 693.81231165, 827.07876687])
    
    scaling_for_plaq4 = ([   5.30056124,   14.89326143,   31.41826795,   56.68725207,
         92.51185921,  140.70372216,  203.07446827,  281.43572232,
        377.59910779,  493.37624741,  630.57876348,  791.01827803,
        976.5064129 , 1188.85478983, 1429.87503047, 1701.37875641])




    def scopes_of_polygons(self):
        return list(map(self.polygon_length, self.polygon_coords))


    def area_of_polygons(self):
        return list(map(self.polygon_area, self.polygon_coords))


    def distance_to_plaquette(self):
        """
        return: list, each entry represents the closeness of a polygon 
        to plaquette (square or triangle)
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

        
    def __call__(self, polygon_object, factors: list=[1., 1., 1., 1.]):
        area, scope, compact, n_plaq = factors
        self.polygon_coords = polygon_object.get_all_polygon_coords()
        self.envelop_polygon = polygon_object.convex_hull()
        #self.qbit_coords = list(polygon_object.qbit_coord_dict.keys())
        list_of_plaquettes = self.distance_to_plaquette()
        #polygon_weights = self.polygon_weights(list_of_plaquettes)
        energy = (
        #        area * self.polygon_area(self.envelop_polygon)  
        #      + scope * self.polygon_length(self.envelop_polygon)
        #      + compact * self.number_of_non_plaquette_neighbours()
        #      + n_plaq * np.dot(np.array(list_of_plaquettes), polygon_weights) 
                sum(list_of_plaquettes)
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
        return: dict with key:value <-> tuple of grid point coordinate: grid point number
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
            list_of_coords: list
            ):
        """
        list_of_coords: list of list(s) of coords
        return: array of energy values
        """
        energy_list = []
        for coords in list_of_coords:
            self.polygon_object.update_qbits_coords(self.qbits, coords)
            energy_list.append(self.fit(self.polygon_object))
        return np.array(energy_list)
