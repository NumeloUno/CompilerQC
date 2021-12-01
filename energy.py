import numpy as np
from polygons import Polygons
from shapely.geometry import Polygon, MultiPoint
from scipy.ndimage import convolve

class Energy(Polygons):

    def __init__(self, polygon_object: Polygons):
        self.polygon_coords = polygon_object.get_all_polyg_coords()
        self.envelop_polygon = polygon_object.convex_hull()
        self.K, self.C = polygon_object.K, polygon_object.C
        #self.qbit_coords = list(polygon_object.qbit_coord_dict.keys())


    def scopes_of_polygons(self):
        return list(map(self.polygon_length, self.polygon_coords))


    def area_of_polygons(self):
        return list(map(self.polygon_area, self.polygon_coords))


    def is_plaquette(self):
        return [self.is_unit_square(coord) if len(coord)==4 
                else self.is_unit_triangle(coord) for coord in self.polygon_coords]


    def intersection_array(self):
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
        self.polygon_coords = polygon_object.get_all_polyg_coords()
        self.envelop_polygon = polygon_object.convex_hull()
        #self.qbit_coords = list(polygon_object.qbit_coord_dict.keys())
        list_of_plaquettes = self.is_plaquette()
        polygon_weights = self.polygon_weights(list_of_plaquettes)
        energy = (
        #        area * self.polygon_area(self.envelop_polygon)  
        #      + scope * self.polygon_length(self.envelop_polygon)
        #      + compact * self.number_of_non_plaquette_neighbours()
              + n_plaq * np.dot(np.array(list_of_plaquettes), polygon_weights) 
              )
        return energy
