import numpy as np
from polygons import Polygons
from shapely.geometry import Polygon, MultiPoint

class Energy(Polygons):

    def __init__(self, polygon_object: Polygons):
        self.polygon_coords = polygon_object.get_all_polyg_coords()
        self.envelop_polygon = polygon_object.convex_hull()
        self.K = polygon_object.K
    def scopes_of_polygons(self):
        return list(map(self.polygon_length, self.polygon_coords))
    def area_of_polygons(self):
        return list(map(self.polygon_area, self.polygon_coords))
    def is_plaquette(self):
        unit_square = list(map(self.is_unit_square, self.polygon_coords))
        unit_triangle = list(map(self.is_unit_triangle, self.polygon_coords))
        return list(map(lambda x: True in x, zip(unit_square, unit_triangle)))
    def intersection_array(self):
        unit_square_grid = self.get_grid_of_unit_squares(self.K)
        intersects = list(map(lambda x: Polygon(self.envelop_polygon).intersects(Polygon(x)),
            unit_square_grid))
        touches = list(map(lambda x: Polygon(self.envelop_polygon).touches(Polygon(x)),
            unit_square_grid))
        return np.logical_xor(intersects, touches)

    # TODO: energie terme aufstellen

    def __call__(self):
        energy = self.polygon_area(self.envelop_polygon)
        return energy
