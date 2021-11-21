import numpy as np
import itertools
from matplotlib import pyplot as plt
from graph import Graph
from shapely.geometry import Point, Polygon, MultiPoint, MultiLineString
from shapely.ops import polygonize
from shapely import wkt

class Polygons():
    def __init__(self,
                 logical_graph: Graph,
                 qbit_coord_dict: dict=None,
                 cycles: list=None):

        self.qbits = logical_graph.qubits_from_graph()
        self.K = len(self.qbits)
        self.C = logical_graph.num_constrains()
        self.qbit_coord_dict = qbit_coord_dict
        if qbit_coord_dict is None:
            self.qbit_coord_dict = self.init_coords_for_qbits()
        cycles = (logical_graph.get_cycles(3)
                + logical_graph.get_cycles(4))
        self.polygons = self.get_all_polygons(cycles)

    def update_qbits_coords(self, qbits, new_coords):
        if type(qbits) == tuple and type(new_coords) == tuple:
            qbits = [qbits]
            new_coords = [new_coords]
        for qbit, new_coord in zip(qbits, new_coords):
            self.qbit_coord_dict[qbit] = new_coord

    @classmethod
    def get_polygon_from_cycle(self, cycle):
        cycle = cycle + [cycle[0]]
        polygon = list(map(tuple, 
            map(sorted, zip(cycle, cycle[1:]))))
        return polygon

    def init_coords_for_qbits(self):
        coords = list(np.ndindex(self.K, self.K))
        np.random.shuffle(coords)
        qbit_coord_dict = dict(zip(self.qbits, coords[:self.K]))
        return qbit_coord_dict

    def get_coords_of_polygon(self, polygon):
        return list(map( self.qbit_coord_dict.get, polygon))

    @staticmethod
    def polygon_length(polygon_coord):
        return Polygon(polygon_coord).length

    @staticmethod
    def polygon_area(polygon_coord):
        return Polygon(polygon_coord).area

    def get_all_polygons(self, cycles):
        return list(map(self.get_polygon_from_cycle, cycles))

    def get_all_polyg_coords(self):
        return list(map(self.get_coords_of_polygon, self.polygons))
            
    def polygon_weights(self, closeness_to_plaquette):
        ctp = np.array(closeness_to_plaquette)
        ones = np.ones_like(ctp)
        weights = np.divide(ones, ctp, out=(ones * 999), where=(ctp != 0))
        normalized_weights = weights / (weights.sum() / self.C)
        return normalized_weights
    @staticmethod
    def is_unit_square(polygon_coord):
        return Polygons.polygon_length(polygon_coord) - 4

    @staticmethod
    def is_unit_triangle(polygon_coord):
        return Polygons.polygon_length(polygon_coord) - (2 + np.sqrt(2))

    def convex_hull(self):
        qbit_coords = list(self.qbit_coord_dict.values())
        pqbits = MultiPoint(qbit_coords)
        envelop_polygon = pqbits.convex_hull
        return list(envelop_polygon.exterior.coords)

    def center_of_convex_hull(self):
        center = Polygon(self.convex_hull()).centroid
        return (center.x, center.y) 

    def qbit_distances_to_center(self):
        center = self.center_of_convex_hull()
        qbit_coords = self.qbit_coord_dict.values()
        distances = list(map(lambda x: Point(center).distance(Point(x)), qbit_coords))
        return distances

    @staticmethod
    def get_grid_of_unit_squares(grid_length):
        """from 37041377 stackoverflow"""
        x = y = np.arange(grid_length)
        hlines = [((x1, yi), (x2, yi)) for x1, x2 in zip(x[:-1], x[1:]) for yi in y]
        vlines = [((xi, y1), (xi, y2)) for y1, y2 in zip(y[:-1], y[1:]) for xi in x]
        grid = list(polygonize(MultiLineString(hlines + vlines)))
        unit_square_coords = [list(square.exterior.coords) for square in grid]
        return unit_square_coords

    def visualize(self, ax, polygon_coords):
        x, y = np.meshgrid(np.arange(self.K), np.arange(self.K))
        ax.scatter(x.flatten(), y.flatten(), color='grey', s=0.6)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        for polygon in polygon_coords:
            fill, facecolor = False, None
            if self.is_unit_square(polygon) == 0:
                fill, facecolor = True, 'blue'
            if self.is_unit_triangle(polygon) == 0:
                fill, facecolor = True, 'red'
            patch = plt.Polygon(polygon,
                    zorder=0,
                    fill=fill,
                    lw=1,
                    facecolor = facecolor)
            ax.add_patch(patch)

        for qbit, coord in self.qbit_coord_dict.items():
            ax.annotate(str(qbit), coord)
            ax.scatter(*coord, color='red')

        return ax
