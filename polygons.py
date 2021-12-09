import numpy as np
import itertools
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon, MultiPoint, MultiLineString
from shapely.ops import polygonize
from shapely import wkt
from graph import Graph

class Polygons():

    def __init__(self,
                 logical_graph: Graph,
                 qbit_coord_dict: dict=None,
                 cycles: list=None
                 ):
        """
        logical_graph: logical graph with N nodes and K edges
        """
        self.qbits = logical_graph.qbits_from_graph()
        self.N = logical_graph.N
        self.K = logical_graph.K 
        self.C = logical_graph.C
        self.qbit_coord_dict = qbit_coord_dict
        if qbit_coord_dict is None:
            self.qbit_coord_dict = self.init_coords_for_qbits()
        if cycles is None:
            cycles = (logical_graph.get_cycles(3)
                    + logical_graph.get_cycles(4))
        self.polygons = self.get_all_polygons(cycles)


    def update_qbits_coords(
            self,
            qbits: list,
            new_coords: list,
            ):
        """
        qbits: list of tuples
        new_coords: list of tuples
        """
        for qbit, new_coord in zip(qbits, new_coords):
            self.qbit_coord_dict[qbit] = new_coord


    @classmethod
    def get_polygon_from_cycle(
            self,
            cycle: list
            ):
        """
        cycle: cycle in graph
        """
        cycle = cycle + [cycle[0]]
        polygon = list(map(tuple, 
            map(sorted, zip(cycle, cycle[1:]))))
        return polygon


    def init_coords_for_qbits(self):
        """
        initialization of qbits with random coordinates 
        return: dictionary with qbits and coordinates
        """
        coords = list(np.ndindex(self.K, self.K))
        np.random.shuffle(coords)
        return dict(zip(self.qbits, coords[:self.K]))


    def get_coords_of_polygon(self, polygon):
        """
        return: list of coords of single polygon
        """
        return list(map( self.qbit_coord_dict.get, polygon))


    @staticmethod
    def polygon_length(polygon_coord):
        return Polygon(polygon_coord).length


    @staticmethod
    def polygon_area(polygon_coord):
        return Polygon(polygon_coord).area


    def get_all_polygons(self, cycles):
        return list(map(self.get_polygon_from_cycle, cycles))


    def get_all_polygon_coords(self):
        """
        return: list of lists, each list contains 
        coordinates of a polygon
        """
        return list(map(self.get_coords_of_polygon, self.polygons))
            
    # TODO: this function can get very important, think more about it
    def polygon_weights(self, closeness_to_plaquette):
        ctp = np.array(closeness_to_plaquette)
        ones = np.ones_like(ctp)
        weights = np.divide(ones, ctp, out=(ones * 999), where=(ctp != 0))
        normalized_weights = weights / (weights.sum() / self.C)
        return normalized_weights


    @staticmethod
    def is_unit_square(polygon_coord):
        """
        measure closeness to plaquette
        """
        return Polygons.polygon_length(polygon_coord) - 4


    @staticmethod
    def is_unit_triangle(polygon_coord):
        """
        measure closeness to plaquette
        """
        return Polygons.polygon_length(polygon_coord) - (2 + np.sqrt(2))


    def convex_hull(self):
        """
        return envelop polygon of all qbits
        """
        qbit_coords = list(self.qbit_coord_dict.values())
        qbits = MultiPoint(qbit_coords)
        envelop_polygon = qbits.convex_hull
        return list(envelop_polygon.exterior.coords)


    def center_of_convex_hull(self):
        center = Polygon(self.convex_hull()).centroid
        return (center.x, center.y) 


    def move_center_to_middle(self):
        """
        calculates the CoM of the convex hull
        and moves the hull to the center of the grid
        """
        center_x, center_y = map(int, self.center_of_convex_hull())
        middle_x = middle_y = self.K // 2
        move_x, move_y = center_x - middle_x, center_y - middle_y
        qbits = list(self.qbit_coord_dict.keys())
        new_coords = [(coord[0] - move_x, coord[1] - move_y)
                for coord in self.qbit_coord_dict.values()]
        self.update_qbits_coords(qbits, new_coords)


    def qbit_distances_to_center(self):
        center = self.center_of_convex_hull()
        qbit_coords = self.qbit_coord_dict.values()
        distances = list(map(lambda x: Point(center).distance(Point(x)), qbit_coords))
        return distances


    @staticmethod
    def get_grid_of_unit_squares(grid_length):
        """
        from 37041377 stackoverflow
        returns coordinates of unit square polygons
        """
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
                    lw=0,
                    facecolor = facecolor)
            ax.add_patch(patch)
        for qbit, coord in self.qbit_coord_dict.items():
            ax.annotate(str(qbit), coord)
            ax.scatter(*coord, color='red')
        x_range = sorted(self.convex_hull(), key=lambda x: x[0])
        y_range = sorted(self.convex_hull(), key=lambda y: y[1])
        ax.set_xlim(min(x_range[0][0], y_range[0][1])-1,
                  max(x_range[-1][0], y_range[-1][1])+1)
        ax.set_ylim(min(x_range[0][0], y_range[0][1])-1,
                  max(x_range[-1][0], y_range[-1][1])+1)
        return ax
