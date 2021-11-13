import numpy as np
from matplotlib import pyplot as plt
from dfs import Graph
from shapely.geometry import Polygon

class Polygons():
    def __init__(self, logical_graph: Graph):
        self.qbits = logical_graph.qubits_from_graph()
        self.K = len(self.qbits)
        self.qbit_coord_dict = self.init_coords_for_qbits()
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
    def distance(xy):
        squared_diff = np.square(np.subtract(*xy))
        return np.sqrt(squared_diff.sum())
    @classmethod
    def polygon_scope(self, polygon_coord):
        polygon = self.get_polygon_from_cycle(polygon_coord)
        return sum(list(map(self.distance, polygon)))
    @staticmethod
    def polygon_area(polygon_coord):
        return Polygon(polygon_coord).area
    def get_all_polygons(self, cycles):
        return list(map(self.get_polygon_from_cycle, cycles))
    def get_all_polyg_coords(self, polygons):
        return list(map(self.get_coords_of_polygon, polygons))
    def visualize(self, ax, polygon_coords):
        x, y = np.meshgrid(np.arange(self.K), np.arange(self.K))
        ax.scatter(x.flatten(), y.flatten(), color='grey', s=0.6)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        for polygon in polygon_coords:
            patch = plt.Polygon(polygon, zorder=0, fill=False, lw=1)
            ax.add_patch(patch)

        for qbit, coord in self.qbit_coord_dict.items():
            ax.annotate(str(qbit), coord)
            ax.scatter(*coord, color='black')

        return ax
