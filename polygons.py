import numpy as np
import random
from itertools import combinations
from math import dist
from matplotlib import pyplot as plt

class Polygons:
    """
    this class converts the edges of a graph to qbits
    and puts them on a grid. the qbits are connected to 
    other qbits if they formed a closed cyle (3 or 4 length) 
    in the graph. So the qbits are connected to polygons of
    length three or four. Some qbits are part of the core 
    (e.g. definded by the core bipartite sets).
    """
    
    unit_triangle_scope = round((np.sqrt(2) + 2) / 3, 5)
    unit_square_scope = round((2 * np.sqrt(2) + 4) / 4, 5)

    def __init__(
        self,
        qbits: 'Qbits'
    ):
        """
        logical_graph: logical graph with N nodes and K edges
        """
        self.qbits = qbits
        self.graph = self.qbits.graph
        self.polygons = Polygons.create_polygons(self.graph.cycles)
        self.qbits.set_all_polygons(self.polygons)


    @staticmethod
    def create_polygon_from_cycle(cycle: list):
        """
        cycle: cycle in graph
        returns polygon
        """
        cycle = cycle + [cycle[0]]
        polygon = list(map(tuple, map(sorted, zip(cycle, cycle[1:]))))
        return polygon

    @staticmethod
    def create_polygons(cycles):
        return list(map(Polygons.create_polygon_from_cycle, cycles))
    
    @staticmethod
    def coords_of_polygon(qubit_to_coord_dict: dict, polygon):
        """
        return: list of coords of single polygon
        """
        return list(map(qubit_to_coord_dict.get, polygon))

    @staticmethod    
    def polygons_coords(qubit_to_coord_dict: dict, polygons):
        """
        return: array of lists, each list contains
        coordinates of a polygon,
        """ 
        return [Polygons.coords_of_polygon(qubit_to_coord_dict, polygon) for polygon in polygons]
    

    @staticmethod
    def scope_of_polygon(polygon_coord):
        """
        return scope of the polygon
        """
        return round(sum([dist(*c) for c in list(combinations(polygon_coord, 2))]) / len(polygon_coord), 5)


    def envelop_rect(self, padding: int=1):
        """
        generate grid coords of envelop rectengular (of all qbits) with padding
        """
        x, y = np.array(self.qbits.coords).T
        x, y = np.meshgrid(np.arange(min(x) - padding, max(x) + (padding + 1)),
                           np.arange(min(y) - padding, max(y) + (padding + 1)))
        return list(zip(x.flatten(), y.flatten()))      

    def found_plaqs(self):
        """
        listing all plaquettes in current compilation,
        without looping over all polygons
        """
        coords = self.envelop_rect(padding=0)
        coord_to_qbit_dict = self.qbits.coord_to_qbit_dict
        # get upper right square of coords for each coord
        square = lambda coord: [coord_to_qbit_dict.get((coord[0]+i, coord[1]+j), np.nan) for i in range(2) for j in range(2)]
        # remove nan in each square
        qbits_in_each_square = [list(filter(lambda v: v==v, square)) for square in list(map(square, coords))]
        # consider only squares with more than 2 qbits
        relevant_squares = [[qbit.qubit for qbit in square ]for square in qbits_in_each_square  if len(square) > 2]
        # list of 3 combinations and 4 combis, 4 combis are appended by all possible 3 combis
        possible_plaqs = [[i] if len(i) == 3 else list(combinations(i, 3))+[i] for i in relevant_squares]
        # count number of nodes in each 3er or 4er combination, divide it by length of combination
        a = [[len(set(sum(x,()))) / len(x) for x in i] for i in possible_plaqs]
        # if there is a one in a 3er combi, there is a plaquette
        # if there is a one in a 4er combi and its 3 combinations (for 3plaqs), its a 4 plaq
        # if there are two ones in a a 4er combi and its 3 combinations (for 3plaqs), the first one is a 3er plaq
        return ([possible_plaqs[i][j] for i,j  in [[i,l.index(1)] for i,l in enumerate(a) if 1 in l]])
    
    @property
    def number_of_plaqs(self):
        return len(self.found_plaqs())
    
    def visualize(self, ax=None, zoom=1):
        if ax is None:
            _, ax = plt.subplots()
        x, y = list(zip(*self.envelop_rect()))
        ax.scatter(x, y, color="grey", s=0.6)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for polygon in self.polygons_coords(self.qbits.qubit_to_coord_dict, self.polygons):
            fill, facecolor = False, None
            if self.scope_of_polygon(polygon) == self.unit_square_scope:
                fill, facecolor = True, "lightblue"
            if self.scope_of_polygon(polygon) == self.unit_triangle_scope:
                fill, facecolor = True, "indianred"
            patch = plt.Polygon(polygon, zorder=0, fill=fill, lw=0, facecolor=facecolor)
            ax.add_patch(patch)
        for qbit, coord in self.qbits.qubit_to_coord_dict.items():
            ax.annotate(str(qbit), coord)
            ax.scatter(*coord, color="red")
