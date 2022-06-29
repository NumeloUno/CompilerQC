import numpy as np
import random
from itertools import combinations
from math import dist, atan2
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

    def __init__(self, qbits: "Qbits", polygons: "list of polygons" = None, line_scaling:float=0):
        """
        qbits: qbits object, qbits represent connections in logical grpah
        polygons:
        line_scaling: if the distance between two qbits is 1.0 or sqrt(2), hence the qbits are
        adjacent, set their distance to line_scaling
        """
        self.qbits = qbits
        self.graph = self.qbits.graph
        if polygons is None:
            polygons = Polygons.create_polygons(self.graph.cycles)
        self.polygons = polygons
        self.qbits.set_all_polygons(self.polygons)
        
        self.line_scaling = line_scaling
        self.core_corner = None

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
        return [
            Polygons.coords_of_polygon(qubit_to_coord_dict, polygon)
            for polygon in polygons
        ]

    @staticmethod
    def scope_of_polygon(polygon_coord):
        """
        return scope of the polygon
        """
        return round(
            sum([dist(*c) for c in list(combinations(polygon_coord, 2))])
            / len(polygon_coord),
            5,
        )

    def envelop_rect(self, padding: int = 1):
        """
        generate grid coords of envelop rectengular (of all qbits) with padding
        """
        x, y = np.array(self.qbits.coords).T
        x, y = np.meshgrid(
            np.arange(min(x) - padding, max(x) + (padding + 1)),
            np.arange(min(y) - padding, max(y) + (padding + 1)),
        )
        return list(zip(x.flatten(), y.flatten()))
 
    @staticmethod
    def corner_of_coords(coords):
        """
        returns the center of the envelop
        of coords
        returns: ((min_x, max_x), (min_y, max_y))
        """
        x_coords, y_coords = list(zip(*coords))
        min_x, max_x = (min(x_coords), max(x_coords))
        min_y, max_y = (min(y_coords), max(y_coords))
        corner = ((min_x, max_x), (min_y, max_y))
        return corner
    
    @staticmethod
    def center_of_coords(coords):
        """
        returns the center of the envelop
        of coords
        """
        ((min_x, max_x), (min_y, max_y)) = Polygons.corner_of_coords(coords)
        c_x = (min_x + max_x) / 2
        c_y = (min_y + max_y) / 2
        return c_x, c_y
    
    def found_plaqs(self):
        """
        listing all plaquettes in current compilation,
        without looping over all polygons
        """
        coords = self.envelop_rect(padding=0)
        coord_to_qbit_dict = self.qbits.coord_to_qbit_dict
        # get upper right square of coords for each coord
        square = lambda coord: [
            coord_to_qbit_dict.get((coord[0] + i, coord[1] + j), np.nan)
            for i in range(2)
            for j in range(2)
        ]
        # remove nan in each square
        qbits_in_each_square = [
            list(filter(lambda v: v == v, square))
            for square in list(map(square, coords))
        ]
        # consider only squares with more than 2 qbits
        relevant_squares = [
            [qbit.qubit for qbit in square]
            for square in qbits_in_each_square
            if len(square) > 2
        ]
        # list of 3 combinations and 4 combis, 4 combis are appended by all possible 3 combis
        possible_plaqs = [
            [i] if len(i) == 3 else list(combinations(i, 3)) + [i]
            for i in relevant_squares
        ]
        # count number of nodes in each 3er or 4er combination, divide it by length of combination
        a = [[len(set(sum(x, ()))) / len(x) for x in i] for i in possible_plaqs]
        # if there is a one in a 3er combi, there is a plaquette
        # if there is a one in a 4er combi and its 3 combinations (for 3plaqs), its a 4 plaq
        # if there are two ones in a a 4er combi and its 3 combinations (for 3plaqs), the first one is a 3er plaq
        return [
            list(possible_plaqs[i][j])
            for i, j in [[i, l.index(1)] for i, l in enumerate(a) if 1 in l]
        ]

    def set_plaquettes_of_qbits(self):
        """
        set the plaquettes of each qbit
        """
        found_plaquettes = self.found_plaqs()
        for qbit in self.qbits:
            qbit.plaquettes = [plaq for plaq in found_plaquettes if qbit.qubit in plaq]
        
    def set_numbers_of_qbits_neighbours(self):
        """"set number of neighbours each qbit has"""
        for qbit in self.qbits:
            self.qbits.set_number_of_qbit_neighbours(qbit) 
           
    @property
    def number_of_plaqs(self):
        return len(self.found_plaqs())

    def visualize(self, ax=None, zoom=1, figsize=(15,15), core_corner=None, check_ancilla_in_core: bool=True):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x, y = list(zip(*self.envelop_rect()))
        ax.scatter(x, y, color="grey", s=0.6)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # color plaquettes
        for polygon in self.polygons_coords(
            self.qbits.qubit_to_coord_dict, self.polygons
        ):
            fill, facecolor, lw = False, None, 0
            if self.scope_of_polygon(polygon) == self.unit_square_scope:
                fill, facecolor, lw = True, "#3A6B35", 14
            if self.scope_of_polygon(polygon) == self.unit_triangle_scope:
                fill, facecolor, lw = True, "#CBD18F", 14
            patch = plt.Polygon(polygon, zorder=0, fill=fill, lw=lw, edgecolor='white', facecolor=facecolor)
            ax.add_patch(patch)
        # color qbits
        for qbit in self.qbits:
            label = ax.annotate(
                    r"{},{}".format(*qbit.qubit),
                    xy=qbit.coord, ha="center", va="center",
                    fontsize=13
                    )
            if qbit.core == False:
                circle = plt.Circle(
                        qbit.coord, radius=0.2, alpha=1.0, lw=0.7,
                        ec='black', fc='#FFC57F'
                        )
            if qbit.core == True:
                circle = plt.Circle(
                        qbit.coord, radius=0.2, alpha=1.0, lw=0.7,
                        ec='black', fc='#E3B448'
                        )
            if qbit.ancilla == True:
                circle = plt.Circle(
                        qbit.coord, radius=0.2, alpha=1.0, lw=0.7,
                        ec='black', fc='#FF7b7b'
                        )     
                if check_ancilla_in_core:
                    assert qbit.ancilla == True and qbit.core == True, "ancilla is not part of the core ?!"
            ax.add_patch(circle)  
            
        if core_corner is not None:
            (min_x, max_x), (min_y, max_y) = core_corner
            envelop = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
            patch = plt.Polygon(envelop, zorder=0, fill=False, lw=2, edgecolor='black', alpha=0.5)
            ax.add_patch(patch)            
        return ax
     
    def draw_lines(self, ax):
        """draw lines of nodes in plot"""
        for node in self.qbits.graph.nodes:
            qbits_path, _ = self.line_to_node(node)
            ax.plot([qbit.coord[0] for qbit in qbits_path],
                   [qbit.coord[1] for qbit in qbits_path])

    def modified_distance(self, x, y):
        """ distance between coord x and coord y
        if coords are adjacent, set line_scaling as distance"""
        distance = dist(x, y)
        # include 1.0 and diagonal distances
        if distance < 1.45:
            distance = self.line_scaling
        return distance
           
    @staticmethod
    def angle_between(point1, point2):
        """points are coords"""
        x1, y1 = point1
        x2, y2 = point2
        return - atan2(-(x2 - x1), y2 - y1)
    
    def get_closest_qbit(self, qbit, temporar_qbits):
        """ get closest qbit to qbits in temporat qbits"""
        qbits_by_distance = np.array([[qbit_, self.modified_distance(qbit.coord, qbit_.coord)] 
                                      for qbit_ in temporar_qbits if qbit!=qbit_])
        # get minimum distance
        minimum = min(qbits_by_distance, key=lambda x: x[1])[1]
        # are there several closest qbits with this minium distance
        qbits_by_distance = qbits_by_distance[qbits_by_distance.T[1] == minimum]
        # take closest qbit, sort them clockwise starting from 18:00
        qbits_by_distance = [(Polygons.angle_between(qbit.coord, close_qbit.coord), close_qbit, d) for close_qbit, d in qbits_by_distance]
        closest_qbit, distance = min(qbits_by_distance)[1:]
        return closest_qbit, distance
    
#         qbits_by_distance = [(qbit_,modified_distance(qbit.coord, qbit_.coord)) for qbit_ in temporar_qbits if qbit!=qbit_]
#         return min(qbits_by_distance, key=lambda x: x[1])

    def length_of_line_node(self, i):
        """
        computes the length of node i, 
        if qbits adjacent, the length is set to line_scaling
        """
        _, distances = self.line_to_node(i)
        return sum(distances)
        
    def line_to_node(self, i):
        """
        returns a line to a node in terms of its qbits,
        and the distance between these qbits
        if qbits adjacent, the length is set to line_scaling
        this function can be seen as an approximation to TSP
        """
        if i not in self.qbits.graph.nodes:
            return 0
        node_qbits = self.qbits_of_node(i)
        temporar_qbits = node_qbits[:]
        # cycle is starting with start_qbit (qbit with the most free neighbours, top and leftmost )
        start_qbit = min([((qbit.number_of_qbit_neighbours-8), -qbit.coord[1], -qbit.coord[0], qbit) for qbit in node_qbits])[-1]
        qbit = start_qbit
        distances = []
        qbit_path = [qbit]
        # loop through qbits, find closest qbit, remove start qbit, search from closest qbit again...
        for idx in range(len(node_qbits)):
            # if end of line is reached, connect starting qbit to form a cyle
            if len(temporar_qbits) == 1:
                temporar_qbits.append(start_qbit)
            closest_qbit, distance = self.get_closest_qbit(qbit, temporar_qbits)
            temporar_qbits.remove(qbit)
            qbit = closest_qbit
            distances.append(distance)
            qbit_path.append(qbit)
        # remove max distance from distances and cut path at max distance
        max_distance = distances.index(max(distances)) + 1 # +1 for proper use in list
        distances = distances[max_distance:] + distances[:max_distance-1]
        qbit_path = qbit_path[max_distance : -1] + qbit_path[:max_distance]
        return qbit_path, distances
    
    def qbits_of_node(self, i):
        """returns the qbits to node i"""
        return [qbit for qbit in self.qbits if i in qbit.qubit]

# TODO: create function to analyse search
# for i in range(100):
# #     delta_energy = 0#(mc.apply('',1))
# #     print("delta energy {} temperature {}".format(delta_energy, mc.temperature))
# #     if type(mc.qbits_to_move[1]) == tuple:
# #         print(f'move {mc.qbits_to_move[0].qubit} to {mc.qbits_to_move[1]} with probability %2.f'% min(1, np.exp(- delta_energy / mc.temperature)))
# #     else:
# #         print(f'switch {mc.qbits_to_move[0].qubit} with {mc.qbits_to_move[1].qubit} with probability %2.f'% min(1, np.exp(- delta_energy / mc.temperature)))
#     polygon_object.visualize()
#     plt.show()
