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

    def __init__(
        self,
        nodes_object: "Nodes",
        polygons: "list of polygons" = None,
        line_scaling: float = 0,
        exponent: float = 1,
        scope_measure: bool=True,
    ):
        """
        qbits: qbits object, qbits represent connections in logical grpah
        polygons:
        line_scaling: if the distance between two qbits is 1.0 or sqrt(2), hence the qbits are
        adjacent, set their distance to line_scaling
        if not scope_measure, then MoI_measure is choosen
        """
        self.nodes_object = nodes_object
        if polygons is None:
            polygons = Polygons.create_polygons(self.nodes_object.qbits.graph.cycles)
        self.polygons = polygons
        self.nodes_object.qbits.set_all_polygons(self.polygons)

        self.line_scaling = line_scaling
        self.core_corner = None
        
        self.exponent = exponent
        self.scope_measure = scope_measure
        self.set_unit_measure()

    def set_unit_measure(self):
        """set the unit scope or MoI of an triangle or square plaquette"""
        if self.scope_measure:
            self.unit_triangle = self.scope_of_polygon([(0,1),(0,2), (1,2)])
            self.unit_square = self.scope_of_polygon([(0,1),(0,2), (1,2),(1,1)])
        if not self.scope_measure:
            self.unit_triangle = self.moment_of_inertia([(0,1),(0,2), (1,2)])
            self.unit_square = self.moment_of_inertia([(0,1),(0,2), (1,2),(1,1)])   
            
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

    def scope_of_polygon(self, polygon_coord):
        """
        return scope of the polygon
        """
        return round(
            sum([dist(*c) ** self.exponent for c in list(combinations(polygon_coord, 2))])
            / len(polygon_coord),
            5,
        )

    def moment_of_inertia(self, polygon_coords):
        center = np.mean(polygon_coords, axis=0)
        return round(
            sum([dist(coord, center) ** self.exponent for coord in polygon_coords]), 5)
    
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

    def set_plaquettes_of_qbits(self):
        """
        set the plaquettes of each qbit
        """
        found_plaquettes = self.nodes_object.qbits.found_plaqs()
        for qbit in self.nodes_object.qbits:
            qbit.plaquettes = [plaq for plaq in found_plaquettes if qbit.qubit in plaq]

    def set_numbers_of_qbits_neighbours(self):
        """ "set number of neighbours each qbit has"""
        for qbit in self.nodes_object.qbits:
            self.nodes_object.qbits.set_number_of_qbit_neighbours(qbit)

    @property
    def number_of_plaqs(self):
        return len(self.nodes_object.qbits.found_plaqs())

    def visualize(
        self,
        ax=None,
        zoom=1,
        figsize=(15, 15),
        core_corner=None,
        check_ancilla_in_core: bool = True,
        envelop_rect=None,
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        if envelop_rect is None:
            envelop_rect = self.nodes_object.qbits.envelop_rect()
        x, y = list(zip(*envelop_rect))
        ax.scatter(x, y, color="grey", s=0.6)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # color plaquettes
        for polygon in self.polygons_coords(
            self.nodes_object.qbits.qubit_to_coord_dict, self.polygons
        ):
            fill, facecolor, lw = False, None, 0
            if self.scope_measure:
                measure = self.scope_of_polygon(polygon)
            if not self.scope_measure:
                measure = self.moment_of_inertia(polygon)
            if measure == self.unit_square:
                fill, facecolor, lw = True, "#3A6B35", 14
            if measure == self.unit_triangle:
                fill, facecolor, lw = True, "#CBD18F", 14
            patch = plt.Polygon(
                polygon,
                zorder=0,
                fill=fill,
                lw=lw,
                edgecolor="white",
                facecolor=facecolor,
            )
            ax.add_patch(patch)
        # color qbits
        for qbit in self.nodes_object.qbits:
            label = ax.annotate(
                r"{},{}".format(*qbit.qubit),
                xy=qbit.coord,
                ha="center",
                va="center",
                fontsize=13,
            )
            if qbit.core == False:
                circle = plt.Circle(
                    qbit.coord, radius=0.2, alpha=1.0, lw=0.7, ec="black", fc="#FFC57F"
                )
            if qbit.core == True:
                circle = plt.Circle(
                    qbit.coord, radius=0.2, alpha=1.0, lw=0.7, ec="black", fc="#E3B448"
                )
            if qbit.ancilla == True:
                circle = plt.Circle(
                    qbit.coord, radius=0.2, alpha=1.0, lw=0.7, ec="black", fc="#FF7b7b"
                )
                if check_ancilla_in_core:
                    assert (
                        qbit.ancilla == True and qbit.core == True
                    ), "ancilla is not part of the core ?!"
            ax.add_patch(circle)

        if core_corner is not None:
            (min_x, max_x), (min_y, max_y) = core_corner
            envelop = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
            patch = plt.Polygon(
                envelop, zorder=0, fill=False, lw=2, edgecolor="black", alpha=0.5
            )
            ax.add_patch(patch)
        return ax

    def draw_lines(self, ax):
        """draw lines of nodes in plot"""
        for node in self.nodes_object.qbits.graph.nodes:
            qbits_path, _ = self.line_to_node(node)
            ax.plot(
                [qbit.coord[0] for qbit in qbits_path],
                [qbit.coord[1] for qbit in qbits_path],
            )

    def color_coords(self, coords_and_color, figsize=(15, 15)):
        """color coords according to color in coords_and_color,
        coords_and_color is of the from [((x,y), float between 0 and 1), ...]"""
        _, ax = plt.subplots(figsize=figsize)
        for coord, color in coords_and_color:
            ax.scatter(*coord, color=str(color), s=3000)
        self.visualize(ax=ax)

    def modified_distance(self, x, y):
        """distance between coord x and coord y
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
        return -atan2(-(x2 - x1), y2 - y1)

    def get_closest_qbit(self, qbit, temporar_qbits):
        """get closest qbit to qbits in temporat qbits"""
        qbits_by_distance = np.array(
            [
                [qbit_, self.modified_distance(qbit.coord, qbit_.coord)]
                for qbit_ in temporar_qbits
                if qbit != qbit_
            ]
        )
        # get minimum distance
        minimum = min(qbits_by_distance, key=lambda x: x[1])[1]
        # are there several closest qbits with this minium distance
        qbits_by_distance = qbits_by_distance[qbits_by_distance.T[1] == minimum]
        # take closest qbit, sort them clockwise starting from 18:00
        qbits_by_distance = [
            (Polygons.angle_between(qbit.coord, close_qbit.coord), close_qbit, d)
            for close_qbit, d in qbits_by_distance
        ]
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
        if i not in self.nodes_object.qbits.graph.nodes:
            return 0
        node_qbits = self.nodes_object.qbits_of_nodes([i])
        temporar_qbits = node_qbits[:]
        # cycle is starting with start_qbit (qbit with the most free neighbours, top and leftmost )
        start_qbit = min(
            [
                (
                    (qbit.number_of_qbit_neighbours - 8),
                    -qbit.coord[1],
                    -qbit.coord[0],
                    qbit,
                )
                for qbit in node_qbits
            ]
        )[-1]
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
        max_distance = distances.index(max(distances)) + 1  # +1 for proper use in list
        distances = distances[max_distance:] + distances[: max_distance - 1]
        qbit_path = qbit_path[max_distance:-1] + qbit_path[:max_distance]
        return qbit_path, distances

    def qbits_of_node(self, i):
        """returns the qbits to node i"""
        return [qbit for qbit in self.nodes_object.qbits if i in qbit.qubit]

    def add_ancillas_to_polygons(self, ancillas, only_four_cycles: bool = False):
        """find all cycles, or only four cycles if True,
        create polygons and extend them"""
        cycles = self.nodes_object.qbits.graph.get_cycles_of_ancillas(ancillas, 4)
        if not only_four_cycles:
            cycles += self.nodes_object.qbits.graph.get_cycles_of_ancillas(ancillas, 3)
        polygons = Polygons.create_polygons(cycles)
        self.polygons += polygons
        return polygons

    def remove_ancillas_from_polygons(self, ancillas):
        """remove ancillas from all polygons"""
        remaining_polygons = []
        for polygon in self.polygons:
            if set(ancillas).isdisjoint(polygon):
                remaining_polygons.append(polygon)
        self.polygons = remaining_polygons
