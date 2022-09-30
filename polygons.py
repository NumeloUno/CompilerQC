import numpy as np
import random
from itertools import combinations
from math import dist, atan2
from matplotlib import pyplot as plt
from shapely.geometry import MultiPoint
import galois
import warnings


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
        exponent: in the measure functions (scope or MoI)
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
    
    def coords_of_changed_polygons(self, polygons_of_interest):
        warnings.warn("function is deprecated, replace with coords_of_polygons()")
        self.coords_of_polygons(polygons_of_interest)
        
    def coords_of_polygons(self, polygons_of_interest):
        """
        return unique polygons coords which changed during the move
        """
        qubit_to_coord_dict = self.nodes_object.qbits.qubit_to_coord_dict
        return Polygons.polygons_coords(qubit_to_coord_dict, polygons_of_interest)
    
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

    @staticmethod
    def is_rectengular(polygon_coord):
        """
        check if a given polygon forms a rectengular
        """
        if len(polygon_coord) == 4:
            return sum([len(set(coords)) == 2 for coords in list(zip(*polygon_coord))]) == 2
        else:
            return False
        
    @staticmethod
    def move_to_center(qubit_coord_dict, K):
        """
        move coords in qubit_coord_dict
        to center of later initialization
        with K qubits
        """
        core_center_x, core_center_y = Polygons.center_of_coords(
            qubit_coord_dict.values()
        )
        center_envelop_of_all_coord = np.sqrt(K) / 2
        delta_cx, delta_cy = int(center_envelop_of_all_coord - core_center_x), int(
            center_envelop_of_all_coord - core_center_y
        )

        for qubit, coord in qubit_coord_dict.items():
            qubit_coord_dict[qubit] = tuple(np.add(coord, (delta_cx, delta_cy)))
        return qubit_coord_dict
            
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

    @staticmethod
    def scale(polygon_coord, radius):
        """
        shrunk polygon, 
        just for visualization reasons
        """
        center = Polygons.center_of_coords(polygon_coord)
        shrunk_polygon = []
        for coord in polygon_coord:
            l = np.subtract(coord, center)
            l = l * (1 - radius / (np.linalg.norm(l)))
            shrunk_polygon.append(tuple(np.add(l, center)))
        return shrunk_polygon
    
    @staticmethod
    def rotate_coords_by_45(coord, rotate: bool):
        if rotate:
            x, y = np.subtract(coord, (1,0))
            return (1 / np.sqrt(2) * (x + y) - 1.4, 1 / np.sqrt(2) * (-x + y))
        else:
            return coord

    def visualize(
        self,
        ax=None,
        zoom=1,
        figsize=(15, 15),
        core_corner=None,
        check_ancilla_in_core: bool = True,
        envelop_rect=None,
        rotate: bool=False,
        radius=0.2,
    ):
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal', 'box')
        ax.axis('off')

        if envelop_rect is None: envelop_rect = self.nodes_object.qbits.envelop_rect()
        ax.scatter(*list(zip(*envelop_rect)), color="white", s=0.6, alpha=0)

        if core_corner is not None:
            (min_x, max_x), (min_y, max_y) = core_corner
            envelop = [Polygons.rotate_coords_by_45(p, rotate) 
                       for p in [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]]
            patch = plt.Polygon(
                envelop, zorder=10, fill=False, lw=10, edgecolor="black", alpha=0.5
            )
            ax.add_patch(patch)

        # color plaquettes
        for polygon in self.polygons_coords(
            self.nodes_object.qbits.qubit_to_coord_dict, self.polygons
        ):
            # if center is part of polygon, error will be thrown
            if Polygons.center_of_coords(polygon) in polygon:
                continue
            fill, facecolor, lw, alpha, edge_alpha = False, None, 0, 0, 0
            if self.scope_measure:
                measure = self.scope_of_polygon(polygon)
            if not self.scope_measure:
                measure = self.moment_of_inertia(polygon)
            if measure == self.unit_square:
                fill, facecolor, lw, alpha, edge_alpha = True, "limegreen", 1, 0.5, 1
                polygon = Polygons.scale(polygon, radius)
            if measure == self.unit_triangle:
                fill, facecolor, lw, alpha, edge_alpha = True, "blue", 1, 0.3, 1
                polygon = Polygons.scale(polygon, radius)
            # if points are on a line, it is not a polygon anymore but a LineString object
            try: polygon = list(zip(*MultiPoint(polygon).convex_hull.exterior.xy))
            except: pass
            polygon = [Polygons.rotate_coords_by_45(p, rotate)
                       for p in polygon]
            patch = plt.Polygon(
                polygon,
                zorder=0,
                fill=fill,
                lw=0,
                facecolor=facecolor,
                alpha=alpha,
            )
            edge = plt.Polygon(
                polygon,
                zorder=0,
                fill=False,
                lw=lw,
                edgecolor="black",
                alpha=edge_alpha
            )
            ax.add_patch(patch)
            ax.add_patch(edge)
        # color qbits
        for qbit in self.nodes_object.qbits:
            coord = Polygons.rotate_coords_by_45(qbit.coord, rotate)
            if len(str(qbit.qubit)) - 4 == 2:
                fontsize = 82 * radius
            if len(str(qbit.qubit)) - 4 == 3:
                fontsize = 70 * radius
            if len(str(qbit.qubit)) - 4 == 4:
                fontsize = 60 * radius

            label = ax.annotate(
                r"{},{}".format(*qbit.qubit),
                xy=coord,
                ha="center",
                va="center",
                fontsize=fontsize,
                zorder=20,
            )
            if qbit.core == False:
                circle = plt.Circle(
                    coord, radius=radius, alpha=1.0, lw=0.7, ec="black", fc="antiquewhite", zorder=20
                )
            if qbit.core == True:
                circle = plt.Circle(
                    coord, radius=radius, alpha=1.0, lw=0.7, ec="black", fc="tan"
                )
            if qbit.ancilla == True:
                circle = plt.Circle(
                    coord, radius=radius, alpha=1.0, lw=0.7, ec="black", fc="lightcoral"
                )
                if check_ancilla_in_core:
                    assert (
                        qbit.ancilla == True and qbit.core == True
                    ), "ancilla is not part of the core ?!"
            ax.add_patch(circle)
        return ax

    def draw_lines(self, ax, rotate: bool=False):
        """draw lines of nodes in plot
        before calling this function, lines have to be initialized, 
        easiest way to do so is by calling energy.line_energy()
        """
        for node in self.nodes_object.qbits.graph.nodes:
            qbits_path, _ = self.line_to_node(node)
            a = ax.plot(
                [Polygons.rotate_coords_by_45(qbit.coord, rotate)[0] for qbit in qbits_path],
                [Polygons.rotate_coords_by_45(qbit.coord, rotate)[1] for qbit in qbits_path],
                linewidth = 10,
                alpha=0.8
            )
            a[0].set_solid_capstyle('round')
        return ax

    def color_coords(
        self,
        ax,
        coords_and_color,
        rotate: bool=False,
        radius: float=3000,
    ):
        """color coords according to color in coords_and_color,
        coords_and_color is of the from [((x,y), float between 0 and 1), ...]"""
        for coord, color in coords_and_color:
            coord = Polygons.rotate_coords_by_45(coord, rotate)
            ax.scatter(*coord, color=str(color), s=radius)

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

    @staticmethod
    def number_of_swap_gates(polygon_coord):
        """
        converts the distance to the qbit 
        (which is closest to Center)
        in swap counts
        """
        center = np.mean(polygon_coord, axis=0)
        closest_coord_to_center = sorted([(dist(coord, center), coord) for coord in polygon_coord])[0][1]
        number_of_swap_gates = 0
        for coord in polygon_coord:
            # -1 since it should not be on the same place as the center qbit
            distance_to_closest_coord = np.abs(np.subtract(closest_coord_to_center, coord)).sum() - 1
            if distance_to_closest_coord > 0:
                number_of_swap_gates += distance_to_closest_coord
        return number_of_swap_gates
    
    def get_generating_constraints(self, constraints: list=None):
        """
        returns dict of generating constraints
        constraints: dict of min qubit of constraint: constraint
        """
        coords_of_plaquettes = constraints
        if coords_of_plaquettes is None:
            coords_of_plaquettes = self.coords_of_polygons(self.nodes_object.qbits.found_plaqs())
        coord_to_qbit_dict = self.nodes_object.qbits.coord_to_qbit_dict
        # if basis froms already echelon form matrix, skip this part
        if set(map(min, sorted(coords_of_plaquettes))) != len(coords_of_plaquettes):
            qubit_to_idx = {qubit: idx for idx, qubit in enumerate(sorted(coord_to_qbit_dict))}
            idx_to_qubit = {idx: qubit for idx, qubit in enumerate(sorted(coord_to_qbit_dict))}
            a = np.zeros((len(coords_of_plaquettes), len(qubit_to_idx)), dtype=int)
            for j, plaq in enumerate(coords_of_plaquettes):
                for qubit in plaq:
                    a[j][qubit_to_idx[qubit]] = 1
            GF = galois.GF(2)
            g = GF(x=a.astype(int))
            coords_of_plaquettes = [[idx_to_qubit[i] for i in (np.where(j==1))[0]] for j in g.row_reduce()]

        generating_constraints = {constraint[0]:set(constraint) for constraint in sorted(coords_of_plaquettes)}
        return generating_constraints

    @staticmethod
    def is_constraint_fulfilled(test_constraint: list, constraints: dict):
        """
        check if test_constraint is implicitly fulfilled
        by constraints (which are generators or basis)
        constraints: dict of min qubit of plaquette: plaquette
        """
        test_constraint = (sorted(test_constraint))
        while len(test_constraint) > 0:
            constraint = constraints.get(test_constraint[0])
            if constraint is None:
                return False
            test_constraint = set(test_constraint)
            test_constraint = sorted((test_constraint - constraint).union(constraint - test_constraint))
        return not test_constraint

