import numpy as np
import random
import networkx as nx
import itertools
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon, MultiPoint, MultiLineString, LineString
from shapely.ops import polygonize
from shapely import wkt
from CompilerQC import Graph


class Polygons:
    """
    this class converts the edges of a graph to qbits
    and puts them on a grid. the qbits are connected to 
    other qbits if they formed a closed cyle (3 or 4 length) 
    in the graph. So the qbits are connected to polygons of
    length three or four. Some qbits are part of the core 
    (definded by the core bipartite sets). If the class is 
    initialized with a qbit to coord dict, the coords are not 
    thrown randomly on the grid.
    """
    def __init__(
        self,
        logical_graph: Graph,
        core_qbit_coord_dict = dict(),
        qbit_coord_dict: dict=None,
    ):
        """
        logical_graph: logical graph with N nodes and K edges
        """
        self.qbits = logical_graph.qbits_from_graph()
        self.N = logical_graph.N
        self.K = logical_graph.K
        self.C = logical_graph.C
        cycles = logical_graph.get_cycles(3) + logical_graph.get_cycles(4)
        self.polygons = Polygons.get_all_polygons(cycles)
        if qbit_coord_dict is None:
            qbit_coord_dict = self.init_coords_for_qbits()
        self.qbit_coord_dict = qbit_coord_dict
        # define qbits and their coords for core and outside core
        self.core_qbits = list(core_qbit_coord_dict.keys())
        self.movable_qbits = list(set(self.qbits) - set(self.core_qbits))
        self.movable_coords = [
            self.qbit_coord_dict[qbit] for qbit in self.movable_qbits
        ]
        self.core_coords = list(core_qbit_coord_dict.values())
        self.update_qbits_coords(self.core_qbits, self.core_coords)
        from warnings import warn
        warn("The maximal grid size is 10 x 10, thats ok since all problems so far can be put on a grid of this size. Of course, this has to be generalized")
    @classmethod
    def from_max_core_bipartite_sets(
            cls,
            logical_graph: Graph,
            core_bipartite_sets: list = [[], []],
            ):
        """
        initialize Polygons object by using bipartite core sets U and V
        """
        cls.U, cls.V = core_bipartite_sets
        core_qbit_coord_dict = Polygons.core_qbits_and_coords_from_sets(
            cls.U,
            cls.V,
        )
        return cls(logical_graph=logical_graph,core_qbit_coord_dict=core_qbit_coord_dict)

    @staticmethod
    def core_qbits_and_coords_from_sets(
        U: list,
        V: list,
    ):
        """
        input: two sets of logical bits, generating the
        complete bipartite graph
        return: list of qbits and their coords
        which build the core of the physical graph
        """
        # coords
        x, y = np.meshgrid(np.arange(len(U)), np.arange(len(V)))
        coords = list(zip(x.flatten(), y.flatten()))
        # qbits
        lbit1, lbit2 = np.meshgrid(U, V)
        qbits = list(zip(lbit1.flatten(), lbit2.flatten()))
        sorted_qbits = list(map(tuple, map(sorted, qbits)))
        return dict(zip(sorted_qbits, coords))

    def update_qbits_coords(
        self,
        qbits: list,
        new_coords: list,
    ):
        """
        qbits: list of tuples
        new_coords: list of tuples
        if new_coord is already occupied by a qbit,
        this qbit is moved to the max coord + (1, 0)
        """
        # copying the lists to detach them from class, otherwise lists are changed during loop
        qbits, new_coords = qbits[:], new_coords[:]
        for qbit, new_coord in zip(qbits, new_coords):
            qbit_coords = self.qbit_coord_dict.values()
            if new_coord in qbit_coords:
                qbit_to_move = dict(
                    zip(self.qbit_coord_dict.values(), self.qbit_coord_dict.keys())
                )[new_coord]
                radius = 1
                while True:
                    free_neighbours = self.free_neighbour_coords(
                    qbit_coords, radius=radius)
                    if free_neighbours != []:
                        break
                    else:
                        radius += 1
                move_to_coord = random.choice(free_neighbours)
                self.qbit_coord_dict[qbit_to_move] = move_to_coord
                self.update_core_and_movable_coords(qbit_to_move, move_to_coord)
            self.qbit_coord_dict[qbit] = new_coord
            self.update_core_and_movable_coords(qbit, new_coord)

    def update_core_and_movable_coords(
        self,
        qbit: tuple,
        new_coord: tuple,
    ):
        """
        update core_coords or movable_coords
        """
        if qbit in self.core_qbits:
            self.core_coords[self.core_qbits.index(qbit)] = new_coord
        elif qbit in self.movable_qbits:
            self.movable_coords[self.movable_qbits.index(qbit)] = new_coord
        else:
            raise AssertionError(
                "qbit is neither in the core nor in the movable part", qbit
            )

    @staticmethod
    def get_polygon_from_cycle(cycle: list):
        """
        cycle: cycle in graph
        returns polygon
        """
        cycle = cycle + [cycle[0]]
        polygon = list(map(tuple, map(sorted, zip(cycle, cycle[1:]))))
        return polygon

    def init_coords_for_qbits(self):
        """
        initialization of qbits with random coordinates
        return: dictionary with qbits and coordinates
        """
        coords = list(np.ndindex(min(10, self.N), min(10, self.N)))
        np.random.shuffle(coords)
        return dict(zip(self.qbits, coords[: self.K]))

    def get_coords_of_polygon(self, polygon):
        """
        return: list of coords of single polygon
        """
        return list(map(self.qbit_coord_dict.get, polygon))

    @staticmethod
    def polygon_length(polygon_coord):
        return Polygon(polygon_coord).length

    @staticmethod
    def polygon_area(polygon_coord):
        return Polygon(polygon_coord).area

    @staticmethod
    def get_all_polygons(cycles):
        return list(map(Polygons.get_polygon_from_cycle, cycles))

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
        also from sanduhr plaquettes
        """
        scope = [
            LineString([*x]).length for x in itertools.combinations(polygon_coord, 2)
        ]
        return sum(scope) - (4 + 2 * np.sqrt(2))

        #return Polygons.polygon_length(polygon_coord) - 4

    @staticmethod
    def is_unit_triangle(polygon_coord):
        """
        measure closeness to plaquette
        """
        return Polygons.polygon_length(polygon_coord) - (2 + np.sqrt(2))

    @staticmethod
    def neighbours(coord: tuple, radius: int=1):
        """ return list of all direct neighbours (to coord) by default,
        if radius is larger than one, more neighbours are considered
        """
        range_ = range(- radius, radius + 1)
        return [(x, y) for x in range_ for y in range_ if (x, y) != coord]

    def free_neighbour_coords(self, qbit_coords: list, radius: int=1):
        """
        return list of all free neighbours coords (which are on the allowed grid) for qbits_coords
        """
        neighbour_list = []
        for qbit_coord in qbit_coords:
            neighbour_list += Polygons.neighbours(qbit_coord, radius=radius)
        neighbour_list = list(set(neighbour_list) - set(qbit_coords))
        grid_coords = list(np.ndindex(min(10, self.N), min(10, self.N)))
        return list(set(neighbour_list).intersection(grid_coords))

    def inside_core_coords(self):
        """
        return: list of coords which are inside the core,
        not at the edge of it
        """
        neighbour_coords = map(Polygons.neighbours, self.core_coords)
        is_inside_core = list(
            map(lambda s: set(s).issubset(self.core_coords), neighbour_coords)
        )
        inside_coords = list(itertools.compress(self.core_coords, is_inside_core))
        return inside_coords

    def polygons_outside_core(self):
        """
        returns all polygons which are not connected to
        coords inside the core
        """
        inside_coords = self.inside_core_coords()
        return [
            coords
            for coords in self.get_all_polygon_coords()
            if set(coords).isdisjoint(inside_coords)
        ]

    def convex_hull(self, qbit_coords: list = None):
        """
        return envelop polygon of qbit_coords, if
        qbit_coords are None, all qbits coords are considered
        """
        if qbit_coords is None:
            qbit_coords = list(self.qbit_coord_dict.values())
        qbits = MultiPoint(qbit_coords)
        envelop_polygon = qbits.convex_hull
        return list(envelop_polygon.exterior.coords)

    def center_of_convex_hull(
        self,
        qbit_coords: list = None,
    ):
        """
        return center coords of enevelop of qbit_coords,
        if None, all qbits are considered
        """
        center = Polygon(self.convex_hull(qbit_coords)).centroid
        return (center.x, center.y)

    def qbit_distances_to_center(
        self,
        qbit_coords: list = None,
    ):
        """
        list of distances for each qbit to center of qbit_coords,
        could e.g be the core_coords --> core_center
        """
        center = self.center_of_convex_hull(qbit_coords)
        if qbit_coords is None:
            qbit_coords = self.qbit_coord_dict.values()
        distances = list(map(lambda x: Point(center).distance(Point(x)), qbit_coords))
        return distances

    def move_center_to_middle(self):
        """
        calculates the CoM of the convex hull
        and moves the hull to the center of the grid
        """
        center_x, center_y = map(int, self.center_of_convex_hull())
        middle_x = middle_y = self.K // 2
        move_x, move_y = center_x - middle_x, center_y - middle_y
        new_coords = [
            (coord[0] - move_x, coord[1] - move_y)
            for coord in self.qbit_coord_dict.values()
        ]
        self.update_qbits_coords(self.qbits, new_coords)
        
    def n_found_plaqs(self):
        """
        number of found plaqs in current layout
        """
        from warnings import warn
        warn("the function n_found_plaqs in polygons.py will be removed, dont use it anymore")
        return len(self.found_plaquettes())

    def visualize(self, ax, polygon_coords, zoom=1):
        x, y = np.meshgrid(np.arange(self.K), np.arange(self.K))
        ax.scatter(x.flatten(), y.flatten(), color="grey", s=0.6)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for polygon in polygon_coords:
            fill, facecolor = False, None
            if len(polygon) == 4 and self.is_unit_square(polygon) == 0:
                fill, facecolor = True, "lightblue"
            if len(polygon) == 3 and self.is_unit_triangle(polygon) == 0:
                fill, facecolor = True, "indianred"
            patch = plt.Polygon(polygon, zorder=0, fill=fill, lw=0, facecolor=facecolor)
            ax.add_patch(patch)
        for qbit, coord in self.qbit_coord_dict.items():
            ax.annotate(str(qbit), coord)
            ax.scatter(*coord, color="red")
        x_range = sorted(self.convex_hull(), key=lambda x: x[0])
        y_range = sorted(self.convex_hull(), key=lambda y: y[1])
        ax.set_xlim(
            min(x_range[0][0], y_range[0][1]) - zoom,
            max(x_range[-1][0], y_range[-1][1]) + zoom,
        )
        ax.set_ylim(
            min(x_range[0][0], y_range[0][1]) - zoom,
            max(x_range[-1][0], y_range[-1][1]) + zoom,
        )
        return ax

    def found_plaquettes(self):
        """
        list all plaquettes which are in the current layout
        """
        plaquettes = np.array(self.polygons, dtype=object)[
                np.array([Polygons.is_unit_square(coord)
                if len(coord) == 4
                else Polygons.is_unit_triangle(coord)
                for coord in self.get_all_polygon_coords()])
                == 0 ]
        return [list(map(tuple, plaquette)) for plaquette in plaquettes]

    def to_nx_graph(self):
        """
        converts the current layout to a graph,
        only valid polygons (plaquettes) are considered
        """
        edges = list(map(Polygons.get_polygon_from_cycle, self.found_plaquettes()))
        edges = [edge for edge_list in edges for edge in edge_list]

        physical_graph = nx.Graph()
        physical_graph.add_nodes_from(self.qbits)
        physical_graph.add_edges_from(edges)

        return nx.to_undirected(physical_graph)
