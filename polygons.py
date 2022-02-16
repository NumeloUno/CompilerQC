import numpy as np
import itertools
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon, MultiPoint, MultiLineString, LineString
from shapely.ops import polygonize
from shapely import wkt
from CompilerQC import Graph


class Polygons:
    def __init__(
        self,
        logical_graph: Graph,
        core_qbit_coord_dict: dict,
    ):
        """
        logical_graph: logical graph with N nodes and K edges
        """
        self.qbits = logical_graph.qbits_from_graph()
        self.N = logical_graph.N
        self.K = logical_graph.K
        self.C = logical_graph.C
        cycles = logical_graph.get_cycles(3) + logical_graph.get_cycles(4)
        self.polygons = self.get_all_polygons(cycles)
        self.qbit_coord_dict = self.init_coords_for_qbits()
        # define qbits and their coords for core and outside core
        self.core_qbits = list(core_qbit_coord_dict.keys())
        self.movable_qbits = list(set(self.qbits) - set(self.core_qbits))
        self.movable_coords = [
            self.qbit_coord_dict[qbit] for qbit in self.movable_qbits
        ]
        self.core_coords = list(core_qbit_coord_dict.values())
        self.update_qbits_coords(self.core_qbits, self.core_coords)

    @classmethod
    def from_max_core_bipartite_sets(
            cls,
            logical_graph: Graph,
            core_bipartite_sets: list = [[], []],
            ):
        """
        initialize Polygons object by using bipartite core sets U and V
        """
        self.U, self.V = core_bipartite_sets
        core_qbit_coord_dict = Polygons.core_qbits_and_coords_from_sets(
            self.U,
            self.V,
        )
        return cls(logical_graph, core_qbit_coord_dict)

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

    # TODO: if qbit not in qbits, raise error. move_to_coord should be better choosen than max
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
                move_to_coord = tuple(np.add(max(qbit_coords), (1, 0)))
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

    @classmethod
    def get_polygon_from_cycle(self, cycle: list):
        """
        cycle: cycle in graph
        """
        cycle = cycle + [cycle[0]]
        polygon = list(map(tuple, map(sorted, zip(cycle, cycle[1:]))))
        return polygon

    def init_coords_for_qbits(self):
        """
        initialization of qbits with random coordinates
        return: dictionary with qbits and coordinates
        """
        coords = list(np.ndindex(self.K, self.K))
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
        scope = [
            LineString([*x]).length for x in itertools.combinations(polygon_coord, 2)
        ]
        return sum(scope) - (4 + 2 * np.sqrt(2))

    #        return Polygons.polygon_length(polygon_coord) - 4

    @staticmethod
    def is_unit_triangle(polygon_coord):
        """
        measure closeness to plaquette
        """
        return Polygons.polygon_length(polygon_coord) - (2 + np.sqrt(2))

    @staticmethod
    def neighbours(coords: tuple):
        x, y = coords
        return [
            (x, y + 1),
            (x + 1, y + 1),
            (x + 1, y),
            (x + 1, y - 1),
            (x, y - 1),
            (x - 1, y - 1),
            (x - 1, y),
            (x - 1, y + 1),
        ]

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
        return [
            self.is_unit_square(coord)
            if len(coord) == 4
            else self.is_unit_triangle(coord)
            for coord in self.get_all_polygon_coords()
        ].count(0)

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
