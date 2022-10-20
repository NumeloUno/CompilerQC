from matplotlib import pyplot as plt
import numpy as np
from CompilerQC import *

class Circle:
    def __init__(
        self,
        coord: tuple,
        name: str=None,
        color: str='lightgrey',
        size: float=0.2,
        fontsize: float=10,
        lw: float=1,
        alpha: float=1,
        edgecolor: str="black",
        up: bool=False,
        down: bool=False,
        zorder: int=10,
    ):
        self.name = name
        self.coord = coord
        self.color = color
        self.size = size
        self.fontsize = fontsize
        self.alpha = alpha
        self.linewidth = lw
        self.zorder = 5
        self.edgecolor = edgecolor
        self.up, self.down = up, down
    @property
    def patch(self):
        patch = plt.Circle(
            self.coord,
            radius=self.size,
            alpha=self.alpha,
            lw=self.linewidth,
            ec=self.edgecolor,
            fc=self.color,
            zorder=self.zorder
        )
        return patch
        
class Edge:
    def __init__(
        self,
        edge: list,
        color: str='lightgrey',
        lw: float=1,
        alpha: float=1,
        edgecolor: str='black',
        ew: float=1,
        zorder: int=0,
    ):
        self.edge = edge
        self.color = color
        self.edgecolor = edgecolor
        self.edgewidth = ew
        self.linewidth = lw
        self.alpha = alpha
        self.zorder = 0
    @property
    def patch(self):
        patch = plt.Polygon(
        self.edge,
        zorder=self.zorder,
        lw=self.linewidth,
        facecolor=self.color,
        fill=True,
        edgecolor=self.edgecolor,
        alpha=self.alpha)
        return patch
    @property
    def xy(self):
        x, y = list(zip(*self.edge))
        return x, y
class Edges:
    def __init__(
        self,
        edges: list=[]
    ):
        self.edges = [Edge(edge) for edge in edges]
    def add_edge(self, edge):
        self.edges.append(edge)
    def __iter__(self):
        for item in self.edges:
            yield item
    @property
    def patches(self):
        patches = []
        for edge in self.edges:
            patches.append(edge.patch)
        return patches


            
class Circles:
    def __init__(
        self,
        circles: list=[]
    ):
        self.circles = circles
    def add_circle(self, circle):
        self.circles.append(circle) 
        
    @property
    def xy(self):
        x, y = zip(*[circle.coord for circle in self.circles])
        return x, y

    def __iter__(self):
        for item in self.circles:
            yield item
    @property
    def patches(self):
        patches = []
        for circle in self.circles:
            patches.append(circle.patch)
        return patches
    @property
    def names(self):
        return [circle.name for circle in self.circles]
    @property
    def coords(self):
        return [circle.coord for circle in self.circles]
    @property
    def name_to_coord_dict(self):
        return {circle.name:circle for circle in self.circles}
    @staticmethod
    def get_circles_of_n_eck(n):
        return list(zip([np.round(np.exp(1j * 2 * np.pi * k / n).real, 3) for k in range(n)], [np.round(np.exp(1j * 2 * np.pi * k / n).imag, 3) for k in range(n)]))

    
def draw_LHZ_from_spin_glass(polygon_object, name_to_coord_dict, polygons, colors_of_polygon):
    self = polygon_object
    ax=None
    zoom=1
    figsize=(15, 15)
    core_corner=None
    check_ancilla_in_core: bool = True
    envelop_rect=None
    rotate: bool=True
    radius=0.2

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
        
        if sum([name_to_coord_dict[i].up for i in qbit.qubit]) % 2 == 0:
            circle = plt.Circle(
                coord, radius=radius, alpha=1.0, lw=0.7, ec="black", fc="#FFC0CB", zorder=20
            )
        else:
            circle = plt.Circle(
                coord, radius=radius, alpha=1.0, lw=0.7, ec="black", fc="lavender", zorder=20
            )

        ax.add_patch(circle)
    for idx, polygon in enumerate(polygons):
        polygon = [Polygons.rotate_coords_by_45(coord, rotate) for coord in polygon]
        patch = plt.Polygon(
            polygon,
            zorder=0,
            lw=8,
            alpha=0.7,
            color=colors_of_polygon[idx],
            fill=False
        )
        ax.add_patch(patch)
    return ax


def visualize_ancilla(polygon_object, ax, polygons):
    
    N = 6
    rotate = True
    graph = Graph.init_without_edges(N, [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)])
    qbits = Qbits.init_qbits_from_dict(graph, dict(zip(graph.qubits, graph.qubits)), assign_to_core=False)
    qbits[(1,4)].ancilla = True
    qbits[(1,4)].core = True
    nodes_object = Nodes(qbits, place_qbits_in_lines=False)
    polygon_object = Polygons(nodes_object, scope_measure=True)
    polygon_object.polygons = polygons

    self = polygon_object
    zoom=1
    figsize=(15, 15)
    core_corner=None
    check_ancilla_in_core: bool = True
    envelop_rect=None
    rotate: bool=False
    radius=0.2

    if ax is None: _, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal', 'box')
    ax.axis('off')

    if envelop_rect is None: envelop_rect = self.nodes_object.qbits.envelop_rect()
    ax.scatter(*list(zip(*envelop_rect)), color="white", s=0.6, alpha=0)

    # color plaquettes
    for polygon in self.polygons_coords(
        self.nodes_object.qbits.qubit_to_coord_dict, self.polygons
    ):
        fill, facecolor, lw, alpha, edge_alpha = False, None, 0, 0, 0
        measure = self.scope_of_polygon(polygon)

        if measure == self.unit_square:
            fill, facecolor, lw, alpha, edge_alpha = True, "limegreen", 1, 0.5, 1
            polygon = Polygons.scale(polygon, radius)
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
        if qbit.core == False:
            circle = plt.Circle(
                coord, radius=radius, alpha=1.0, lw=0.7, ec="black", fc="antiquewhite", zorder=20
            )
        if qbit.ancilla == True:
            circle = plt.Circle(
                coord, radius=radius, alpha=1.0, lw=0.7, ec="black", fc="lightcoral"
            )
        ax.add_patch(circle)
        
        
def visualize_without_labels(
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
            "",
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
