from matplotlib import pyplot as plt
import numpy as np
from CompilerQC import *
from pathlib import Path
import os
import matplotlib as mpl
import pandas as pd


class Circle:
    def __init__(
        self,
        coord: tuple,
        name: str = None,
        color: str = "lightgrey",
        size: float = 0.2,
        fontsize: float = 10,
        lw: float = 1,
        alpha: float = 1,
        edgecolor: str = "black",
        up: bool = False,
        down: bool = False,
        zorder: int = 10,
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
            zorder=self.zorder,
        )
        return patch


class Edge:
    def __init__(
        self,
        edge: list,
        color: str = "lightgrey",
        lw: float = 1,
        alpha: float = 1,
        edgecolor: str = "black",
        ew: float = 1,
        zorder: int = 0,
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
            alpha=self.alpha,
        )
        return patch

    @property
    def xy(self):
        x, y = list(zip(*self.edge))
        return x, y


class Edges:
    def __init__(self, edges: list = []):
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
    def __init__(self, circles: list = []):
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
        return {circle.name: circle for circle in self.circles}

    @staticmethod
    def get_circles_of_n_eck(n):
        return list(
            zip(
                [np.round(np.exp(1j * 2 * np.pi * k / n).real, 3) for k in range(n)],
                [np.round(np.exp(1j * 2 * np.pi * k / n).imag, 3) for k in range(n)],
            )
        )


def draw_LHZ_from_spin_glass(
    polygon_object, name_to_coord_dict, polygons, colors_of_polygon
):
    self = polygon_object
    ax = None
    zoom = 1
    figsize = (15, 15)
    core_corner = None
    check_ancilla_in_core: bool = True
    envelop_rect = None
    rotate: bool = True
    radius = 0.2

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")
    ax.axis("off")

    if envelop_rect is None:
        envelop_rect = self.nodes_object.qbits.envelop_rect()
    ax.scatter(*list(zip(*envelop_rect)), color="white", s=0.6, alpha=0)

    if core_corner is not None:
        (min_x, max_x), (min_y, max_y) = core_corner
        envelop = [
            Polygons.rotate_coords_by_45(p, rotate)
            for p in [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
        ]
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
        try:
            polygon = list(zip(*MultiPoint(polygon).convex_hull.exterior.xy))
        except:
            pass
        polygon = [Polygons.rotate_coords_by_45(p, rotate) for p in polygon]
        patch = plt.Polygon(
            polygon,
            zorder=0,
            fill=fill,
            lw=0,
            facecolor=facecolor,
            alpha=alpha,
        )
        edge = plt.Polygon(
            polygon, zorder=0, fill=False, lw=lw, edgecolor="black", alpha=edge_alpha
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
                coord,
                radius=radius,
                alpha=1.0,
                lw=0.7,
                ec="black",
                fc="#FFC0CB",
                zorder=20,
            )
        else:
            circle = plt.Circle(
                coord,
                radius=radius,
                alpha=1.0,
                lw=0.7,
                ec="black",
                fc="lavender",
                zorder=20,
            )

        ax.add_patch(circle)
    for idx, polygon in enumerate(polygons):
        polygon = [Polygons.rotate_coords_by_45(coord, rotate) for coord in polygon]
        patch = plt.Polygon(
            polygon, zorder=0, lw=8, alpha=0.7, color=colors_of_polygon[idx], fill=False
        )
        ax.add_patch(patch)
    return ax


def visualize_ancilla(polygon_object, ax, polygons, s=0.6):

    N = 6
    rotate = True
    graph = Graph.init_without_edges(
        N, [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    )
    qbits = Qbits.init_qbits_from_dict(
        graph, dict(zip(graph.qubits, graph.qubits)), assign_to_core=False
    )
    qbits[(1, 4)].ancilla = True
    qbits[(1, 4)].core = True
    nodes_object = Nodes(qbits, place_qbits_in_lines=False)
    polygon_object = Polygons(nodes_object, scope_measure=True)
    polygon_object.polygons = polygons

    self = polygon_object
    zoom = 1
    figsize = (15, 15)
    core_corner = None
    check_ancilla_in_core: bool = True
    envelop_rect = None
    rotate: bool = False
    radius = 0.2

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")
    ax.axis("off")

    if envelop_rect is None:
        envelop_rect = self.nodes_object.qbits.envelop_rect(padding=0)
    ax.scatter(*list(zip(*envelop_rect)), color="grey", s=s, alpha=1)

    # color plaquettes
    for polygon in self.polygons_coords(
        self.nodes_object.qbits.qubit_to_coord_dict, self.polygons
    ):
        fill, facecolor, lw, alpha, edge_alpha = False, None, 0, 0, 0
        measure = self.scope_of_polygon(polygon)

        if measure == self.unit_square:
            fill, facecolor, lw, alpha, edge_alpha = True, "limegreen", 1, 0.5, 1
            polygon = Polygons.scale(polygon, radius)
        polygon = [Polygons.rotate_coords_by_45(p, rotate) for p in polygon]
        patch = plt.Polygon(
            polygon,
            zorder=0,
            fill=fill,
            lw=0,
            facecolor=facecolor,
            alpha=alpha,
        )
        edge = plt.Polygon(
            polygon, zorder=0, fill=False, lw=lw, edgecolor="black", alpha=edge_alpha
        )
        ax.add_patch(patch)
        ax.add_patch(edge)
    # color qbits
    for qbit in self.nodes_object.qbits:
        coord = Polygons.rotate_coords_by_45(qbit.coord, rotate)
        if qbit.qubit not in [qubit for polygon in polygons for qubit in polygon]:
            continue
        else:
            if qbit.core == False:
                circle = plt.Circle(
                    coord,
                    radius=radius,
                    alpha=1.0,
                    lw=0.7,
                    ec="black",
                    fc="antiquewhite",
                    zorder=20,
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
    rotate: bool = False,
    radius=0.2,
    corner_lw=10,
    color_of_grid_points="white",
    alpha_of_grid_points=0,
    padding=1,
    s=0.6,
    without_label=True,
    fontsize=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")
    ax.axis("off")

    if envelop_rect is None:
        envelop_rect = self.nodes_object.qbits.envelop_rect(padding=padding)
    ax.scatter(
        *list(zip(*envelop_rect)),
        color=color_of_grid_points,
        s=s,
        alpha=alpha_of_grid_points,
    )

    if core_corner is not None:
        (min_x, max_x), (min_y, max_y) = core_corner
        envelop = [
            Polygons.rotate_coords_by_45(p, rotate)
            for p in [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
        ]
        patch = plt.Polygon(
            envelop, zorder=10, fill=False, lw=corner_lw, edgecolor="black", alpha=0.5
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
        try:
            polygon = list(zip(*MultiPoint(polygon).convex_hull.exterior.xy))
        except:
            pass
        polygon = [Polygons.rotate_coords_by_45(p, rotate) for p in polygon]
        patch = plt.Polygon(
            polygon,
            zorder=10,
            fill=fill,
            lw=0,
            facecolor=facecolor,
            alpha=alpha,
        )
        edge = plt.Polygon(
            polygon, zorder=10, fill=False, lw=lw, edgecolor="black", alpha=edge_alpha
        )
        ax.add_patch(patch)
        ax.add_patch(edge)
    # color qbits
    for qbit in self.nodes_object.qbits:
        coord = Polygons.rotate_coords_by_45(qbit.coord, rotate)
        if fontsize is None:
            if len(str(qbit.qubit)) - 4 == 2:
                fontsize = 82 * radius
            if len(str(qbit.qubit)) - 4 == 3:
                fontsize = 70 * radius
            if len(str(qbit.qubit)) - 4 == 4:
                fontsize = 60 * radius
        label_text = ""
        if not without_label:
            label_text = r"{},{}".format(*qbit.qubit)
        label = ax.annotate(
            label_text,
            xy=coord,
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=20,
        )
        if qbit.core == False:
            circle = plt.Circle(
                coord,
                radius=radius,
                alpha=1.0,
                lw=0.7,
                ec="black",
                fc="antiquewhite",
                zorder=20,
            )
        if qbit.core == True:
            circle = plt.Circle(
                coord, radius=radius, alpha=1.0, lw=0.7, ec="black", fc="tan", zorder=20
            )
        if qbit.ancilla == True:
            circle = plt.Circle(
                coord,
                radius=radius,
                alpha=1.0,
                lw=0.7,
                ec="black",
                fc="lightcoral",
                zorder=20,
            )
            if check_ancilla_in_core:
                assert (
                    qbit.ancilla == True and qbit.core == True
                ), "ancilla is not part of the core ?!"
        ax.add_patch(circle)
    return ax


def draw_LHZ_from_spin_glass_(
    polygon_object, name_to_coord_dict, polygons, ffontsize=None, invert=False
):
    self = polygon_object
    ax = None
    zoom = 1
    figsize = (15, 15)
    core_corner = None
    check_ancilla_in_core: bool = True
    envelop_rect = None
    rotate: bool = True
    radius = 0.2

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")
    ax.axis("off")

    if envelop_rect is None:
        envelop_rect = self.nodes_object.qbits.envelop_rect()
    ax.scatter(*list(zip(*envelop_rect)), color="white", s=0.6, alpha=0)

    if core_corner is not None:
        (min_x, max_x), (min_y, max_y) = core_corner
        envelop = [
            Polygons.rotate_coords_by_45(p, rotate)
            for p in [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
        ]
        patch = plt.Polygon(
            envelop, zorder=10, fill=False, lw=10, edgecolor="black", alpha=0.5
        )
        ax.add_patch(patch)

    # color plaquettes
    for p, polygon in zip(
        self.polygons,
        self.polygons_coords(
            self.nodes_object.qbits.qubit_to_coord_dict, self.polygons
        ),
    ):

        fill, facecolor, lw, alpha, edge_alpha = False, None, 0, 0, 0
        if self.scope_measure:
            measure = self.scope_of_polygon(polygon)
        if not self.scope_measure:
            measure = self.moment_of_inertia(polygon)
        if measure == self.unit_square:
            fill, facecolor, lw, alpha, edge_alpha = True, "limegreen", 1, 0.1, 1
            polygon = Polygons.scale(polygon, radius)
            if sorted(p) in polygons:
                alpha = 0.5
        if measure == self.unit_triangle:
            fill, facecolor, lw, alpha, edge_alpha = True, "blue", 1, 0.1, 1
            if sorted(p) in polygons:
                alpha = 0.4
            polygon = Polygons.scale(polygon, radius)
        # if points are on a line, it is not a polygon anymore but a LineString object
        try:
            polygon = list(zip(*MultiPoint(polygon).convex_hull.exterior.xy))
        except:
            pass
        polygon = [Polygons.rotate_coords_by_45(p, rotate) for p in polygon]
        patch = plt.Polygon(
            polygon,
            zorder=0,
            fill=fill,
            lw=0,
            facecolor=facecolor,
            alpha=alpha,
        )
        edge = plt.Polygon(
            polygon, zorder=0, fill=False, lw=lw, edgecolor="black", alpha=edge_alpha
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
        if ffontsize is None:
            ffontsize = fontsize
        label = ax.annotate(
            r"{},{}".format(*qbit.qubit),
            xy=coord,
            ha="center",
            va="center",
            fontsize=ffontsize,
            zorder=20,
        )

        if invert:
            if sum([name_to_coord_dict[i].up for i in qbit.qubit]) % 2 == 0:
                circle = plt.Circle(
                    coord,
                    radius=radius,
                    alpha=1.0,
                    lw=0.7,
                    ec="black",
                    fc="lavender",
                    zorder=20,
                )
            else:
                circle = plt.Circle(
                    coord,
                    radius=radius,
                    alpha=1.0,
                    lw=0.7,
                    ec="black",
                    fc="#FFC0CB",
                    zorder=20,
                )
        else:
            if sum([name_to_coord_dict[i].up for i in qbit.qubit]) % 2 == 0:
                circle = plt.Circle(
                    coord,
                    radius=radius,
                    alpha=1.0,
                    lw=0.7,
                    ec="black",
                    fc="#FFC0CB",
                    zorder=20,
                )
            else:
                circle = plt.Circle(
                    coord,
                    radius=radius,
                    alpha=1.0,
                    lw=0.7,
                    ec="black",
                    fc="lavender",
                    zorder=20,
                )

        ax.add_patch(circle)

    return ax


def coords_from_corner(corner):
    (min_x, max_x), (min_y, max_y) = corner
    x, y = np.meshgrid(
        np.arange(min_x, max_x + 1),
        np.arange(min_y, max_y + 1),
    )
    return list(zip(x.flatten(), y.flatten()))


def decay_measure_per_qbit(self, qbits_of_interest):
    """
    gives an exponential decaying weight on each polygon,
    for each qbit. Decay is faster for larger decay_rates
    """
    polygons_of_interest = [
        [tuple(polygon) for polygon in qbit.polygons]
        for qbit in qbits_of_interest
        if not isinstance(qbit, tuple)
    ]
    qubit_to_coord_dict = self.polygon_object.nodes_object.qbits.qubit_to_coord_dict
    polygons_coords_of_interest = [
        Polygons.polygons_coords(qubit_to_coord_dict, polygons)
        for polygons in polygons_of_interest
    ]
    if self.polygon_object.scope_measure:
        qbits_measure = [
            (sorted(list(map(self.polygon_object.scope_of_polygon, polygon_coords))))
            for polygon_coords in polygons_coords_of_interest
        ]
    if not self.polygon_object.scope_measure:
        qbits_measure = [
            (sorted(list(map(self.polygon_object.moment_of_inertia, polygon_coords))))
            for polygon_coords in polygons_coords_of_interest
        ]
    measure = np.array(
        [
            1 * np.exp(-self.decay_rate * np.arange(len(qbit_measure)))
            for qbit_measure in qbits_measure
        ],
        dtype=object,
    )
    polygons_coords = self.polygon_object.polygons_coords(
        self.polygon_object.nodes_object.qbits.qubit_to_coord_dict,
        list(map(list, polygons_of_interest))[0],
    )
    return measure, polygons_coords


def MB_speed(v, m, T):
    """Maxwell-Boltzmann speed distribution for speeds"""
    kB = 1.38e-23
    return (
        (m / (2 * np.pi * kB * T)) ** 1.5
        * 4
        * np.pi
        * v ** 2
        * np.exp(-m * v ** 2 / (2 * kB * T))
    )


def visualize_(
    self,
    polygons,
    ax=None,
    zoom=1,
    figsize=(15, 15),
    core_corner=None,
    check_ancilla_in_core: bool = True,
    envelop_rect=None,
    rotate: bool = False,
    radius=0.2,
    without_label=True,
    fontsize=20,
):
    polygons = list(map(sorted, polygons))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")
    ax.axis("off")

    if envelop_rect is None:
        envelop_rect = self.nodes_object.qbits.envelop_rect()
    ax.scatter(*list(zip(*envelop_rect)), color="white", s=0.6, alpha=0)

    if core_corner is not None:
        (min_x, max_x), (min_y, max_y) = core_corner
        envelop = [
            Polygons.rotate_coords_by_45(p, rotate)
            for p in [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
        ]
        patch = plt.Polygon(
            envelop, zorder=10, fill=False, lw=10, edgecolor="black", alpha=0.5
        )
        ax.add_patch(patch)

    # color plaquettes
    for p, polygon in zip(
        self.polygons,
        self.polygons_coords(
            self.nodes_object.qbits.qubit_to_coord_dict, self.polygons
        ),
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
            fill, facecolor, lw, alpha, edge_alpha = True, "limegreen", 1, 0.6, 1
            if sorted(p) not in polygons:
                alpha = 0.3
            polygon = Polygons.scale(polygon, radius)
        if measure == self.unit_triangle:
            fill, facecolor, lw, alpha, edge_alpha = True, "blue", 1, 0.5, 1
            if sorted(p) not in polygons:
                alpha = 0.1
            polygon = Polygons.scale(polygon, radius)
        # if points are on a line, it is not a polygon anymore but a LineString object
        try:
            polygon = list(zip(*MultiPoint(polygon).convex_hull.exterior.xy))
        except:
            pass
        polygon = [Polygons.rotate_coords_by_45(p, rotate) for p in polygon]
        patch = plt.Polygon(
            polygon,
            zorder=0,
            fill=fill,
            lw=0,
            facecolor=facecolor,
            alpha=alpha,
        )
        edge = plt.Polygon(
            polygon, zorder=0, fill=False, lw=lw, edgecolor="black", alpha=edge_alpha
        )
        ax.add_patch(patch)
        ax.add_patch(edge)
    # color qbits
    for qbit in self.nodes_object.qbits:
        coord = Polygons.rotate_coords_by_45(qbit.coord, rotate)

        label_text = ""
        if not without_label:
            label_text = r"{},{}".format(*qbit.qubit)
        label = ax.annotate(
            label_text,
            xy=coord,
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=20,
        )
        if qbit.core == False:
            circle = plt.Circle(
                coord,
                radius=radius,
                alpha=1.0,
                lw=0.7,
                ec="black",
                fc="antiquewhite",
                zorder=20,
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


def plot_df(df, isin, x_axis, y_axis, ax, title, 
           color=["red", "blue", "lime", "black", "orange", "cyan", "magenta"]):
    df_ = (
        df.loc[df["number_of_setting"].isin(isin)]
        .groupby([x_axis, "number_of_setting"])
        .mean()
    )
    ax = (
        df_.reset_index()
        .pivot(x_axis, "number_of_setting", y_axis)
        .plot(
            ax=ax,
            marker="8",
            linestyle="--",
            lw=1,
            alpha=1,
            grid=False,
            markersize=10,
            markerfacecolor="w",
            markeredgewidth=1.5,
            color=color,
        )
    )
    return ax


def advanced_plot_df(
    df,
    name,
    title,
    setting_names,
    setting_numbers,
    legend_title,
    x_axis,
    y_axis,
    ax,
    subplot=False,
    compare=False,
    save=True,
    with_title=False,
    default_lhz=(11.3, 0.8),
    default_database=(8.3, 3.2),
    color=["red", "blue", "lime", "black", "orange", "cyan", "magenta"],
):
    min_x_axis = df[name][x_axis].min()
    max_x_axis = df[name][x_axis].max()

    plot_df(
        df[name], isin=setting_numbers, x_axis=x_axis, y_axis=y_axis, ax=ax, title=title, color=color
    )
    ax.axhline(1, linestyle="--", dashes=(7, 3), lw=1, color="grey")

    if "LHZ" in name:
        problem_folder = "lhz"
    else:
        problem_folder = "training_set"
    if compare:
        without_compilation(problem_folder).loc[:max_x_axis][y_axis].plot(
            ax=ax,
            marker="8",
            linestyle="--",
            lw=1,
            alpha=0.9,
            title=title,
            grid=False,
            markersize=10,
            markerfacecolor="w",
            markeredgewidth=1.5,
            color="grey",
        )
        if problem_folder == "training_set":
            ax.text(
                *default_database, "non-compiled", color="grey", fontsize=15, alpha=0.7
            )
        else:
            ax.text(*default_lhz, "non-compiled", color="grey", fontsize=15, alpha=0.7)
    if subplot:
        if len(setting_names) > 5:
            labelspacing = 0
        else:
            labelspacing = 0.5
        ax.legend(
            setting_names,
            title=legend_title,
            title_fontsize=15,
            borderpad=1,
            labelspacing=labelspacing,
            fontsize=14,
        )
    else:
        ax.legend(
            setting_names,
            title=legend_title,
            title_fontsize=15,
            borderpad=1,
            labelspacing=1.3,
            fontsize=14,
        )
    if with_title:
        ax.set_title(label=title, fontsize=15)
    else:
        ax.set_title(label="", fontsize=0)
    ax.yaxis.set_label_coords(-0.1, 0.5)

    ax.set_xlabel(x_axis, fontsize=18)
    # ax.set_xticks([i for i in range(min_x_axis, max_x_axis+1) if i%2==0])
    ax.set_yticks(np.arange(1, 4.2, 0.8))
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_yscale("log")
    #      ax.set_ylim(0, 10)
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())  # <---- Added

    if not subplot:
        ax.set_ylabel(r"#CNOTs $\frac{compiled}{lhz}$", rotation=90, fontsize=18)
    else:
        ax.set_xlabel("")
    if save:
        ax.figure.savefig(
            paths.plots / f"ThesisFigures/Plots/{title}_compare_{compare}.pdf",
            bbox_inches="tight",
        )

    return ax


def plot_settings_in_subplot(
    df,
    name,
    titles,
    all_setting_numbers_and_names,
    x_axis,
    y_axis,
    nx,
    ny,
    figsize=(15, 15),
    wspace=0.5,
    hspace=0.3,
    axis=None,
    fig=None,
    y_label_coord=(0.03, 0.5),
    x_label_coord=(0.5, 0.04),
    color=["red", "blue", "lime", "black", "orange", "cyan", "magenta"],
):
    if axis is None:
        fig, axs = plt.subplots(nx, ny, figsize=figsize)
    min_indices = []
    for idx, (setting_numbers_and_names, title) in enumerate(
        zip(all_setting_numbers_and_names, titles)
    ):
        try:
            setting_numbers, setting_names = setting_numbers_and_names
            if len(setting_names) == 2:
                setting_names, legend_title = setting_names
            else:
                setting_names = setting_names[0]
                legend_title = ""
            min_idx = (
                df[name]
                .loc[df[name]["number_of_setting"].isin(setting_numbers)]
                .groupby("number_of_setting")
                .mean()["CNOT_ratio"]
                .idxmin()
            )
            min_indices.append(min_idx)
            if nx == 1:
                if axis is None:
                    ax = axs[idx]
                else:
                    ax = axis
            else:
                ax = axs[idx // ny, idx % ny]
            advanced_plot_df(
                df,
                name,
                title,
                setting_names,
                setting_numbers,
                legend_title,
                x_axis,
                y_axis,
                ax,
                subplot=True,
                save=False,
                with_title=True,
                color=color,
            )
        except:
            print(setting_numbers_and_names)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if axis is None:
        fig.text(*x_label_coord, x_axis, ha="center", fontsize=20)
        fig.text(
            *y_label_coord, r"#CNOTs $\frac{compiled}{lhz}$", va="center", fontsize=20, rotation=90,
        )
        return axs, min_indices
    else:
        return None


without_compilation = (
    lambda problem_folder: pd.read_csv(
        paths.cwd
        / f"benchmark/plot_scripts/CNOT_ratio_of_{problem_folder}_before_compilation.csv"
    )
    .groupby("K")
    .mean()
)


def color_qubits(
    self,
    ax=None,
    zoom=1,
    figsize=(15, 15),
    core_corner=None,
    check_ancilla_in_core: bool = True,
    envelop_rect=None,
    rotate: bool = False,
    radius=0.2,
    qubit_colors=None,
    fontsize=20,
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")
    ax.axis("off")

    if envelop_rect is None:
        envelop_rect = self.nodes_object.qbits.envelop_rect()
    ax.scatter(*list(zip(*envelop_rect)), color="white", s=0.6, alpha=0)

    if core_corner is not None:
        (min_x, max_x), (min_y, max_y) = core_corner
        envelop = [
            Polygons.rotate_coords_by_45(p, rotate)
            for p in [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
        ]
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
        try:
            polygon = list(zip(*MultiPoint(polygon).convex_hull.exterior.xy))
        except:
            pass
        polygon = [Polygons.rotate_coords_by_45(p, rotate) for p in polygon]
        patch = plt.Polygon(
            polygon,
            zorder=0,
            fill=fill,
            lw=0,
            facecolor=facecolor,
            alpha=alpha,
        )
        edge = plt.Polygon(
            polygon, zorder=0, fill=False, lw=lw, edgecolor="black", alpha=edge_alpha
        )
        ax.add_patch(patch)
        ax.add_patch(edge)
    # color qbits
    for qbit in self.nodes_object.qbits:
        coord = Polygons.rotate_coords_by_45(qbit.coord, rotate)

        label = ax.annotate(
            r"{},{}".format(*qbit.qubit),
            xy=coord,
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=20,
        )
        circle = plt.Circle(
            coord, radius=radius, alpha=1, lw=0.7, ec="black", fc="white", zorder=19
        )
        ax.add_patch(circle)

        circle = plt.Circle(
            coord,
            radius=radius,
            alpha=qbit.alpha,
            lw=0.7,
            ec="black",
            fc=qbit.color,
            zorder=20,
        )
        ax.add_patch(circle)
    return ax
