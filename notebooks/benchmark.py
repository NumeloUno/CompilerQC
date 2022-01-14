import numpy as np
import random
from CompilerQC import Graph
from CompilerQC import core
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint
from CompilerQC import Polygons
from CompilerQC import Energy
from CompilerQC import MC
from scipy.special import binom


def evaluate(polygon_object, n_steps, temperature, terms, operations):
    mc = MC(polygon_object, temperature=1, UV=[U, V])
    mc.terms = terms

    for i in range(n_steps):
        mc.update_temperature(temperature)
        for operation, n in operations:
            mc.apply(operation, n)
    return polygon_object.C - mc.energy.distance_to_plaquette().count(0)


def get_success_rates(polygon_object):
    n_steps = [
        100,
        # 500,
        # 1000
    ]
    temps = [
        0.01,
        1,
        10,
        # 100
    ]
    terms = [
        [1, 0, 1],
        [0, 1, 0],
        # [0,0,1]
    ]
    steps = 3
    operations = [
        # [('contract', steps)],
        # [('contract', steps),('swap_lines_in_core', steps)],
        [("contract", steps), ("swap_lines_in_core", steps), ("grow_core", steps)],
        # [('contract', steps),('swap_lines_in_core', steps),('grow_core', steps),('swap', steps)],
        # [('grow_core', steps)],
        [("grow_core", steps), ("swap_lines_in_core", steps)],
        # [('grow_core', steps),('swap_lines_in_core', steps),('contract', steps)],
        # [('grow_core', steps),('swap_lines_in_core', steps),('contract', steps),('swap', steps)],
        # [('grow_core', steps),('swap', steps),('grow_core', steps)],
    ]
    success_rates = []
    for n_step in n_steps:
        print(n_step)
        for temp in temps:
            print(temp)
            for term in terms:
                print(term)
                for operation in operations:
                    print(operation)
                    success_rate = []
                    for i in range(100):
                        success = evaluate(
                            polygon_object, n_step, temp, term, operation
                        )
                        success_rate.append(success)
                    success_rates.append(success_rate)
    return success_rates


for N in range(4, 8):

    graph = Graph.fully(N)

    nn = core.largest_complete_bipartite_graph(graph)
    K_nn = core.complete_bipartite_graph(*nn)
    U, V = core.parts_of_complete_bipartite_graph(graph.to_nx_graph(), K_nn)
    core_qbits, core_coords = core.qbits_and_coords_of_core(U, V)

    polygon_object = Polygons(graph, core_qbits)
    polygon_object.update_qbits_coords(core_qbits, core_coords)
    succes_rate = get_success_rates(polygon_object)
    np.save("1success_rates_for_" + str(N) + ".npy", np.array(succes_rate))
