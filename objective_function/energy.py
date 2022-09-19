import numpy as np
import networkx as nx
from CompilerQC import Polygons, paths
import pickle
from shapely.geometry import LineString
import shapely
from shapely.ops import unary_union
from shapely.geometry import CAP_STYLE, JOIN_STYLE, MultiPoint, Polygon
import math
from shapely.validation import make_valid
from itertools import combinations


class Energy(Polygons):
    def __init__(
        self,
        polygon_object,
        scaling_for_plaq3: float = 0,
        scaling_for_plaq4: float = 0,
        scaling_model: str = None,
        basic_energy: bool = True,
        decay_weight: bool = False,
        decay_rate: float = 1,
        subset_weight: bool = False,
        sparse_density_penalty: bool = False,
        sparse_density_factor: float = 1,
        line: bool = False,
        line_exponent: float = 1,
        bad_line_penalty: float = 0,
        line_factor: float = 1,
        low_noplaqs_penalty: bool = False,
        low_noplaqs_factor: float = 1,
        all_constraints: bool=False,
    ):
        """
        if a plaquette has been found, assign the scaling_for_plaq to it.
        If a model ['LHZ', 'MLP' or 'INIT'] is assigned, scale them according to
        this model
        """
        self.polygon_object = polygon_object
        self.all_constraints = all_constraints
        # polygon weights
        self.basic_energy = basic_energy

        self.decay_weight = decay_weight
        self.decay_rate = decay_rate
        
        self.subset_weight = subset_weight
        
        self.sparse_density_penalty = sparse_density_penalty
        self.sparse_density_factor = sparse_density_factor
        
        self.line = line
        self.line_exponent = line_exponent
        self.bad_line_penalty = bad_line_penalty
        self.line_factor = line_factor
        
        self.low_noplaqs_penalty = low_noplaqs_penalty
        self.low_noplaqs_factor = low_noplaqs_factor
        
        # scaling for plaquettes
        self.scaling_for_plaq3 = scaling_for_plaq3
        self.scaling_for_plaq4 = scaling_for_plaq4
        self.scaling_model = scaling_model
        self.set_scaling_from_model()



    def set_scaling_from_model(self):
        """
        load and set plaquette scaling from given model
        INIT: estimate energy by energy of initial configuration
        """
        predicted_energy = None
        if self.scaling_model is not None:
            assert self.scaling_model in [
                "MLP",
                "INIT",
                "LHZ",
            ], "selected model does not exist, choose from 'MLP', 'INIT' or 'LHZ'"

            assert self.polygon_object.exponent in [0.5, 1, 2, 3, 4, 5, 6] or self.scaling_model=="INIT", "their is no valid LHZ or MLP scaling for this exponent, if you want to keep this exponent, choose INIT as scaling_model"
            if self.polygon_object.scope_measure: 
                measure = "Scope"
            else:
                measure = "MoI"

            if self.scaling_model == "MLP":
                loaded_model = pickle.load(
                    open(paths.energy_scalings / f"MLP_{measure}_{self.polygon_object.exponent}.sav", "rb")  
                )
                predicted_energy = loaded_model.predict(
                    [
                        [
                            self.polygon_object.nodes_object.qbits.graph.N,
                            self.polygon_object.nodes_object.qbits.graph.K,
                            self.polygon_object.nodes_object.qbits.graph.C,
                            self.polygon_object.nodes_object.qbits.graph.number_of_3_cycles,
                            self.polygon_object.nodes_object.qbits.graph.number_of_4_cycles,
                        ]
                    ]
                )[0]

            if self.scaling_model == "INIT":
                predicted_energy, _ = self.__call__(self.polygon_object.nodes_object.qbits)

            if self.scaling_model == "LHZ":
                poly_coeffs = np.load(paths.energy_scalings / f"LHZ_{measure}_{self.polygon_object.exponent}.npy")
                poly_function = np.poly1d(poly_coeffs)
                predicted_energy = poly_function(
                    self.polygon_object.nodes_object.qbits.graph.C
                )

            scaling_for_plaq3 = scaling_for_plaq4 = (
                predicted_energy / self.polygon_object.nodes_object.qbits.graph.C
            )
            self.scaling_for_plaq3, self.scaling_for_plaq4 = (
                scaling_for_plaq3,
                scaling_for_plaq4,
            )
            # update unit scope/MoI if the function has been updated
            self.polygon_object.set_unit_measure()
            
        self.set_scaling_for_energy_terms()
        return predicted_energy

    def set_scaling_for_energy_terms(self):
        """the different energy terms contribute with different orders to the total energy,
        by calling this function, they will be scaled to the same order, 
        with the factors (line_factor, ...) you can vary the strength.
        Therefore, the blank energy (measured_energy) without any additional energy term is calculated, 
        then the additional (e.g. line) term is computed with a factor of 1, the ratio 
        measured_energy / line_energy is the new factor for the line term"""
        if not self.basic_energy:
            return
        (line_factor,
         sparse_density_factor, low_noplaqs_factor) = (self.line_factor,
                                                       self.sparse_density_factor, self.low_noplaqs_factor)
        (self.line_factor,
         self.sparse_density_factor, self.low_noplaqs_factor) = (0 , 0 , 0)

        measure_energy = int(self.__call__(self.polygon_object.nodes_object.qbits)[0])
        line_energy_value, sparse_density_penalty, low_noplaqs_penalty = measure_energy, measure_energy, measure_energy
        if self.line:
            line_energy_value = int(self.line_energy(self.polygon_object.nodes_object.qbits))
        if self.sparse_density_penalty:
            sparse_density_penalty = int(sum(self.penalty_for_sparse_plaquette_density()))
        if self.low_noplaqs_penalty:
            low_noplaqs_penalty = int(sum(self.penalty_for_low_number_of_plaquettes()))
        self.line_factor = line_factor * measure_energy / (1 + line_energy_value)
        self.sparse_density_factor = sparse_density_factor * measure_energy / (1 + sparse_density_penalty)
        self.low_noplaqs_factor = low_noplaqs_factor * measure_energy / (1 + low_noplaqs_penalty)

    def scopes_of_polygons(self):
        """return array of the scopes of all changed polygons,"""
        return np.array(
            list(map(self.polygon_object.scope_of_polygon, self.polygons_coords_of_interest))
        )
    
    def moments_of_inertia_of_polygons(self):
        """return array of moment of inertia of all changed polygons """
        return np.array(list(map(self.polygon_object.moment_of_inertia, self.polygons_coords_of_interest)))

    def measure_of_polygons_for_analysis(self):
        """
        return four arrays of measure:
        nonplaqs3, nonplaqs4, plaqs3, plaqs4
        """
        # consider all polygons
        self.polygons_coords_of_interest = self.polygon_object.polygons_coords(
            self.polygon_object.nodes_object.qbits.qubit_to_coord_dict,
            self.polygon_object.polygons,
        )
        if self.polygon_object.scope_measure:
            measure = np.array(
                list(map(self.polygon_object.scope_of_polygon, self.polygons_coords_of_interest))
            )
        if not self.polygon_object.scope_measure:
            measure = np.array(
                list(map(self.polygon_object.moment_of_inertia, self.polygons_coords_of_interest))
            )           
        polygon_coords_lengths = np.array(
            list(map(len, self.polygons_coords_of_interest))
        )
        nonplaqs3 = measure[
            np.logical_and(
                polygon_coords_lengths == 3,
                measure != self.polygon_object.unit_triangle,
            )
        ]
        nonplaqs4 = measure[
            np.logical_and(
                polygon_coords_lengths == 4,
                measure != self.polygon_object.unit_square,
            )
        ]
        plaqs3 = measure[
            np.logical_and(
                polygon_coords_lengths == 3,
                measure == self.polygon_object.unit_triangle,
            )
        ]
        plaqs4 = measure[
            np.logical_and(
                polygon_coords_lengths == 4,
                measure == self.polygon_object.unit_square,
            )
        ]
        return nonplaqs3, nonplaqs4, plaqs3, plaqs4

    def scaled_measure(self, measure):
        """
        input: array of scopes/MoIs of polygons
        return: array of scopes/MoIs of polygons, if a polygon is a plaquette,
        the scaling_for_plaq is substracted
        """
        measure[
            measure == self.polygon_object.unit_triangle
        ] -= self.scaling_for_plaq3
        measure[
            measure == self.polygon_object.unit_square
        ] -= self.scaling_for_plaq4
        return measure    

    def coords_of_changed_polygons(self, polygons_of_interest):
        """
        return unique polygons coords which changed during the move
        """
        qubit_to_coord_dict = self.polygon_object.nodes_object.qbits.qubit_to_coord_dict
        return Polygons.polygons_coords(qubit_to_coord_dict, polygons_of_interest)

    def changed_polygons(self, qbits_of_interest: list):
        """
        return unique polygons which changed during the move
        """
        # all polygons which changed, as tuples not as lists (-> hashable)
        changed_polygons = [
            tuple(polygon)
            for qbit in qbits_of_interest
            if not isinstance(qbit, tuple)
            for polygon in qbit.polygons
        ]
        # remove duplicates (only unique polygons)
        return [list(k) for k in set(changed_polygons)]

    def subset_weights(self, polygons_of_interest):
        """
        return list of weights for all polygons scopes/MoIs
        give the polygon a weight if it has no qbits, which
        are in the diagonal of a four plaquette
        note: checked
        """

        found_plaqs = self.polygon_object.nodes_object.qbits.found_plaqs()
        forbidden_edges = [
            ((p[0], p[3]), (p[1], p[2])) for p in found_plaqs if len(p) == 4
        ]
        forbidden_edges = [set(i) for j in forbidden_edges for i in j]
        weights = np.array(
            [
                any([edge.issubset(polygon) for edge in forbidden_edges])
                if not (set(polygon) in map(set, found_plaqs))
                else False
                for polygon in polygons_of_interest
            ]
        )

        return 1 - weights.astype(int)

    def penalty_for_low_number_of_plaquettes(self):
        """
        there is a penalty
        if qbit is not involved in many plaquettes
        """
        # update plaquettes of each qbit
        self.polygon_object.set_plaquettes_of_qbits()
        # 3.5 and not 4 since then a small weight for all qbits is left (also for those which are already in 4 plaquettes)
        return [
            3.5 - len(qbit.plaquettes)
            for qbit in self.polygon_object.nodes_object.qbits.shell_qbits
        ]

    def penalty_for_sparse_plaquette_density(self):
        """
        there is a penalty
        if a shell qbit has a lot of neighbours,
        but is involved only in few plaquettes.
        note: checked, for LHZ negative value
        """
        # update plaquettes of each qbit
        self.polygon_object.set_plaquettes_of_qbits()
        self.polygon_object.set_numbers_of_qbits_neighbours()

        # if a qbit is involved in n plaquettes, it has in average qbits_per_plaq[n] neighbours which build these plaquettes
        qbits_per_plaq = [0, 2.5, 5, 6.5, 8]

        # the number of neighbour qbits (8-count(nan)) minus the avg. number of qbits which are in plaquettes
        return [
            np.abs(
                qbit.number_of_qbit_neighbours - qbits_per_plaq[len(qbit.plaquettes)]
            )
            / 8
            for qbit in self.polygon_object.nodes_object.qbits.shell_qbits
        ]

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
                self.scaled_measure(np.array(qbit_measure))
                * np.exp(-self.decay_rate * np.arange(len(qbit_measure)))
                for qbit_measure in qbits_measure
            ],
            dtype=object,
        )
        distances_to_plaquette = np.hstack(measure.flatten()).sum()
        return distances_to_plaquette

    # TODO: maybe make a multiplicator penalty instead of addition
    def line_energy(self, qbits_of_interest):
        """return the lengths to the lines corresponding to the nodes involved in
        qbits of interest. If a line ends without connection to the outerspace (so if it has only
        occupied neighbours), add a bad_line_penalty"""
        # update since it is needed to determine start node in polygon_object.line_to_node()
        self.polygon_object.set_numbers_of_qbits_neighbours()
        # update since it is needed to check if begin or end of line is bad
        self.polygon_object.set_plaquettes_of_qbits()
        nodes_of_interest = set(
            [
                node
                for qbit in qbits_of_interest
                if not isinstance(qbit, tuple)
                for node in qbit.qubit
            ]
        )
        line_energy_value = 0
        for node in nodes_of_interest:
            qbits_path, distances = self.polygon_object.line_to_node(node)
            distances = np.power(np.array(distances), self.line_exponent)
            line_energy_value += (distances).sum()
            if self.bad_line_penalty:
                # check if line is self intersecting, if so add penalty
                if not LineString(([qbit.coord for qbit in qbits_path])).is_simple:
                    line_energy_value += self.bad_line_penalty                
                # check if line is ending or starting bad, if so add penalty
                start_qbit, end_qbit = qbits_path[0], qbits_path[-1]
                number_of_free_neighbours_of_start = (
                    self.polygon_object.nodes_object.qbits.neighbours(
                        start_qbit.coord
                    ).count(np.nan)
                )
                number_of_free_neighbours_of_end = (
                    self.polygon_object.nodes_object.qbits.neighbours(end_qbit.coord).count(
                        np.nan
                    )
                )
                if (
                    number_of_free_neighbours_of_start == 0
                    and len(start_qbit.plaquettes) < 2
                ):
                    line_energy_value += self.bad_line_penalty
                if number_of_free_neighbours_of_end == 0 and len(end_qbit.plaquettes) < 2:
                    line_energy_value += self.bad_line_penalty
        return line_energy_value

    def consider_all_constraints(self):
        """
        weight all constraints, even those which are fulfilled implict
        """
        plaquettes = [([self.polygon_object.nodes_object.qbits[qubit].coord for qubit in plaquette]) for plaquette in self.polygon_object.nodes_object.qbits.found_plaqs()]
        plaquettes_ = [(Polygon(plaquette).envelope) if len(plaquette)==4 else Polygon(plaquette) for plaquette in plaquettes]
        union = unary_union(plaquettes_)

        # the case where two triangle plaquettes from a rectengular four constraint is not covered yet
        triangle_plaquettes = [plaq for plaq in plaquettes if len(plaq)==3]
        four_constraints = []
        for combi in list(combinations(triangle_plaquettes, 2)):
            constraint = set(combi[0] + combi[1]) - set(combi[0]).intersection(combi[1])
            if len(constraint) == 4:
                four_constraints.append(constraint)
                
        constraints, ps = [], []
        for polygon_coord in self.polygon_object.polygons_coords(qubit_to_coord_dict=self.polygon_object.nodes_object.qbits.qubit_to_coord_dict, polygons=self.polygon_object.polygons):
            p = Polygon(polygon_coord)
            if not p.is_valid:
                p = make_valid(p)
            if math.isclose(union.intersection(p).area, p.area):
                constraints.append(-self.scaling_for_plaq4 / self.polygon_object.scope_of_polygon(polygon_coord) ** 2)
                ps.append(polygon_coord)
            elif set(polygon_coord) in four_constraints:
                constraints.append(-self.scaling_for_plaq4 / self.polygon_object.scope_of_polygon(polygon_coord) ** 2)
                ps.append(polygon_coord)
            else:
                constraints.append(self.polygon_object.scope_of_polygon(polygon_coord))
                
        self.constraints = constraints
        self.ps = ps
        
        return np.array(constraints), len(plaquettes)

    # insert in __call__:
    
    def __call__(self, qbits_of_interest):
        """
        return the energy and number of plaquettes of the polygons belonging to
        qbits of interest
        """
        if self.all_constraints:
            energy_to_return, number_of_plaquettes = self.consider_all_constraints()
            return energy_to_return.sum(), number_of_plaquettes

        polygons_of_interest = self.changed_polygons(qbits_of_interest)
        self.polygons_coords_of_interest = self.coords_of_changed_polygons(
            polygons_of_interest
        )
        if self.polygon_object.scope_measure:
            measure = self.scopes_of_polygons() 
        if not self.polygon_object.scope_measure:
            measure = self.moments_of_inertia_of_polygons()
            
        number_of_plaquettes = len(
            measure[
                np.logical_or(
                    measure == self.polygon_object.unit_triangle,
                    measure == self.polygon_object.unit_square,
                )
            ]
        )

        distances_to_plaquette = self.scaled_measure(measure)

        energy_to_return = 0

        if self.decay_weight:
            distances_to_plaquette = self.decay_measure_per_qbit(qbits_of_interest)
            # decay is not working with terms which use the distances_to_plaquette array
            self.subset_weight = False

        # remove polygons with edges which are already in the diagonal of a 4 plaquette
        if self.subset_weight:
            weights = self.subset_weights(polygons_of_interest)
            distances_to_plaquette = np.dot(distances_to_plaquette, weights)

        if self.line:
            line_energy_value = self.line_energy(qbits_of_interest)
            energy_to_return += self.line_factor * line_energy_value

        # if a qbit is surrounded by qbits but has no/few plaquettes, penalize it
        if self.sparse_density_penalty:
            energy_to_return += self.sparse_density_factor * sum(
                self.penalty_for_sparse_plaquette_density()
            )
          
        if self.low_noplaqs_penalty:
            energy_to_return += self.low_noplaqs_factor * sum(
                self.penalty_for_low_number_of_plaquettes()
            )

        energy_to_return += int(self.basic_energy) * round(distances_to_plaquette.sum(), 5)

        return energy_to_return, number_of_plaquettes


class Energy_core(Energy):
    def __init__(
        self,
        polygon_object,
        scaling_for_plaq3: float = 0,
        scaling_for_plaq4: float = 0,
        scaling_model: str = None,
    ):
        self.polygon_object = polygon_object
        
        # scaling for plaquettes
        self.scaling_for_plaq3 = scaling_for_plaq3
        self.scaling_for_plaq4 = scaling_for_plaq4
        if scaling_model is not None:
            self.set_scaling_from_model(scaling_model)

        # energy terms
        self.only_largest_core = False
        self.only_rectangulars = False
        self.only_number_of_plaquettes = False

        # search for largest conncected core
        self.only_squares_in_core = False
        

    def rectengular_energy(self, qbits_of_interest):
        """
        consider only scopes of rectengular shape
        note: to find a minimal energy configuration,
        it will simply find a config with as least rectengulars as possible
        """
        polygons_of_interest = self.changed_polygons(qbits_of_interest)
        self.polygons_coords_of_interest = self.coords_of_changed_polygons(
            polygons_of_interest
        )
        self.polygons_coords_of_interest = [
            polygon
            for polygon in self.polygons_coords_of_interest
            if Polygons.is_rectengular(polygon)
        ]

    def qbits_in_max_core(self, K):
        """return the number of qbits
        in the maximum connected core
        if only_squares_in_core is True,
        triangulars are not considered
        by choosing the largest core
        if renaming is given in the graph, qbits are renamed according to graph.renaming: {old_node:new_node}
        since graph.K in the core search may be different from the graph.K in the unmodified graph, we hand over the original K"""
        G = nx.Graph()
        plaquettes = self.polygon_object.nodes_object.qbits.found_plaqs()
        if self.only_squares_in_core:
            plaquettes = [plaq for plaq in plaquettes if len(plaq) == 4]
        for l in plaquettes:
            nx.add_path(G, l)
        list_of_cores = list(nx.connected_components(G))
        if list_of_cores == []:
            return dict(), ((0, 0), (0, 0)), {}
        max_core_qbits = max(list_of_cores, key=lambda x: len(x))

        qubit_coord_dict = {
            tuple(
                sorted(self.polygon_object.nodes_object.qbits[qbit].qubit)
            ): self.polygon_object.nodes_object.qbits[qbit].coord
            for qbit in max_core_qbits
        }

        # move core to center of later initialization
        qubit_coord_dict = Polygons.move_to_center(qubit_coord_dict, K)
        # coords of envelop of core
        core_coords = list(qubit_coord_dict.values())
        corner = Polygons.corner_of_coords(core_coords)

        # ancillas in core
        ancillas_in_core = {
            qbit.qubit: qubit_coord_dict[qbit.qubit]
            for qbit in self.polygon_object.nodes_object.qbits
            if qbit.ancilla == True and qbit.qubit in qubit_coord_dict
        }
        # rename qubits
        renaming = self.polygon_object.nodes_object.qbits.graph.renaming
        if renaming is not None:
            qubit_coord_dict = {
                (renaming[qubit[0]], renaming[qubit[1]]): coord
                for qubit, coord in qubit_coord_dict.items()
            }
            ancillas_in_core = {
                (renaming[qubit[0]], renaming[qubit[1]]): coord
                for qubit, coord in ancillas_in_core.items()
            }

        return qubit_coord_dict, corner, ancillas_in_core

    def __call__(self, qbits_of_interest):
        """
        return the energy and number of plaquettes of the polygons belonging to
        qbits of interest
        """

        if self.only_rectangulars:
            self.rectengular_energy(qbits_of_interest)
        else:
            polygons_of_interest = self.changed_polygons(qbits_of_interest)
            self.polygons_coords_of_interest = self.coords_of_changed_polygons(
                polygons_of_interest
            )

        scopes = self.scopes_of_polygons()
        number_of_plaquettes = len(
            scopes[
                np.logical_or(
                    scopes == self.polygon_object.unit_triangle,
                    scopes == self.polygon_object.unit_square,
                )
            ]
        )

        if self.only_number_of_plaquettes:
            return number_of_plaquettes, number_of_plaquettes

        distances_to_plaquette = self.scaled_measure(scopes)

        if self.spring_energy:
            return round((distances_to_plaquette ** 2).sum(), 5), number_of_plaquettes
        else:
            return round((distances_to_plaquette).sum(), 5), number_of_plaquettes
