import numpy as np
import networkx as nx
from CompilerQC import Polygons, paths
import pickle

class Energy(Polygons):
    def __init__(
        self,
        polygon_object,
        scaling_for_plaq3: float = 0,
        scaling_for_plaq4: float = 0,
        scaling_model: str=None,
    ):
        """
        if a plaquette has been found, assign the scaling_for_plaq to it.
        If a model ['LHZ', 'MLP' or 'maxC'] is assigned, scale them according to 
        this model
        """
        self.polygon_object = polygon_object

        # scaling for plaquettes
        if scaling_model is not None:
            self.set_scaling_from_model(scaling_model)
        self.scaling_for_plaq3 = scaling_for_plaq3
        self.scaling_for_plaq4 = scaling_for_plaq4
        # polygon weights
        
        self.decay_weight = False
        self.decay_rate = 1
        self.subset_weight = False
        self.square_scope = False
        self.sparse_density_penalty = False
        self.sparse_density_factor = 0
        self.line = False
        self.line_exponent = 1
        self.bad_line_penalty = 100  
                
    def set_scaling_from_model(self, scaling_model:str):
        """load and set plaquette scaling from given model"""
        assert scaling_model in ['MLP', 'maxC', 'LHZ'], "selected model does not exist, choose from ['MLP', 'maxC', 'LHZ']"
        
        if scaling_model == 'MLP':
            loaded_model = pickle.load(open(paths.energy_scalings / 'MLPregr_model.sav', 'rb'))
            predicted_energy = loaded_model.predict([[self.polygon_object.nodes_object.qbits.graph.N, self.polygon_object.nodes_object.qbits.graph.K, self.polygon_object.nodes_object.qbits.graph.C, self.polygon_object.nodes_object.qbits.graph.number_of_3_cycles, self.polygon_object.nodes_object.qbits.graph.number_of_4_cycles]])[0]

        if scaling_model == 'maxC':
            poly_coeffs = np.load(paths.energy_scalings / 'energy_max_C_fit.npy')
            poly_function = np.poly1d(poly_coeffs)
            predicted_energy = poly_function(self.polygon_object.nodes_object.qbits.graph.C)

        if scaling_model == 'LHZ':
            poly_coeffs = np.load(paths.energy_scalings / 'LHZ_energy_C_fit.npy')
            poly_function = np.poly1d(poly_coeffs)
            predicted_energy = poly_function(self.polygon_object.nodes_object.qbits.graph.C)

        scaling_for_plaq3 = scaling_for_plaq4 = predicted_energy / self.polygon_object.nodes_object.qbits.graph.C  
        self.scaling_for_plaq3, self.scaling_for_plaq4 = scaling_for_plaq3, scaling_for_plaq4

    def scopes_of_polygons(self):
        """return array of the scopes of all changed polygons,"""
        return np.array(
            list(map(self.scope_of_polygon, self.polygons_coords_of_interest))
        )

    def scopes_of_polygons_for_analysis(self):
        """
        return four arrays of scopes:
        nonplaqs3, nonplaqs4, plaqs3, plaqs4
        """
        # consider all polygons
        self.polygons_coords_of_interest = self.polygon_object.polygons_coords(
            self.polygon_object.nodes_object.qbits.qubit_to_coord_dict, self.polygon_object.polygons
        )

        scopes = np.array(
            list(map(self.scope_of_polygon, self.polygons_coords_of_interest))
        )
        polygon_coords_lengths = np.array(
            list(map(len, self.polygons_coords_of_interest))
        )
        nonplaqs3 = scopes[
            np.logical_and(
                polygon_coords_lengths == 3,
                scopes != self.polygon_object.unit_triangle_scope,
            )
        ]
        nonplaqs4 = scopes[
            np.logical_and(
                polygon_coords_lengths == 4,
                scopes != self.polygon_object.unit_square_scope,
            )
        ]
        plaqs3 = scopes[
            np.logical_and(
                polygon_coords_lengths == 3,
                scopes == self.polygon_object.unit_triangle_scope,
            )
        ]
        plaqs4 = scopes[
            np.logical_and(
                polygon_coords_lengths == 4,
                scopes == self.polygon_object.unit_square_scope,
            )
        ]
        return nonplaqs3, nonplaqs4, plaqs3, plaqs4

    def scaled_distance_to_plaquette(self, scopes):
        """
        input: array of scopes of polygons
        return: array of scopes of polygons, if a polygon is a plaquette,
        the scaling_for_plaq is substracted
        """
        scopes[
            scopes == self.polygon_object.unit_triangle_scope
        ] -= self.scaling_for_plaq3
        scopes[
            scopes == self.polygon_object.unit_square_scope
        ] -= self.scaling_for_plaq4
        return scopes

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
        return list of weights for all polygons scopes
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
            qbit.number_of_qbit_neighbours - qbits_per_plaq[len(qbit.plaquettes)]
            for qbit in self.polygon_object.nodes_object.qbits.shell_qbits
        ]

    def decay_scopes_per_qbit(self, qbits_of_interest):
        """
        gives an exponential decaying weight on each polygon,
        for each qbit. Decay is faster for larger decay_rates
        """
        # scope
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
        qbits_scopes = [
            (sorted(list(map(self.scope_of_polygon, polygon_coords))))
            for polygon_coords in polygons_coords_of_interest
        ]

        scopes = np.array(
            [
                self.scaled_distance_to_plaquette(np.array(qbit_scopes))
                * np.exp(-self.decay_rate * np.arange(len(qbit_scopes)))
                for qbit_scopes in qbits_scopes
            ]
        )

        distances_to_plaquette = np.hstack(scopes.flatten()).sum()

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
        nodes_of_interest = set([node for qbit in qbits_of_interest if not isinstance(qbit, tuple) for node in qbit.qubit])
        line_energy_value = 0
        for node in nodes_of_interest:
            qbits_path, distances = self.polygon_object.line_to_node(node)
            distances = np.power(np.array(distances), self.line_exponent)
            line_energy_value += (distances).sum()
            #check if line is ending or starting bad, if so add penalty
            start_qbit, end_qbit = qbits_path[0], qbits_path[-1]
            number_of_free_neighbours_of_start = self.polygon_object.nodes_object.qbits.neighbours(start_qbit.coord).count(np.nan)
            number_of_free_neighbours_of_end = self.polygon_object.nodes_object.qbits.neighbours(end_qbit.coord).count(np.nan)
            if number_of_free_neighbours_of_start == 0 and len(start_qbit.plaquettes) < 2:
                line_energy_value += self.bad_line_penalty
            if number_of_free_neighbours_of_end == 0 and len(end_qbit.plaquettes) < 2:
                line_energy_value += self.bad_line_penalty                
        return line_energy_value

    def __call__(self, qbits_of_interest):
        """
        return the energy and number of plaquettes of the polygons belonging to
        qbits of interest
        """
        
        polygons_of_interest = self.changed_polygons(qbits_of_interest)
        self.polygons_coords_of_interest = self.coords_of_changed_polygons(
            polygons_of_interest
        )
        scopes = self.scopes_of_polygons()
        number_of_plaquettes = len(
            scopes[
                np.logical_or(
                    scopes == self.polygon_object.unit_triangle_scope,
                    scopes == self.polygon_object.unit_square_scope,
                )
            ]
        )

        distances_to_plaquette = self.scaled_distance_to_plaquette(scopes)   
        
        energy_to_return = 0
        
        if self.decay_weight:
            distances_to_plaquette = self.decay_scopes_per_qbit(qbits_of_interest)
            # decay is not working with terms which use the distances_to_plaquette array
            self.square_scope = self.subset_weight = False
        
        if self.square_scope:
            distances_to_plaquette = (distances_to_plaquette ** 2)
            
        #remove polygons with edges which are already in the diagonal of a 4 plaquette
        if self.subset_weight:
            weights = self.subset_weights(polygons_of_interest)
            distances_to_plaquette = np.dot(distances_to_plaquette, weights)
           
        if self.line:
            line_energy_value = self.line_energy(qbits_of_interest)
            energy_to_return += line_energy_value
        
        # if a qbit is surrounded by qbits but has no/few plaquettes, penalize it
        if self.sparse_density_penalty:
            energy_to_return += self.sparse_density_factor * sum(
                self.penalty_for_sparse_plaquette_density()
            )
            
        energy_to_return += round(distances_to_plaquette.sum(), 5) 
        
        return energy_to_return, number_of_plaquettes

class Energy_core(Energy):
    def __init__(
        self,
        polygon_object,
        scaling_for_plaq3: float = 0,
        scaling_for_plaq4: float = 0,
        model_name: str=None,
    ):
        self.polygon_object = polygon_object

        # scaling for plaquettes
        if model_name is not None:
            self.set_scaling_from_model(model_name)
        self.scaling_for_plaq3 = scaling_for_plaq3
        self.scaling_for_plaq4 = scaling_for_plaq4

        # energy terms
        self.only_largest_core = False
        self.spring_energy = False
        self.only_rectangulars = False
        self.only_number_of_plaquettes = False
        self.size_of_max_core = False

        # search for largest conncected core
        self.only_squares_in_core = False

    @staticmethod
    def is_rectengular(polygon_coord):
        """
        check if a given polygon forms a rectengular
        """
        return sum([len(set(coords)) == 2 for coords in list(zip(*polygon_coord))]) == 2

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
            if Energy_core.is_rectengular(polygon)
        ]

    def qbits_in_max_core(self):
        """return the number of qbits
        in the maximum connected core
        if only_squares_in_core is True,
        triangulars are not considered 
        by choosing the largest core"""
        G = nx.Graph()
        plaquettes = self.polygon_object.nodes_object.qbits.found_plaqs()
        if self.only_squares_in_core:
            plaquettes = [plaq for plaq in plaquettes if len(plaq)==4]
        for l in plaquettes:
            nx.add_path(G, l)
        list_of_cores = list(nx.connected_components(G))
        if list_of_cores == []:
            return dict(), ((0, 0), (0, 0))
        max_core_qbits = max(list_of_cores, key=lambda x:len(x))

        qubit_coord_dict = {
            tuple(sorted(self.polygon_object.nodes_object.qbits[qbit].qubit))
            :self.polygon_object.nodes_object.qbits[qbit].coord
            for qbit in max_core_qbits}
        
        # move core to center of later initialization 
        core_center_x, core_center_y = Polygons.center_of_coords(qubit_coord_dict.values())
        center_envelop_of_all_coord = (np.sqrt(self.polygon_object.nodes_object.qbits.graph.K)) / 2
        delta_cx, delta_cy = int(center_envelop_of_all_coord-core_center_x), int(center_envelop_of_all_coord-core_center_y)
        
        for qubit, coord in qubit_coord_dict.items():
            qubit_coord_dict[qubit] = tuple(np.add(coord, (delta_cx, delta_cy)))
        # coords of envelop of core
        core_coords = list(qubit_coord_dict.values())
        corner = Polygons.corner_of_coords(core_coords)  
        
        # ancillas in core
        ancillas_in_core = {qbit.qubit: qbit.coord for qbit in self.polygon_object.nodes_object.qbits 
                            if qbit.ancilla==True and qbit.qubit in qubit_coord_dict}

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
                    scopes == self.polygon_object.unit_triangle_scope,
                    scopes == self.polygon_object.unit_square_scope,
                )
            ]
        )

        if self.only_number_of_plaquettes:
            return number_of_plaquettes, number_of_plaquettes

        distances_to_plaquette = self.scaled_distance_to_plaquette(scopes)

        if self.spring_energy:
            return round((distances_to_plaquette ** 2).sum(), 5), number_of_plaquettes
        else:
            return round((distances_to_plaquette).sum(), 5), number_of_plaquettes