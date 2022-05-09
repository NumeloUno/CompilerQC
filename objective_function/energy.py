import numpy as np
from CompilerQC import Polygons

class Energy(Polygons):
    def __init__(
        self,
        polygon_object,
        scaling_for_plaq3: float=0,
        scaling_for_plaq4: float=0,
    ):
        self.polygon_object = polygon_object
        
        # scaling for plaquettes
        self.scaling_for_plaq3 = scaling_for_plaq3
        self.scaling_for_plaq4 = scaling_for_plaq4
        
        # polygon weights
        self.subset_weight = False
        self.subset_weight_offset = 0
        self.threshold_weight = False
        self.threshold = 0
        self.min_C_scope = False
        self.min_C_scopes_factor = 0
        self.sparse_density_penalty = False
        self.sparse_density_factor = 0
        self.square_scope = False
        self.min_4_scopes = False
        self.min_4_scope_weight = 1
        
    def scopes_of_polygons(self):
        """ return array of the scopes of all changed polygons,
        """
        return np.array(list(map(self.scope_of_polygon, self.polygons_coords_of_interest)))
    
    def rectengular_energy(self, qbits_of_interest):
        """consider only scopes of rectengular shape"""
        polygons_of_interest = self.changed_polygons(qbits_of_interest)
        self.polygons_coords_of_interest = self.coords_of_changed_polygons(polygons_of_interest)
#         self.polygons_coords_of_interest = [
# polygon for polygon in e.polygons_coords_of_interest
# if len({coord[0] for coord in polygon}) == 2 & len({coord[1] for coord in polygon}) == 2
# ]
        self.polygons_coords_of_interest = [
                polygon for polygon in self.polygons_coords_of_interest
                if (
                        polygon[0][0] == polygon[3][0]
                    and polygon[0][1] == polygon[1][1]
                    and polygon[1][0] == polygon[2][0]
                    and polygon[2][1] == polygon[3][1]
                ) ]
        scopes = np.array(list(map(self.scope_of_polygon, self.polygons_coords_of_interest)))
        number_of_plaquettes = len(scopes[
            np.logical_or(scopes == self.polygon_object.unit_triangle_scope,
                           scopes == self.polygon_object.unit_square_scope)])
        return scopes, number_of_plaquettes
    
    def scopes_of_polygons_for_analysis(self):
        """
        return four arrays of scopes:
        nonplaqs3, nonplaqs4, plaqs3, plaqs4
        """
        # consider all polygons
        self.polygons_coords_of_interest = self.polygon_object.polygons_coords(
        self.polygon_object.qbits.qubit_to_coord_dict, self.polygon_object.polygons)
        
        scopes = np.array(list(map(self.scope_of_polygon, self.polygons_coords_of_interest)))
        polygon_coords_lengths = np.array(list(map(len, self.polygons_coords_of_interest)))
        nonplaqs3 = scopes[ np.logical_and(polygon_coords_lengths == 3, scopes != self.polygon_object.unit_triangle_scope)] 
        nonplaqs4 = scopes[ np.logical_and(polygon_coords_lengths == 4, scopes != self.polygon_object.unit_square_scope)] 
        plaqs3 = scopes[ np.logical_and(polygon_coords_lengths == 3, scopes == self.polygon_object.unit_triangle_scope)] 
        plaqs4 = scopes[ np.logical_and(polygon_coords_lengths == 4, scopes == self.polygon_object.unit_square_scope)] 
        return nonplaqs3, nonplaqs4, plaqs3, plaqs4
    
    def scaled_distance_to_plaquette(
        self,
        scopes
            ):
        """
        input: array of scopes of polygons
        return: array of scopes of polygons, if a polygon is a plaquette, 
        the scaling_for_plaq is substracted
        """
        scopes[ scopes == self.polygon_object.unit_triangle_scope ] -= self.scaling_for_plaq3
        scopes[ scopes == self.polygon_object.unit_square_scope ] -= self.scaling_for_plaq4
        return scopes
    
    def coords_of_changed_polygons(self, polygons_of_interest):
        """
        return unique polygons coords which changed during the move
        """
        qubit_to_coord_dict = self.polygon_object.qbits.qubit_to_coord_dict
        return Polygons.polygons_coords(qubit_to_coord_dict, 
                                        polygons_of_interest)

    def changed_polygons(self, qbits_of_interest: list):
        """
        return unique polygons which changed during the move
        """
        # all polygons which changed, as tuples not as lists (-> hashable)
        changed_polygons = [tuple(polygon) for qbit in qbits_of_interest if not isinstance(qbit, tuple)
                                   for polygon in qbit.polygons]      
        # remove duplicates (only unique polygons)
        return [list(k) for k in set(changed_polygons)]
    
    def threshold_weights(self, scopes, threshold = 2):
        """
        give the polygons a weigth,
        if scope is below a given 
        threshold
        """
        weights = np.ones_like(scopes)
        weights[scopes > threshold] = 0
        return weights

    def subset_weights(self, polygons_of_interest, offset):
        """
        give the polygon a weight if it has no qbits, which 
        are in the diagonal of a four plaquette
        """
        
        found_plaqs = self.polygon_object.found_plaqs()
        forbidden_edges = [((p[0], p[3]),(p[1], p[2])) for p in found_plaqs if len(p)==4]
        forbidden_edges = [set(i) for j in forbidden_edges for i in j]
        weights = np.array([any([edge.issubset(polygon) for edge in forbidden_edges]) 
         if not (set(polygon) in map(set,found_plaqs)) 
         else False for polygon in polygons_of_interest])
        
        return 1 - weights.astype(int) + offset
    
    
    def min_C_scopes(self, factor):
        """
        return four arrays of scopes:
        nonplaqs3, nonplaqs4, plaqs3, plaqs4
        """
        # consider all polygons
        self.polygons_coords_of_interest = self.polygon_object.polygons_coords(
        self.polygon_object.qbits.qubit_to_coord_dict, self.polygon_object.polygons)
        
        scopes = np.array(list(map(self.scope_of_polygon, self.polygons_coords_of_interest)))        
        number_of_plaquettes = len(scopes[
        np.logical_or(scopes == self.polygon_object.unit_triangle_scope,
                       scopes == self.polygon_object.unit_square_scope)])
        
        polygon_coords_lengths = np.array(list(map(len, self.polygons_coords_of_interest)))
        polygon3 = scopes[polygon_coords_lengths == 3] 
        polygon4 = scopes[polygon_coords_lengths == 4] 
           
        polygon3 = sorted(polygon3)
        polygon4 = sorted(polygon4)
        C = self.polygon_object.qbits.graph.C
        C = int(factor * C)
        return polygon3[:C]+polygon4[:C], number_of_plaquettes
    
    
    def penalty_for_sparse_plaquette_density(self):
        """
        there is a penalty
        if a qbit has a lot of neighbours,
        but is involved only in few plaquettes.
        """
        # update plaquettes of each qbit
        self.polygon_object.set_plaquettes_of_qbits()
        self.polygon_object.set_numbers_of_qbits_neighbours()

        # if a qbit is involved in n plaquettes, it has in average qbits_per_plaq[n] neighbours which build these plaquettes
        qbits_per_plaq = [0, 2.5, 5, 6.5, 8]
        
        # the number of neighbour qbits (8-count(nan)) minus the avg. number of qbits which are in plaquettes
        return ([qbit.number_of_qbit_neighbours
                 - qbits_per_plaq[len(qbit.plaquettes)] for qbit in self.polygon_object.qbits])
    
    def penalty_for_isolated_qbits(self):
        """penalty"""
        self.polygon_object.set_numbers_of_qbits_neighbours()
        return 8 - np.array([qbit.number_of_qbit_neighbours for qbit in self.polygon_object.qbits])
 
    
    def __call__(self, qbits_of_interest):
        """
        return the energy and number of plaquettes of the polygons belonging to 
        qbits of interest
        """
 
        if self.min_C_scope:
            scopes, number_of_plaquettes = self.min_C_scopes(self.min_C_scopes_factor)
            return round(sum(scopes), 5), number_of_plaquettes
        
        if self.min_4_scopes:
            #scope
            polygons_of_interest = [[tuple(polygon) for polygon in qbit.polygons] 
                                    for qbit in qbits_of_interest if not isinstance(qbit, tuple)]  
            qubit_to_coord_dict = self.polygon_object.qbits.qubit_to_coord_dict
            polygons_coords_of_interest = [Polygons.polygons_coords(qubit_to_coord_dict, polygons)
                                           for polygons in polygons_of_interest]
            scopes = ([(sorted(list(map(self.scope_of_polygon, polygon_coords))))
                    for polygon_coords in polygons_coords_of_interest])
            

            scopes = np.array([self.scaled_distance_to_plaquette(np.array(scope))
                      * np.exp(-self.min_4_scope_weight*np.arange(len(scope)))
                      for scope in scopes])
            
            distances_to_plaquette = scopes.sum()

            #number of plaquettes
            polygons_of_interest = self.changed_polygons(qbits_of_interest)
            self.polygons_coords_of_interest = self.coords_of_changed_polygons(polygons_of_interest)
            scopes = self.scopes_of_polygons()
            number_of_plaquettes = len(scopes[
                np.logical_or(scopes == self.polygon_object.unit_triangle_scope,
                               scopes == self.polygon_object.unit_square_scope)])

            return round(distances_to_plaquette, 5) , number_of_plaquettes

        
        polygons_of_interest = self.changed_polygons(qbits_of_interest)
        self.polygons_coords_of_interest = self.coords_of_changed_polygons(polygons_of_interest)
        scopes = self.scopes_of_polygons()
        number_of_plaquettes = len(scopes[
            np.logical_or(scopes == self.polygon_object.unit_triangle_scope,
                           scopes == self.polygon_object.unit_square_scope)])
        
        distances_to_plaquette = self.scaled_distance_to_plaquette(scopes)
        
        if self.subset_weight:
            weights = self.subset_weights(polygons_of_interest, self.subset_weight_offset)
            return round(np.dot(distances_to_plaquette, weights), 5), number_of_plaquettes

        if self.threshold_weight:
            weights = self.threshold_weights(scopes, self.threshold)
            return round(np.dot(distances_to_plaquette, weights), 5), number_of_plaquettes
        
        if self.sparse_density_penalty:
            return (
                round(distances_to_plaquette.sum(), 5) 
                + self.sparse_density_factor
                * sum(self.penalty_for_sparse_plaquette_density())
            ), number_of_plaquettes
        
        if self.square_scope:
            return round((distances_to_plaquette ** 2).sum(), 5) , number_of_plaquettes
        else:
            return round(distances_to_plaquette.sum(), 5) , number_of_plaquettes

