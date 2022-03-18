import numpy as np
from CompilerQC import Polygons

default_schedule = {
        'ignore_inside_polygons':False,
        'scaled_as_LHZ': False,
        'individual_scaling': False,
        'polygon_weight': False,
        'no_scaling': False,
        }


class Energy(Polygons):
    def __init__(
        self,
        polygon_object,
        scaling_for_plaq3: float=0,
        scaling_for_plaq4: float=0,
    ):
        self.polygon_object = polygon_object
        self.scaling_for_plaq3 = scaling_for_plaq3
        self.scaling_for_plaq4 = scaling_for_plaq4
        
    def scopes_of_polygons(self):
        """ return array of the scopes of all changed polygons,
        """
        return np.array(list(map(self.scope_of_polygon, self.polygons_coords_of_interest)))
    
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
    
    def coords_of_changed_polygons(self, qbits_of_interest):
        """
        return unique polygons which changed during the move
        """
        qubit_to_coord_dict = self.polygon_object.qbits.qubit_to_coord_dict
        # all polygons which changed, as tuples not as lists (-> hashable)
        changed_polygons_coords = [tuple(polygon) for qbit in qbits_of_interest if not isinstance(qbit, tuple)
                                   for polygon in qbit.polygons_coords(qubit_to_coord_dict)]      
        # remove duplicates (only unique polygons)
        return [list(k) for k in set(changed_polygons_coords)]
    
    def __call__(self,
            qbits_of_interest,
            schedule=dict(),
                ):
        """
        return the energy and number of plaquettes of the polygons belonging to 
        qbits of interest
        """
#         default_schedule.update(schedule)
#         if default_schedule['ignore_inside_polygons']:
#             pass
        self.polygons_coords_of_interest = self.coords_of_changed_polygons(qbits_of_interest)
        scopes = self.scopes_of_polygons()
        number_of_plaquettes = len(scopes[
            np.logical_or(scopes == self.polygon_object.unit_triangle_scope,
                           scopes == self.polygon_object.unit_square_scope)])
        
        distances_to_plaquette = self.scaled_distance_to_plaquette(scopes)
        return round(distances_to_plaquette.sum(), 5), number_of_plaquettes

