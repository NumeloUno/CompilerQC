from CompilerQC import Graph, Polygons
import random
import numpy as np

class Qbits():
    """
    Qbits can be randomly initialized (random coords for qubits)
    or it can be initialized from dictionary {qubit: coord}.
    """
    def __init__(self,
                 graph: Graph,
                 qbits:'list of Qbit objects'):
        self.graph = graph
        self.qubits = {qbit.qubit:qbit for qbit in qbits} 
        
    @classmethod
    def init_qbits_from_dict(cls, graph, qubit_coord_dict: dict=None):
        """
        initialization of qbits according to qubit to coord dict, 
        if dict is not complete, remaining qbits are initialized 
        with random coordinates
        return: list ob qbit objects
        """
        core_coords = list(qubit_coord_dict.values())
        core_qubits = list(qubit_coord_dict.keys())

        if len(qubit_coord_dict) < graph.K:
            init_grid_size = int(np.sqrt(graph.K)) + 1
            coords = list(np.ndindex(init_grid_size, init_grid_size))
            assert len(coords) > graph.K, "init qbits: more qbits than coords"
            remaining_coords = [coord for coord in coords if coord not in core_coords]
            np.random.shuffle(remaining_coords)
            remaining_qubits = [qubit for qubit in graph.qbits if qubit not in core_qubits]
            remaining_qubit_coord_dict = dict(zip(remaining_qubits, remaining_coords))
            qubit_coord_dict.update(remaining_qubit_coord_dict)
        return cls(graph, [Qbit(qubit, coord) for qubit, coord in qubit_coord_dict.items()])

    def __getitem__(self, qbit):
        return self.qubits[qbit]
    
    def __iter__(self):
        return iter(self.qubits.values())
    
    @property  
    def coords(self):
        return [qbit.coord for qbit in self.qubits.values()]
    
    @property
    def coord_to_qbit_dict(self):
        """ coord to qbit object dict,
        whereas qubit_to_coord_dict is a qubit tuple to coord dict"""
        return {qbit.coord:qbit for qubit, qbit in self.qubits.items()}
        
    def qbit_from_coord(self, coord):
        """ if on coord sits a qbit, return qbit, else return NaN"""
        return self.coord_to_qbit_dict.get(coord, np.nan)
    
    def update(self, qubit, coord):
        self.qubits[qubit].coord = coord
        
    @property     
    def qubit_to_coord_dict(self):
        qubit_to_coord_dict = {qubit:qbit.coord for qubit, qbit in self.qubits.items()}
        assert len(set(list(qubit_to_coord_dict.keys()))) == len(set(list(qubit_to_coord_dict.values()))), 'non unique coords or qbits!'
        return qubit_to_coord_dict
    
    @property  
    def core_qbits_(self):
        return [qbit for qubit, qbit in self.qubits if qbit.core == True]
    
    def assign_core_qbits(self, core_qubits):
        for qubit, qbit in self.qubits.items():
            if qubit in core_qubits:
                qbit.core = True
    
    def set_all_polygons(self, all_polygons: list):
        """
        calling this function sets for all qbits
        the attribute polygons from None to a list of all 
        polygons they are involved in. This function is called in the __init__ of 
        Polygons.
        """
        for qbit in self.qubits.values():
            qbit.set_polygons(all_polygons)
            
    def random(self):
        """
        return random qbit
        """
        return random.choice(list(self.qubits.values()))
    
    def swap_qbits(self, qbit1, qbit2):
        """
        swap two qubits with each other
        """
        self.qubits[qbit1.qubit].coord, self.qubits[qbit2.qubit].coord = self.qubits[qbit2.qubit].coord ,self.qubits[qbit1.qubit].coord
    

    
    
    
class Qbit():
    """
    each qbit is an object with a name (qubit) and a coord.
    """    
    
    def __init__(
        self,
        qubit: tuple,
        coord: tuple, 
        core: bool=False,
        ):
        self._qubit = qubit
        self.coord = coord
        # self.polygons is set with set_polygons(), this function is called in init of Polygons
        self.polygons = None
        
    @property
    def qubit(self):
        """ qubit cant be overwritten like this"""
        return self._qubit
        
    def involved_in_plaqs(self, polygon_object):
        """
        returns list of plaquettes qbit is part of 
        """
        return [plaq for plaq in polygon_object.found_plaqs() if self.qubit in plaq]
    
    def set_polygons(self, all_polygons):
        """
        set polyons which belong to this qbit, this function is called
        in the init of Polygons()
        """
        if self.polygons is None:
             self.polygons = [polygon for polygon in all_polygons if self.qubit in polygon]
        
    def polygons_coords(self, qubit_to_coord_dict: dict):
        """ return coords of polygons which belong to this qbit"""
        return Polygons.polygons_coords(qubit_to_coord_dict, self.polygons)
    
