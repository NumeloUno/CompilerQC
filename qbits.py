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
                 qbits:'list of Qbit objects'
                ):
        self.graph = graph
        self.qubits = {qbit.qubit:qbit for qbit in qbits} 
        self.update_core_shell()

        
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
    
    def set_core_qbits(self):
        """
        the attribute core_qbits should not change over time
        thus it is not a property object anymore, it will change only
        when core_qbits_() is called
        """
        self.core_qbits = [qbit for qubit, qbit in self.qubits.items() if qbit.core == True]

    def set_core_qbit_coords(self):
        """see core_qbits_()"""
        self.core_qbit_coords = [qbit.coord for qbit in self.core_qbits]
        
    def set_shell_qbits(self):
        """see core_qbits_()"""
        self.shell_qbits = [qbit for qubit, qbit in self.qubits.items() if qbit.core == False]
    
    def set_shell_qbit_coords(self):
        """see core_qbits_()"""
        self.shell_qbit_coords = [qbit.coord for qbit in self.shell_qbits]
    
    def assign_core_qbits(self, core_qubits):
        assert set(core_qubits).issubset(self.qubits.keys()), 'core qubit not in qubits' 
        for qubit in core_qubits:
            self.qubits[qubit].core = True
        self.update_core_shell()
        
    def update_core_shell(self):
        """
        set attributes like core_qbits/shell_qbits
        and their coords: core_qbit_coords/shell_qbit_coors
        """
        self.set_core_qbits(), self.set_core_qbit_coords()
        self.set_shell_qbits(), self.set_shell_qbit_coords()
        
    
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

    def random_shell_qbit(self):
        """
        return random qbit which is outside the core, in the shell
        """
        return random.choice(self.shell_qbits)
    
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
        self.core = core
        # self.polygons is set with set_polygons(), this function is called in init of Polygons
        self.polygons = None
        # self.plaquettes is set with set_plaquettes_of_qbits(), this function has to be called after Polygons is initialized
        self.plaquettes = None
        
    @property
    def qubit(self):
        """ qubit cant be overwritten like this"""
        return self._qubit
        
    
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
    
