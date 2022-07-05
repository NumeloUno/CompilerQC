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
    def init_qbits_from_dict(
        cls,
        graph,
        qubit_coord_dict: dict=None,
        assign_to_core: bool=True,
    ):
        """
        initialization of qbits according to qubit to coord dict, 
        return Qbits object
        """
        qbits = Qbits.qbits_from_dict(graph, qubit_coord_dict, assign_to_core)
        return cls(graph, qbits)
    
    @staticmethod
    def qbits_from_dict(
        graph,
        qubit_coord_dict: dict=None,
        assign_to_core: bool=True,
    ):
        """
        if dict is not complete, remaining qbits are initialized 
        with random coordinates
        return: list ob qbit objects
        """
        # change what the dicts point to - otherwise some annoying behaviour
        qubit_coord_dict = qubit_coord_dict.copy()
        
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
            
        qbits = [Qbit(qubit, coord) for qubit, coord in qubit_coord_dict.items()] 
        
        if assign_to_core:
            for qbit in qbits:
                if qbit.qubit in core_qubits:
                    qbit.core = True
        return qbits
        
    def update_qbits_from_dict(
        self,
        qubit_coord_dict: dict=None,
        assign_to_core: bool=True,
    ):
        """
        if Qbits object is already initialized,
        use this function to update the core and move
        qbits somewhere else to make some space
        """
        qbits = Qbits.qbits_from_dict(self.graph, qubit_coord_dict, assign_to_core)
        for qbit in qbits:
            # for adding ancilla
            if qbit.qubit not in self.qubits.keys():
                self.qubits.update({qbit.qubit:qbit})
            self.qubits[qbit.qubit].coord = qbit.coord
            self.qubits[qbit.qubit].core = qbit.core
        self.update_core_shell()

    def __getitem__(self, qbit):
        return self.qubits[qbit]
    
    def __iter__(self):
        return iter(self.qubits.values())
    
    @staticmethod
    def remove_core(qbits: 'Qbits object'):
        """ remove core assignment and 
        reinitialize coords"""
        # remove core
        for qbit in qbits:
            qbit.core = False
        qbits.update_core_shell()
        Qbits.reinit_coords(qbits, with_core=False)

    @staticmethod
    def reinit_coords(qbits: 'Qbits object', with_core: bool):
        """reinitialize coords"""
        init_grid_size = int(np.sqrt(qbits.graph.K)) + 1
        coords = list(np.ndindex(init_grid_size, init_grid_size))
        np.random.shuffle(coords) 
        if with_core:
            coords = [coord for coord in coords if coord not in qbits.core_qbit_coords]
            qbits = [qbit for qbit in qbits if qbit.core==False]
        for qbit, coord in zip(qbits, coords):
            qbit.coord = coord     
    
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
        
    def mark_ancillas(self, ancilla_names):
        """
        mark all ancillas in ancilla_names as such
        """
        for qubit in ancilla_names:
            self.qubits[qubit].ancilla = True
        
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
            
    def append_polygons(self, new_polygons: list):
        """
        new polygons which are added due to e.g.
        adding an ancilla, are added to the coresponding
        polygons attribute of each qbit object
        """
        for polygon in new_polygons:
            for qbit in self.qubits.values():
                if qbit.qubit in polygon:
                    qbit.polygons.append(polygon)
            
    def neighbours(self, coord):
        """
        return the 8 neighbours of a coord, if the neighbour is (not) occupied by another 
        qbit, (np.nan) the qbit object will be returned in a list of length 8
        """
        neighbours_ = lambda coord: [
            self.coord_to_qbit_dict.get((coord[0] + i, coord[1] + j), np.nan)
            for i in range(-1,2)
            for j in range(-1,2)
        ]
        return neighbours_(coord)
    
    @staticmethod
    def neighbour_coords(coord):
        """ return 8 neighbour coords and the coord itself as a list"""
        return [(coord[0] + i, coord[1] + j)
        for i in range(-1, 2) for j in range(-1, 2)]
    
    def set_number_of_qbit_neighbours(self, qbit):
        """
        set the number of occupied neighbours a qbit has,
        called via Polygons()
        """
        qbit.number_of_qbit_neighbours = (8 - self.neighbours(qbit.coord).count(np.nan))

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

    def random_weighted_shell_qbit(self, weights):
        """
        return random qbit weigthed by weights
        """
        return random.choices(self.shell_qbits, weights=weights, k=1)[0]

    def swap_qbits(self, qbit1, qbit2):
        """
        swap two qubits with each other
        """
        self.qubits[qbit1.qubit].coord, self.qubits[qbit2.qubit].coord = self.qubits[qbit2.qubit].coord ,self.qubits[qbit1.qubit].coord
        
    def add_ancillas_to_qbits(self, ancillas, new_polygons):
        ancilla_qbits = []
        for ancilla, coord in ancillas.items():
            qbit = Qbit(ancilla, coord)
            qbit.ancilla = True
            qbit.core = True
            ancilla_qbits.append(qbit)
        self.qubits.update({ancilla_qbit.qubit: ancilla_qbit for ancilla_qbit in ancilla_qbits})
        self.update_core_shell()
        self.append_polygons(new_polygons)
        return ancilla_qbits
    
    def remove_ancillas_from_qbits(self, ancillas):
        
        for qbit in self.qubits.values():
            if qbit.qubit in ancillas:
                continue
            # remove ancillas from qbit.polygons, creating new list is faster
            remaining_polygons = []
            for polygon in qbit.polygons:
                if set(ancillas).isdisjoint(polygon):
                     remaining_polygons.append(polygon)
            qbit.polygons = remaining_polygons

        # remove ancilla from qbits
        for ancilla in ancillas:
            self.qubits.pop(ancilla,None)
        self.update_core_shell() 


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
        self.ancilla = False
        # self.polygons is set with set_polygons(), this function is called in init of Polygons
        self.polygons = []
        # self.plaquettes is set with set_plaquettes_of_qbits(), this function has to be called after Polygons is initialized
        self.plaquettes = None
        self.number_of_qbit_neighbours = None
        
    @property
    def qubit(self):
        """ qubit cant be overwritten like this"""
        return self._qubit
    
    def set_polygons(self, all_polygons):
        """
        set polyons which belong to this qbit, this function is called
        in the init of Polygons()
        """
        #if self.polygons is None: warum war dieses if statement überhaupt hier?
        self.polygons = [polygon for polygon in all_polygons if self.qubit in polygon]
        
    def polygons_coords(self, qubit_to_coord_dict: dict):
        """ return coords of polygons which belong to this qbit"""
        return Polygons.polygons_coords(qubit_to_coord_dict, self.polygons)
    
