from CompilerQC import *
import numpy as np
import random 

class Node():
    """
    logical node from the problem graph
    """
    def __init__(self, name, coord):
        """
        each node has a name (integer),
        in the bipartite core picture, we can assgin 
        each node a coordinate,
        each node also has some qbits it is involved in, 
        this wont change during the process
        """
        self.name = name
        self.coord = coord
        self.qbits = None
    
class Nodes():
    
    def __init__(self, qbits, coords=None):
        """
        if not specified, assign the nodes random coords, 
        main part here is the self.nodes object, 
        its a dict with the nodes (integers) as keys and
        node object as values (as in the Qbits class)
        qbits are placed according to the coords of the nodes
        """
        self.qbits = qbits
        self.logical_nodes = list(range(self.qbits.graph.N))
        if coords is None:
            coords = self.logical_nodes[:]
            np.random.shuffle(coords)
        node_objects = [Node(n, coord) for n, coord in zip(self.logical_nodes, coords)]
        self.nodes = dict(zip(self.logical_nodes, node_objects))
        self.set_qbits_of_nodes()
        self.place_qbits_in_lines()
        
        # for ancillas
        self.diagonal_coords = set([(i, i) for i in range(self.qbits.graph.N)])

    def place_qbits_in_lines(self):
        """
        place qbits on the grid, such that they form 
        straight lines
        """
        for qbit in self.qbits:
            self.nodes_of_qubit(qbit)
            
    def set_qbits_of_nodes(self):
        """
        each node is involved in several qbits,
        added: each qbit gets attribute length_of_its_nodes,
        which is a tuple (number of qbits involved in the first node,
        number of qbits involved in the second node)
        """
        for node_name, node in self.nodes.items():
            node.qbits = [qbit for qbit in self.qbits if node_name in qbit.qubit]
            
        self.length_of_nodes = {i: len(self.qbits_of_nodes([i])) for i in self.nodes}
        for qbit in self.qbits:
            qbit.length_of_its_nodes = tuple([self.length_of_nodes.get(node) for node in qbit.qubit])
            
            
    def add_ancillas_to_nodes(self, ancillas):
        """append ancillas to node.qbits"""
        for ancilla in ancillas: 
            for node_name, node_object in self.nodes.items():
                if node_name in ancilla:
                    node_object.qbits.append(self.qbits[ancilla])

    def remove_ancillas_from_nodes(self, ancillas):
        """remove ancillas from node.qbits"""
        for ancilla in ancillas: 
            for node_name, node_object in self.nodes.items():
                if node_name in ancilla:
                    node_object.qbits = [qbit for qbit in node_object.qbits if qbit.qubit != ancilla]
                
    def nodes_of_qubit(self, qbit):
        """
        first identify which part of the qubit corresponds to which node,
        e.g if a qubti is (1, 4) it could come from the node 1 and node 4
        or from the node 4 and node 1, depending on the order, the
        coordinate will change - this function may be more complicated than
        it has to be
        """
        if self.nodes[qbit.qubit[0]].coord > self.nodes[qbit.qubit[1]].coord:
            qbit.coord = (self.nodes[qbit.qubit[1]].coord, self.nodes[qbit.qubit[0]].coord)
        else:
            qbit.coord = (self.nodes[qbit.qubit[0]].coord, self.nodes[qbit.qubit[1]].coord)
            
    def nodes_to_swap(self):
        return random.sample(self.nodes.keys(), 2)

    def swap_nodes(self, i, j):
        """
        swap coord of two nodes
        """
        self.nodes[i].coord, self.nodes[j].coord = self.nodes[j].coord, self.nodes[i].coord
        self.update_qbits_after_swap([i, j])
        
    def qbits_of_nodes(self, nodes_of_interest: list):
        """
        return list of qbits of several nodes,
        remove overlap
        """
        return list(
        set(qbit for node_name in nodes_of_interest
            for qbit in self.nodes[node_name].qbits)
        )

    def update_qbits_after_swap(self, nodes_of_interest: list):
        for qbit in self.qbits_of_nodes(nodes_of_interest):
            self.nodes_of_qubit(qbit)      
    @property
    def order(self):
        """
        return the order of the logical nodes
        """
        return [node for coord, node in 
                list(sorted(
                    [(node_obj.coord, name) for name, node_obj in self.nodes.items()]
                ))]
    
    @staticmethod
    def boolean_neighbours(qbits_coords, coord):
        """
        return list of booleans for the 8 neighbours of a coord,
        if the neighbour is occupied
        """
        neighbours_ = lambda coord: [
            (coord[0] + i, coord[1] + j) in qbits_coords
            for i in range(-1,2)
            for j in range(-1,2)
        ]
        return neighbours_(coord)

    @staticmethod
    def number_of_plaquettes_of_ancilla(qbits_coords, coord):
        """
        if an ancilla would be placed at coord, 
        number_of_possible_plaquettes tells you how many
        plaquettes this ancilla generates
        """
        neighbours = Nodes.boolean_neighbours(qbits_coords, coord)
        number_of_possible_plaquettes = sum(
            [
                sum([neighbours[0], neighbours[1], neighbours[3]]) == 3,
                sum([neighbours[2], neighbours[1], neighbours[5]]) == 3,
                sum([neighbours[8], neighbours[7], neighbours[5]]) == 3,
                sum([neighbours[6], neighbours[7], neighbours[3]]) == 3

            ]
        )
        return number_of_possible_plaquettes

    def name_of_ancilla(self, coord):
        """
        given a coord,
        return the qubit/name of the 
        possible ancilla
        """
        if 0 <= coord[0] < self.qbits.graph.N  and 0 <= coord[1] < self.qbits.graph.N:
            return tuple(sorted([self.order[coord[0]], self.order[coord[1]]]))
        else:
            return None
    
    def propose_coords(self):
        """
        return qbits coords (call it here to reduce running time,
        return coords for ancillas (coords which have at least one neighbour,
        all other coords makes no sense
        """
        qbit_coords = self.qbits.coords
        coords_for_ancillas = list(set(
            [coord for coords in qbit_coords
             for coord in self.qbits.neighbour_coords(coords)])
                                   - set(qbit_coords)
                                   - self.diagonal_coords
                                  )
        return qbit_coords, coords_for_ancillas
    
    def propose_ancillas(self, allowed_number):
        """
        returns a list of [number_of_new_plaqs, coord, qubit_ancilla) 
        of length of allowed_number of ancillas
        """
        qbit_coords, coords_for_ancillas = self.propose_coords()
        ancillas = []
        for coord in coords_for_ancillas:
            number_of_new_plaqs = Nodes.number_of_plaquettes_of_ancilla(qbit_coords, coord)
            # if below 2, this ancilla makes no sense
            if number_of_new_plaqs > 1:
                ancillas.append((number_of_new_plaqs, self.name_of_ancilla(coord), coord))
        ancillas.sort(reverse=True)
        best_ancillas = {ancilla: coord for number_of_new_plaqs, ancilla, coord 
                in ancillas[:allowed_number] if number_of_new_plaqs == 4}
        if len(best_ancillas) == 0:
            best_ancillas = {ancilla: coord for _, ancilla, coord 
                in ancillas[:1]}
        return best_ancillas
    
    def propose_ancillas_to_remove(self):
        """
        remove ancillas which are:
        * only in plaquettes where already other ancillas are involved or 
        ** only involved in one plaquette
        *** or not involved in plaquettes
        """
        all_ancillas = [qbit.qubit for qbit in self.qbits if qbit.ancilla==True]
        coords_around_ancillas = set([coord for qbit in self.qbits if qbit.ancilla==True for coord in Qbits.neighbour_coords(qbit.coord)])
        found_plaquettes = self.qbits.found_plaquettes_around_coords(coords_around_ancillas)
        # search for all ancillas, which are involved in plaquettes where only this ancillas takes part (intersection == 1) *
        ancillas_in_plaquettes = []
        for plaquette in found_plaquettes:
            intersection = list(set(all_ancillas).intersection(plaquette))
            if len(intersection) == 1:
                ancillas_in_plaquettes.append(intersection[0])
        # keep ancillas, which are part in >= 2 plaquettes  ** 
        ancillas_to_keep = [ancilla for ancilla in ancillas_in_plaquettes
                            if ancillas_in_plaquettes.count(ancilla) >= 2]  
        # additionally, remove ancillas which are not involved in plaquettes (***) 
        ancillas_to_remove = list(set(all_ancillas) - set(ancillas_to_keep))
        return ancillas_to_remove