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
        logical_nodes = list(range(self.qbits.graph.N))
        if coords is None:
            coords = logical_nodes[:]
            np.random.shuffle(coords)
        node_objects = [Node(n, coord) for n, coord in zip(logical_nodes, coords)]
        self.nodes = dict(zip(logical_nodes, node_objects))
        self.set_qbits_of_nodes()
        self.place_qbits_in_lines()

    def place_qbits_in_lines(self):
        """
        place qbits on the grid, such that they form 
        straight lines
        """
        for qbit in self.qbits:
            self.nodes_of_qubit(qbit)
            
    def set_qbits_of_nodes(self):
        """
        each node is involved in several qbits
        """
        for node_name, node in self.nodes.items():
            node.qbits = [qbit for qbit in self.qbits if node_name in qbit.qubit]
            
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
            for qbit in nodes_object.nodes[node_name].qbits)
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
                    [(node_obj.coord, name) for name, node_obj in nodes_object.nodes.items()]
                ))]