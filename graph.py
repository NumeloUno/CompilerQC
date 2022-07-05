import numpy as np
import networkx as nx

# TODO: graph.nodes is the same as logical_nodes in nodes.py, replace logical nodes!
class Graph:
    def __init__(self, adj_matrix):
        """
        adj_matrix: adjacency matrix is a square matrix used to represent a finite graph
        N, K: N(odes), K(anten) of logical graph
        C: C(onstraints) = number of plaquettes in phys. graph
        """
        self.adj_matrix = adj_matrix
        self.N = self.num_nodes()
        self.K = self.num_edges()
        self.clear()
        three_cycles, four_cycles = self.get_cycles(3), self.get_cycles(4)
        self.number_of_3_cycles, self.number_of_4_cycles = (
            len(three_cycles),
            len(four_cycles),
        )
        self.cycles = three_cycles + four_cycles
        self.qbits = self.qbits_from_graph()
        self.nodes = self.nodes_from_graph()
        self.C = self.num_constrains()

    def to_nx_graph(self):
        """converts Graph object to networkx object"""
        return nx.from_numpy_array(self.adj_matrix)

    def clear(self):
        self.visited = [False] * self.N
        self.count = 0
        self.paths = []

    def DFS(self, n, node, start_node, path):
        """
        depth first search algo seachrs for closed path with lenght 'n' in graph starting by
        'start_node'
        """
        self.visited[node] = True
        if n == 0:
            if self.adj_matrix[node][start_node]:
                self.count += 1
                self.paths.append(path)
            self.visited[node] = False
            return
        for new_node in range(self.N):
            if not self.visited[new_node] and self.adj_matrix[node][new_node]:
                next_path = path[:]
                next_path.append(new_node)
                self.DFS(n - 1, new_node, start_node, next_path)
        self.visited[node] = False
        return

    @staticmethod
    def sort_up_to_permutation(cycle):
        i = cycle.index(min(cycle))
        return cycle[i:] + cycle[:i]

    def DFS_for_ancilla(self, end_node, n, node, start_node, path):
        """
        depth first search algo seachrs for closed path with lenght 'n' in graph starting by
        'start_node'
        """
        self.visited[node] = True
        if n == 0:
            if node == end_node:
                self.count += 1
                self.paths.append(Graph.sort_up_to_permutation(path))
            self.visited[node] = False
            return
        for new_node in range(self.N):
            if not self.visited[new_node] and self.adj_matrix[node][new_node]:
                next_path = path[:]
                next_path.append(new_node)
                self.DFS_for_ancilla(end_node, n - 1, new_node, start_node, next_path)
        self.visited[node] = False
        return
    
    def get_cycles_of_ancilla(self, ancilla, length):
        """
        get cycles of a specific ancilla, 
        the ancilla does not have to be added to the adj 
        matrix in advance
        """
        self.clear()
        start_node, end_node = ancilla
        self.DFS_for_ancilla(end_node, length - 1, start_node, start_node, path=[start_node])
        return self.paths
    
    def get_cycles_of_ancillas(self, ancillas, length):
        """
        get all cycles of ancillas, and remove duplicates
        ancillas have to be updated to the adj matrix in advance
        by graph.update_ancillas()
        """
        paths = []
        for ancilla in ancillas:
            self.get_cycles_of_ancilla(ancilla, length)
            paths.append(list(map(tuple, self.paths)))
        self.paths = list(map(list, set(
            [path for paths_of_ancilla in paths for path in paths_of_ancilla]
        )))
        self.drop_duplicates()
        return self.paths

    def find_cycles(self, length):
        """
        searches for all closed cycles in graph by iterating
        over N - (n+1) nodes if path lenght is n
        """
        self.clear()
        for start_node in range(self.N - (length - 1)):
            self.DFS(length - 1, start_node, start_node, path=[start_node])
            self.visited[start_node] = True

    def cyclic_permutation(self, path):
        rev_list = list(reversed(path))
        return [rev_list.pop()] + rev_list

    def drop_duplicates(self):
        """each cycle is counted twice,
        here one count is removed"""
        for p in self.paths:
            cycl_perm = self.cyclic_permutation(p)
            if cycl_perm in self.paths:
                self.paths.remove(cycl_perm)

    def get_cycles(self, length):
        self.find_cycles(length)
        self.drop_duplicates()
        return self.paths

    def add_ancillas_to_adj_matrix(self, ancillas):
        for (i, j) in ancillas:
            self.adj_matrix[i][j] = self.adj_matrix[j][i] = 1
    
    def remove_ancillas_from_adj_matrix(self, ancillas):
        for (i, j) in ancillas:
            self.adj_matrix[i][j] = self.adj_matrix[j][i] = 0

    def remove_ancillas(self, ancillas):
        self.remove_ancillas_from_adj_matrix(ancillas)
        self.K = self.num_edges()
        for ancilla in ancillas:
            self.qbits.remove(ancilla)
        assert self.K == len(self.qbits), "number of qbits is not equal to K"
        self.nodes = self.nodes_from_graph()
        self.C = self.num_constrains()   
        
    def update_ancillas(self, ancillas):
        self.add_ancillas_to_adj_matrix(ancillas)
        self.K = self.num_edges()
        for ancilla in ancillas:
            self.qbits.append(ancilla)
        assert self.K == len(self.qbits), "number of qbits is not equal to K"
        self.nodes = self.nodes_from_graph()
        self.C = self.num_constrains()     
        
    def qbits_from_graph(self):
        triup_adj_graph = np.triu(self.adj_matrix, k=1)
        qbits = np.argwhere(triup_adj_graph == 1)
        return list(map(tuple, qbits))

    # TODO: count degeneracies D in graph
    def num_constrains(self):
        """C = K - N + 1"""
        return int(self.K - len(self.nodes) + 1)

    # TODO: replace by len(self.nodes)
    def num_nodes(self):
        return int(self.adj_matrix.shape[0])

    def nodes_from_graph(self):
        """return nodes"""
        return list({node for qubit in self.qbits for node in qubit})

    def num_edges(self):
        return int(self.adj_matrix.sum() // 2)

    @classmethod
    def complete(cls, n):
        """generates complete connected graph with n nodes"""
        return cls(np.ones((n, n)) - np.identity(n))

    @classmethod
    def init_without_edges(cls, n, edges_to_remove: list):
        """generate graph without edges in edges_to_remove"""
        adj_matrix = np.ones((n, n)) - np.identity(n)
        for edge in edges_to_remove:
            adj_matrix[edge], adj_matrix[edge[::-1]] = 0, 0
        return cls(adj_matrix)

    # TODO: random graph may have no cycles, fix this-> check if C,K,N are compatibe, add density parameter
    @classmethod
    def random(cls, n):
        """generates graph with n nodes and random connections"""
        matrix = np.triu(np.random.randint(2, size=(n, n)), k=1)
        matrix += matrix.transpose()
        return cls(matrix)
