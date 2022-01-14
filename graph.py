import numpy as np
import networkx as nx


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
        self.C = self.num_constrains()
        self.clear()

    def to_nx_graph(self):
        """converts Graph object to networkx object"""
        return nx.from_numpy_matrix(self.adj_matrix)

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

    def find_cycles(self, length):
        """
        searches for all closed cycles in graph by iterating
        over N - (n+1) nodes if path lenght is n
        """
        self.clear()
        for start_node in range(self.N - (length - 1)):
            self.DFS(length - 1, start_node, start_node, path=[start_node])
            self.visited[start_node] = True

    def cyclic_permutation(iself, path):
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

    def qbits_from_graph(self):
        triup_adj_graph = np.triu(self.adj_matrix, k=1)
        qbits = np.argwhere(triup_adj_graph == 1)
        return list(map(tuple, qbits))

    # TODO: count degeneracies D in graph
    def num_constrains(self):
        """C = K - N + 1"""
        return int(self.K - self.N + 1)

    def num_nodes(self):
        return int(self.adj_matrix.shape[0])

    def num_edges(self):
        return int(self.adj_matrix.sum() // 2)

    @classmethod
    def complete(cls, n):
        """generates complete connected graph with n nodes"""
        return cls(np.ones((n, n)) - np.identity(n))

    # TODO: random graph may have no cycles, fix this
    @classmethod
    def random(cls, n):
        """generates graph with n nodes and random connections"""
        matrix = np.triu(np.random.randint(2, size=(n, n)), k=1)
        matrix += matrix.transpose()
        return cls(matrix)
