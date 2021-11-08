import numpy as np

class Graph():
    
    def __init__(self, adj_matrix):
        """
        adj_matrix: adjacency matrix is a square matrix used to represent a finite graph
        """
        self.graph = adj_matrix
        self.V = self.graph.shape[0]
        self.visited = [False] * self.V
        self.count = 0
        self.paths = []

    def DFS(self, n, node, start_node, path):
        """
        depth first search algo seachrs for closed path with lenght 'n' in graph starting by 
        'start_node'
        """
        self.visited[node] = True

        if n == 0:
            if self.graph[node][start_node] == 1:
                self.count += 1
                self.paths.append(path)
            self.visited[node] = False
            return
        for new_node in range(self.V):
            if self.visited[new_node] == False and self.graph[node][new_node] == 1:
                next_path = path[:]
                next_path.append(new_node)
                self.DFS(n-1, new_node, start_node, next_path)
        
        self.visited[node] = False
        return
    
    def find_cycles(self, length):
        """
        searches for all closed cycles in graph by iterating 
        over V - (n+1) nodes if path lenght is n 
        """
        self.count = 0
        for start_node in range(self.V - (length - 1)):
            self.DFS(length-1, start_node, start_node, path = [start_node])
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

    @classmethod
    def fully(cls, n):
        """ generates fully connected graph with n nodes"""
        return cls(np.ones((n,n)) - np.identity(n))

    @classmethod
    def random(cls, n):
        """generates graph with n nodes and random connections"""
        return cls(np.random.randint(2, size=(n,n)))
