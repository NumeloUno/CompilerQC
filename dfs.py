import numpy as np

class Graph():
    
    def __init__(self, adj_matrix):
        """
        adj_matrix: adjacency matrix is a square matrix used to represent a finite graph
        """
        self.graph = adj_matrix
        self.V = self.graph.shape[0]
        self.clear()

    def clear(self):
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
        self.clear()
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
    # TODO: count degeneracies D in graph
    def num_constrains(self):
        """ C = K - N + 1"""
        return int(self.graph.sum() / 2 - self.V + 1)
    @classmethod
    def fully(cls, n):
        """ generates fully connected graph with n nodes"""
        return cls(np.ones((n,n)) - np.identity(n))

    @classmethod
    def random(cls, n):
        """generates graph with n nodes and random connections"""
        return cls(np.random.randint(2, size=(n,n)))

class Cycle_Graph():

    def __init__(self, cycl3, cycl4):
        
        self.sets = self.concatenate(
                cycl3, cycl4)

    def concatenate(self, cycl3, cycl4):
        return cycl3 + cycl4

    def cycle_adj_matrix(self):
        N = range(len(self.sets))
        matrix = [[self.distance(i, j) if i > j else 0 for i in N]
                for j in N]
        return matrix

    def distance(self, i, j):
       """measure the distance between 
       two closed loops"""
       cycli = self.get_list_of_neigb_tuples(i)
       cyclj = self.get_list_of_neigb_tuples(j)
       #return len(set(cycli) & set(cyclj))
       intersection = (map(lambda x: ''.join(map(str, x)),(set(cycli) & set(cyclj))))
       return ','.join(list(intersection))

    def get_list_of_neigb_tuples(self, index):
        cycle = self.sets[index]
        cycle = cycle + [cycle[0]]
        cycle = list(map(tuple, 
            map(sorted, zip(cycle, cycle[1:]))))
        return cycle

    def print_matrix(self, matrix):
        """ from stack overflow 13214809"""
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print ('\n'.join(table))


