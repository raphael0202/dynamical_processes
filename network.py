import networkx as nx
import random
import matplotlib.pyplot as plt


class ER():
    def __init__(self, n, p=0.1):
        """ Initialize parameters of graph """
        self.n = n
        self.m = n * (n - 1) * p / 2
        self.p = p

        self.G = nx.Graph()
        self.construct_graph()

    def construct_graph(self):
        """ Construct nodes and edges of the graph"""
        self.add_nodes(range(self.n))
        self.add_edges_p()
        self.show()

    def adjacency_matrix(self):
        """ Get adjacency matrix """
        return nx.adjacency_matrix(self.G)

    def add_nodes(self, nodes):
        """ Add nodes to the graph """
        self.G.add_nodes_from(nodes)

    def add_edges_m(self):
        """ Add edges to the graph according to the first ER model """
        while self.G.size() < self.m:
            N = random.sample(range(1, self.n + 1), 2)
            if not self.G.has_edge(min(N), max(N)): self.G.add_edge(min(N), max(N))

    def add_edges_p(self):
        """ Add edges to the graph according to the second ER model """
        for i in range(1, self.n + 1):
            for j in range(i + 1, self.n + 1):
                if random.random() < self.p: self.G.add_edge(i, j)

    def show(self):
        """ Plot the graph"""

        # nx.draw(self.G)
        #nx.draw_graphviz(self.G)
        #nx.draw_shell(self.G)
        nx.draw_circular(self.G)
        plt.draw()
        plt.show()


class WS():
    def __init__(self, n, k, p, M=None):
        """ Initialize parameters of graph """
        self.n = n
        self.k = k
        self.p = p
        self.G = nx.Graph()
        self.M = M
        if M is not None:
            self.construct_from_matrice()
        else:
            self.construct_graph()

    def construct_from_matrice(self):
        """ Construct nodes and edges of the graph"""
        self.G = nx.from_numpy_matrix(self.M)

    def construct_graph(self):
        """ Construct nodes and edges of the graph"""
        self.add_nodes(range(self.n))
        self.construct_edges()
        self.rewire_edges()

    def adjacency_matrix(self):
        """ Get adjacency matrix """
        return nx.adjacency_matrix(self.G)

    def add_nodes(self, nodes):
        """ Add nodes to the graph """
        self.G.add_nodes_from(nodes)

    def construct_edges(self):
        """ Generate a regular ring lattice """
        for V in xrange(0, self.n - 1):

            for n in xrange(1, self.k / 2 + 1):
                if not self.G.has_edge(V, (V + n) % self.n):
                    self.G.add_edge(V, (V + n) % self.n)

            for n in xrange(-1, -(self.k / 2 + 1), -1):
                if not self.G.has_edge(V, (V + n) % self.n):
                    self.G.add_edge(V, (V + n) % self.n)

    def rewire_edges(self):
        """Rewire edges of the graph with a probability p"""
        for V in self.G.nodes():
            for N in self.G.edges(V):
                if random.random() < self.p:
                    L = filter(lambda x: x != V and x not in self.G.neighbors(V), xrange(0, self.n - 1))
                    if len(L) > 0:
                        Nnew = random.choice(L)
                    else:
                        continue
                    self.G.add_edge(V, Nnew)
                    self.G.remove_edge(*N)

    def show(self):
        """ Plot the graph"""

        # nx.draw(self.G)
        #nx.draw_graphviz(self.G)
        #nx.draw_shell(self.G)
        nx.draw_circular(self.G)
        plt.draw()
        plt.show()


if __name__ == '__main__':
    # G = WS(20,4,1.)
    G = ER(20, 0.5)
    G.show()
