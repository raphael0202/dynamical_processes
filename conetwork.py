import networkx as nx
import random
import matplotlib.pyplot as plt


class CoNet():
    def __init__(self, M):
        """ Init co occurrence graph """
        self.M = M
        self.G = self.construct_graph()

    def construct_graph(self):
        """ Construct graph from matrix """
        return nx.from_numpy_matrix(self.M)

    def show(self):
        """ Plot the graph"""
        nx.draw(self.G)
        plt.draw()
        plt.show()