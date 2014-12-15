import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

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
        nx.draw_circular(self.G)
        plt.draw()
        plt.show()

if __name__ == '__main__':

	M = np.matrix([[1,0.8,-0.2],[0,1,-0.5],[0.1,0,1]])
	G = CoNet(M)
