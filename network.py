import networkx as nx
import random
import matplotlib.pyplot as plt

class WS() :

	def __init__(self,n,k,p):
		""" Initialize parameters of graph """
		self.n = n
		self.k = k
		self.p = p
		self.G = nx.Graph()
		self.construct_graph()

	def construct_graph(self):
		""" Construct nodes and edges of the graph"""
		self.add_nodes( range(self.n) )
		self.construct_edges()
		self.rewire_edges()

	def adjacency_matrix(self):
		""" Get adjacency matrix """
		return nx.adjacency_matrix(self.G)

	def add_nodes(self,nodes):
		""" Add nodes to the graph """
		self.G.add_nodes_from(nodes)

	def construct_edges(self):
		""" Generate a regular ring lattice """
		for V in xrange(0,self.n-1) : 

			for n in xrange(1,self.k/2+1) : 
				if not self.G.has_edge(V, (V+n)%self.n) : self.G.add_edge(V, (V+n)%self.n) 

			for n in xrange(-1,-(self.k/2+1),-1) : 		
				if not self.G.has_edge(V, (V+n)%self.n) : self.G.add_edge(V, (V+n)%self.n) 	

	def rewire_edges(self):
		"""Rewire edges of the graph with a proba p"""	
		for V in self.G.nodes():
			for N in self.G.edges(V):
				if random.random() < self.p : 
					L = filter(lambda x : x!=V and x not in self.G.neighbors(V), xrange(0,self.n-1))
					if len(L) > 0 : Nnew = random.choice( L )
					else : continue
					self.G.add_edge(V, Nnew)
					self.G.remove_edge(*N)

	def show(self):
		""" Plot the graph"""

		nx.draw(self.G)  
		plt.draw()
		plt.show()
