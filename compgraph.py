import networkx as nx
import random, itertools
import matplotlib.pyplot as plt
import numpy as np

class CompGraph():
		
	def __init__(self, G1, G2) :
		
		self.I = G1
		self.C = G2

	def comparaison(self) :
		
		TN = TP = FN = FP = 0.
		# Liens possible dans C :
		L = list(itertools.combinations(self.C.nodes() ,2))
		for l in L :
			if self.C.has_edge(*l) and self.I.has_edge(*l) and self.sign(*l) : TP +=1
			if self.C.has_edge(*l) and not self.I.has_edge(*l) : FP +=1
			if not self.C.has_edge(*l) and not self.I.has_edge(*l) : TN +=1
			if not self.C.has_edge(*l) and self.I.has_edge(*l) : FN +=1
		return TP / (TP + FN) , TN / (TN + FP)

	def sign(self,v1,v2):
		return True if np.sign(self.C.get_edge_data(v1,v2)['weight']) == np.sign(self.I.get_edge_data(v1,v2)['weight']) else False
