import networkx as nx
import random, itertools
import matplotlib.pyplot as plt

class CompGraph():
		
	def __init__(self, G1, G2) :
		
		self.I = G1
		self.C = G2

	def comparaison(self) :
		
		TN = TP = FN = FP = 0.
		# Liens possible dans C :
		L = list(itertools.combinations(self.C.nodes() ,2))
		for l in L :
			if self.C.has_edge(*l) and self.I.has_edge(*l) : TP +=1 ##
			if self.C.has_edge(*l) and not self.I.has_edge(*l) : FP +=1
			if not self.C.has_edge(*l) and not self.I.has_edge(*l) : TN +=1
		FN = len(self.I.edges()) - TP

		return TP / (TP + FN) , TN / (TN + FP)
