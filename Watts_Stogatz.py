# -*- coding: utf8 -*-

import numpy as np
import random, sys, itertools
from scipy.integrate import odeint
import scipy.stats as stats
import matplotlib.pyplot as pypl

#--VARIABLES-----#

# Nombre d'espèces :
N = 10
# Nombre de communautés :
N_communities = 3
# Nombre d'espèce par communauté :
N_species_local = 5
# Fraction d'espèce partagée d'une communauté à l'autre :
F_shared = 0.8
# Nombre d'espèce partagée en fonction de la fraction définie :
N_shared_species = int(N_species_local * F_shared)
#Nombre de ré-échantillonage :
N_resampling = 100

# Vecteur des taux de croissance ]0,1] :
R = np.random.uniform(0,1,N) 
while 0 in R : R = np.append( np.delete(R,np.where(R==0)), np.random.uniform(0,1))

# Vecteurs des capacités limites [1,100] :
K_uniform   = 1 + ( np.random.beta(1,1  ,N) * 100 )
K_uneven    = 1 + ( np.random.beta(1,1.5,N) * 100 )
K_lognormal = np.random.lognormal(0,1,N)
K_lognormal = 1 + ( (K_lognormal - min(K_lognormal))/max(K_lognormal) * 100 )

# Vecteurs des abondances d'espèces :
X = np.random.uniform(10,100,N) 

#--MODELE-----#

# Watts-Stogatz :
# N: Nombre de noeuds
# k: Degré moyen N >= k >= ln(N) >= 1
# B: 0 <= B <= 1
# L: Nombre de liens dans le graphe

k = 2
L = N*k/2 

# Matrice d'interaction :
l = np.append(np.ones(L),np.zeros(N*N - L)) 
np.random.shuffle(l)

A = (np.around(np.random.uniform(-1,1,(N,N)),2) * l.reshape(N,N)) + 2*np.identity(N)
A[A>1]=1

#--ECHANTILLONAGE-----#

v_shared_species = np.append(np.ones(N_shared_species),np.zeros(N - N_shared_species))
np.random.shuffle( v_shared_species )
i = filter(lambda x : v_shared_species[x]!=1, range(len(v_shared_species)))

# Matrice des espèces présentent dans chaque communautés :
M = np.zeros( (N_communities, N) )
for community in range(N_communities) :
	s = random.sample(i, N_species_local - N_shared_species)
	M[community,] = v_shared_species
	M[community,][s] = 1

#--DYNAMIQUE-----#

def lotka_voltera(X, t, R, A, K): return R * X * (1 - (np.dot(A,X) / K))

def condition_equilibre(X) : return True if  (abs(X[:,-2] - X[:,-1]) < 0.05).all() else False

#--RESOLUTION-----#

# Résolutions différentes pour chaque communautés :

# Matrice de densité :
D = np.zeros((N,N_communities))
# Dictionnaire des listes de paires d'espèce de la communauté :
d_paires = {}

for community in range(N_communities) :

	X_community = X[np.where(M[community,] > 0)]
	A_community = A[np.where(M[community,] > 0)][:,np.where(M[community,] > 0)[0].tolist()]
	R_community = R[np.where(M[community,] > 0)]
	K_community = K_uniform[np.where(M[community,] >0)]

	t = np.arange(0., 500., 1.)
	X_community = odeint(lotka_voltera, X_community, t, args=(R_community, A_community, K_community))
	if not condition_equilibre(X_community.transpose()) : print "La stabilité n'est pas atteinte pour la communauté n°%s" % (community+1)
	
	D[np.where(M[community,] > 0),community] = X_community.transpose()[:,-1]
	#pypl.plot(t,X_community)
	#pypl.show()

	d_paires[community] = list(itertools.combinations( np.arange(1,N+1)[np.where(M[community,] > 0)] ,2))

#### Association metrics ####

# Steadman :
#-----------
# rho, pvalue = scipy.stats.spearmanr(x,y)

# Pearson :
#----------
# pvalue = scipy.stats.pearsonr(x, y)

# Bray-Curtis dissimilarity  :
#-----------------------------
#d = scipy.spatial.distance.braycurtis(x, y)

# Jaccard index :
#----------------
"""
def compute_jaccard_index(x, y):
	# Avec x,y des sets
    n = len(x.intersection(y))
    return n / float(len(x) + len(y) - n)
"""

# Kendall coefficient :
#----------------------
#tau, pvalue = scipy.stats.kendalltau(x, y)

