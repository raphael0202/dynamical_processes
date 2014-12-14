# -*- coding: utf8 -*-

import numpy as np
import random, sys, itertools
from scipy.integrate import odeint
import scipy.stats as stats
import matplotlib.pyplot as pypl
from network import *
from conetwork import *
from compgraph import *

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
# Seuil de significativité :
alpha = 0.1

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

k = int(0.3 * N )
p = 0.2
G = WS(N,k,p)

# Matrice d'interaction :
A = np.multiply( G.adjacency_matrix() , np.around(np.random.uniform(-1,1,(N,N)),2) )
np.fill_diagonal(A, 1)
A[A == -0.] = 0.
print "A =\n", A
G = WS(N,k,p,A)

# Affiche le graphe :
G.show()

#--ECHANTILLONAGE-----#

v_shared_species = np.append(np.ones(N_shared_species),np.zeros(N - N_shared_species))
np.random.shuffle( v_shared_species )
i = filter(lambda x : v_shared_species[x]!=1, xrange(len(v_shared_species)))

print 'Espèces partagées =', v_shared_species

# Matrice des espèces présentent dans chaque communautés :
M = np.zeros( (N_communities, N) )
for community in xrange(N_communities) :
	s = random.sample(i, N_species_local - N_shared_species)
	M[community,] = v_shared_species
	M[community,][s] = 1

print "M = \n", M

#--DYNAMIQUE-----#

def lotka_voltera(X, t, R, A, K): return R * X *  (1 - (np.squeeze(np.asarray(np.dot(A,X)) / K)))

def condition_equilibre(X) : return True if  (abs(X[:,-2] - X[:,-1]) < 0.05).all() else False

#--RESOLUTION-----#

print "# RESOLUTION"
# Matrice de densité :
D = np.zeros((N,N_communities))

for community in xrange(N_communities) :

	X_community = X[np.where(M[community,] > 0)]
	A_community = A[np.where(M[community,] > 0)][:,np.where(M[community,] > 0)[0].tolist()]
	R_community = R[np.where(M[community,] > 0)]
	K_community = K_uniform[np.where(M[community,] >0)]

	t = np.arange(0., 3000., 1.)
	X_community = odeint(lotka_voltera, X_community, t, args=(R_community, A_community, K_community))
	if not condition_equilibre(X_community.transpose()) : print "La stabilité n'est pas atteinte pour la communauté n°%s" % (community+1)
	
	D[np.where(M[community,] > 0),community] = X_community.transpose()[:,-1]
	#pypl.plot(t,X_community)
	#pypl.show()

print "D = \n", D
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~		
def cprint(label) : 
	""" Fonction d'affichage dynamique en console """

	sys.stdout.write("\r%s" % label) 
	sys.stdout.flush()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

#--CO-OCCURENCE-----#

print "# CO-OCCURENCE"
# Couples d'espèces :
C = list(itertools.combinations(np.arange(N)[np.where(v_shared_species > 0)], 2))
# Matrice du réseau de co-occurence :
Mcc = Acc = np.zeros((N,N))

print "## Resampling"
for i, pair in enumerate(C) :
	cprint("%s/%s" % (i, len(C)))

	density_spec1 , density_spec2 = D[pair[0]], D[pair[1]]
	rho, pvalue = stats.spearmanr( density_spec1 , density_spec2 )

	# Resampling :
	rho_null = np.zeros(N_resampling)
	for i in xrange(N_resampling) : 
		random.shuffle(density_spec1)
		rho_null[i] = stats.spearmanr(density_spec1 , density_spec2)[0]
	print " distri_null = ", rho_null
	print " rho = ", rho
	pvalue = len(rho_null[np.where(rho_null >= rho)]) / float(N_resampling) ##
	print " pval = ", pvalue
	if 1- pvalue <= alpha : Mcc[pair[0],pair[1]] = Mcc[pair[1],pair[0]] = rho

print "# GRAPHE CO-OCCURENCE"
# Graphe de co-occurence :
Acc[Mcc != 0 ] = 1
print "Acc = \n", Acc
Gcc = CoNet(Acc)
Gcc.show()

Comp = CompGraph(G.G,Gcc.G)
sensibilite , specificite = Comp.comparaison()
print "Sensi = ", sensibilite
print "Speci = ", specificite

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

