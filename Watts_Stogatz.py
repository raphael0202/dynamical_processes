# -*- coding: utf8 -*-

import numpy as np
import random

#--VARIABLES-----#

# Nombre d'espèces :
N = 10 

# Vecteur des taux de croissance ]0,1] :
R = np.random.uniform(0,1,N) 
while 0 in R : R = np.append( np.delete(R,np.where(R==0)) ,np.random.uniform(0,1))

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
# K: Degré moyen N >= K >= ln(N) >= 1
# B: 0 <= B <= 1
# L: Nombre de liens dans le graphe

K = 2
L = N*K/2 

# Matrice d'interaction :
l = np.append(np.ones(L),np.zeros(N*N - L)) 
np.random.shuffle(l)

A = (np.around(np.random.uniform(-1,1,(N,N)),2) * l.reshape(N,N)) + 2*np.identity(N)
A[A>1]=1

