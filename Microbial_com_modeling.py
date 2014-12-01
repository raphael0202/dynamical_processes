# -*- coding: utf8 -*-

# In[1]:

from __future__ import division
import scipy.stats as stats
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

sns.set(font="Liberation Sans")

N = 400  # Number of distinct species
M = 100  # number of distinct species in the local community (M < N)

r = stats.uniform.rvs(loc=0, scale=1, size=M)  # growth rate, uniform distribution between 0 and 1, vector of size M
while np.any(r == 0.):
    print("0 value")
    for index, value in enumerate(r):
        if value == 0.:
            r[index] = stats.uniform.rvs(loc=0, scale=1)


# k: carrying capacity
k_even = stats.beta.rvs(a=1, b=1, loc=0, scale=1, size=M)  # uniform distribution
k_uneven = stats.beta.rvs(a=1, b=1.5, loc=0, scale=1, size=M)  # uneven distribution


# Scaling of carrying capacity k between 1 and 100
k_even = 1. + k_even * 100
k_uneven = 1. + k_uneven * 100


# ## Interaction matrix A

## Random Erdös-Renyi model

p = 2 * 2 / (N * (N - 1))  # Here, average of 2 interactions per species
# Probability that a link exist between two random nodes. Here, 2 interactions for each species in average

A_ER = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            A_ER[i][j] = 1
        else:
            if random.random() <= p:
                A_ER[i][j] = stats.uniform.rvs(loc=-1, scale=2)


## Subsampling
# We construct 'NB_LOCAL_COMMUNITY' communities composed of 'M' species each, with the following constraint:
# Each local community has to share a fraction 'FRACTION_SHARED' of species with the rest of the local communities.

NB_LOCAL_COMMUNITY = 300  # Number of local communities
FRACTION_SHARED = 0.80  # fraction of species that need to be shared between each pair of local community.
#FRACTION_SHARED * NB_LOCAL_COMMUNITY must be an integer
NB_COMMON_SPECIES = int(FRACTION_SHARED * M)

common_species_list = random.sample(xrange(N), NB_COMMON_SPECIES)  # List of species that need to be shared between
# each local community. xrange(N): list of the species, NB_COMMON_SPECIES: number of species to be chosen

local_comm_species = np.zeros((NB_LOCAL_COMMUNITY, M), dtype=int)  # Matrix representing the species (in the form of integers)
# chosen for each local population
local_comm_species[:, 0:NB_COMMON_SPECIES] = common_species_list

remaining_species = [x for x in xrange(N) if
                     x not in common_species_list]  # List of species that have not be chosen yet

for comm in xrange(NB_LOCAL_COMMUNITY):
    local_comm_species[comm, NB_COMMON_SPECIES:M] = random.sample(remaining_species, M - NB_COMMON_SPECIES)
    # We sample the rest of the species for each local community


# Initial abundance of species x_0
x_0 = stats.uniform.rvs(loc=10, scale=90, size=(NB_LOCAL_COMMUNITY, M))  # Uniform distribution between 10 and 100


def derivative(x, t0, A, k, r):
    return r * x * (1 - (np.dot(A, x) / k))


def steady_state(population_density, EPSILON=0.05, TIME_RANGE_PERCENT=10):
    """Check if all the populations of the community have reach a steady-state value."""

    time_range = int(population_density.shape[1] / TIME_RANGE_PERCENT)  # We select only the last timepoints for each
    # population
    population_density_reduced = population_density[:, -time_range:-1]

    for specie in range(len(population_density)):
        steady_state_value = population_density_reduced[specie][-1]
        steady_state_range = steady_state_value * np.array([1 - EPSILON, 1 + EPSILON])

        if np.any(population_density_reduced[specie] < steady_state_range[0]) or np.any(population_density_reduced[specie] > steady_state_range[1]):
            plt.plot(population_density[specie])
            plt.show()
            return False  # The population "specie" is not in the acceptable range of value for a steady-state

    return True


## We extract from the A_ER matrix the A matrix corresponding only to the species present in local_comm_species[0],
# in order to speed up computation (it avoids unnecessary calculus)


A = A_ER[:, local_comm_species[0, :]]
A = A[local_comm_species[0, :], :]  # We get the interaction matrix with only species present in the local population

## Alternative integration method through the ode function

t = np.arange(0., 500., 1.)

x = np.zeros((NB_LOCAL_COMMUNITY, M, len(t)))

for local_community_index in xrange(NB_LOCAL_COMMUNITY):
    x[local_community_index] = odeint(derivative, x_0[local_community_index], t, args=(A, k_even, r)).transpose()
    if not steady_state(x[local_community_index]):
        raise ValueError("One of the population has not reach a steady-value, increase the maximum time.")

