# -*- coding: utf8 -*-

from __future__ import division
import scipy.stats as stats
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from math import sqrt

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

NB_LOCAL_COMMUNITY = 20  # Number of local communities
FRACTION_SHARED = 0.80  # fraction of species that need to be shared between each pair of local community.
#FRACTION_SHARED * NB_LOCAL_COMMUNITY must be an integer
NB_COMMON_SPECIES = int(FRACTION_SHARED * M)

common_species_list = np.random.choice(N, NB_COMMON_SPECIES, replace=False)  # List of species that need to be shared
# between each local community. xrange(N): list of the species, without repetition (replace=False).
# NB_COMMON_SPECIES: number of species to be chosen

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

    for specie in xrange(len(population_density)):
        steady_state_value = population_density_reduced[specie][-1]
        steady_state_range = steady_state_value * np.array([1 - EPSILON, 1 + EPSILON])

        if np.any(population_density_reduced[specie] < steady_state_range[0]) or np.any(population_density_reduced[specie] > steady_state_range[1]):
            plt.plot(population_density[specie])
            plt.show()
            return False  # The population "specie" is not in the acceptable range of value for a steady-state

    return True


def t_test_spearman(spearman_rho, N):
    return spearman_rho / sqrt((1 - spearman_rho**2)/(N - 2))


### We extract from the A_ER matrix the A matrix corresponding only to the species present in local_comm_species[0],
### in order to speed up computation (it avoids unnecessary calculus)

t = np.arange(0., 2000., 1.)
x = np.zeros((NB_LOCAL_COMMUNITY, M, len(t)))

# for local_community_index in xrange(NB_LOCAL_COMMUNITY):
#
#     # We get the interaction matrix with only species present in the local population
#
#     A = A_ER[:, local_comm_species[local_community_index, :]]  # First the columns
#     A = A[local_comm_species[local_community_index, :], :]  # Then the lines
#
#     x[local_community_index] = odeint(derivative, x_0[local_community_index], t, args=(A, k_even, r)).transpose()
#     if not steady_state(x[local_community_index]):
#         raise ValueError("One of the population has not reach a steady-value, increase the maximum time.")
#
#
# steady_state_densities = x[:, :, -1]
# np.save("densities", steady_state_densities)

steady_state_densities = np.load("densities.npy")

### Computation of the correlation coefficient (Spearman rho here)

couple_species = []
p_value_spearman = np.zeros((N, N))
NB_RESAMPLING = 1000

couple_species = []  # list of all the possible couple of species present in the local communities
local_comm_species_unique = list(set(local_comm_species.flatten()))

for specie_1 in common_species_list:
    for specie_2 in common_species_list[specie_1:]:
        couple_species.append((specie_1, specie_2))  # We store in the couple in a set

for specie_1, specie_2 in couple_species:

    null_distrib_rho = np.zeros(NB_RESAMPLING * NB_LOCAL_COMMUNITY)

    ## Computation of Spearman coefficient for all pairs

    density_specie_1 = steady_state_densities[local_comm_species == specie_1]  # We obtain the density of specie_1
    # for each local community in an array of length NB_LOCAL_COMMUNITY
    density_specie_2 = steady_state_densities[local_comm_species == specie_2]
    spearman_rho, _p_value = stats.spearmanr(density_specie_1, density_specie_2)

    spy = 0
    for resampling in xrange(NB_RESAMPLING):
        density_random_specie = np.zeros(NB_LOCAL_COMMUNITY)
        for local_community in xrange(NB_LOCAL_COMMUNITY):
            random_specie = np.random.choice(M)  # We chose 1 specie among all the species present in
            # the local community
            density_random_specie[local_community] = steady_state_densities[local_community, random_specie]

            ## Computation of the Spearman coefficient for the null distribution

            null_distrib_rho[spy], _p_value = stats.spearmanr(density_specie_1, density_random_specie)
            spy += 1

    ## Computation of the p-value

    p_value_spearman[specie_1, specie_2] = len(null_distrib_rho[null_distrib_rho >= spearman_rho]) / len(null_distrib_rho)
    p_value_spearman[specie_2, specie_1] = p_value_spearman[specie_1, specie_2]

    #TODO: implement the correction for multiple comparison by Benjamini and Hochberg (1995):
    #http://statsmodels.sourceforge.net/devel/generated/statsmodels.sandbox.stats.multicomp.multipletests.html#statsmodels.sandbox.stats.multicomp.multipletests

np.save("p_value", p_value_spearman)