# -*- coding: utf8 -*-

from __future__ import division
import scipy.stats as stats
from scipy.integrate import odeint
import numpy as np
import statsmodels
import matplotlib.pyplot as plt
import seaborn as sns
import random
from math import sqrt
import graph
sns.set(font="Liberation Sans")


def subsample_local_pop(nb_species, nb_local_community, fraction_shared):
    """"""
    # We construct 'nb_local_community' communities composed of 'M' species each, with the following constraint:
    # Each local community has to share a fraction 'fraction_shared' of species with the rest of the local communities.

    #fraction_shared * nb_local_community must be an integer
    NB_COMMON_SPECIES = int(fraction_shared * M)

    common_species_list = np.random.choice(nb_species, NB_COMMON_SPECIES, replace=False)  # List of species that need
    # to be shared between each local community. xrange(nb_species): list of the species, without repetition
    # (replace=False).
    # NB_COMMON_SPECIES: number of species to be chosen

    local_comm_species = np.zeros((nb_local_community, M), dtype=int)  # Matrix representing the species (in the form
    # of integers) chosen for each local population
    local_comm_species[:, 0:NB_COMMON_SPECIES] = common_species_list

    remaining_species = [x for x in xrange(nb_species) if
                         x not in common_species_list]  # List of species that have not be chosen yet

    for comm in xrange(nb_local_community):
        local_comm_species[comm, NB_COMMON_SPECIES:M] = random.sample(remaining_species, M - NB_COMMON_SPECIES)
        # We sample the rest of the species for each local community

    return local_comm_species, common_species_list


def derivative(x, t0, A, k, r):
    return r * x * (1 - (np.dot(A, x) / k))


def steady_state_check(population_density, epsilon=0.1, time_range_percent=10):
    """Check if all the populations of the community have reach a steady-state value."""

    time_range = int(population_density.shape[1] / time_range_percent)  # We select only the last timepoints for each
    # population
    population_density_reduced = population_density[:, -time_range:-1]

    for specie in xrange(len(population_density)):
        steady_state_value = population_density_reduced[specie][-1]
        steady_state_range = steady_state_value * np.array([1 - epsilon, 1 + epsilon])

        if np.any(population_density_reduced[specie] < steady_state_range[0]) or np.any(population_density_reduced[specie] > steady_state_range[1]):
            plt.plot(population_density[specie])
            plt.show()
            return False  # The population "specie" is not in the acceptable range of value for a steady-state

    return True


def get_steady_state_densities(nb_local_community, M, local_comm_species, x_0, A_ER, k_even, r, t_max=2000., t_min=0, ts=1.):

    t = np.arange(t_min, t_max, ts)
    x = np.zeros((nb_local_community, M, len(t)))

    for local_community_index in xrange(nb_local_community):

        # We get the interaction matrix with only species present in the local population

        A = A_ER[:, local_comm_species[local_community_index, :]]  # First the columns
        A = A[local_comm_species[local_community_index, :], :]  # Then the lines

        x[local_community_index] = odeint(derivative, x_0[local_community_index], t, args=(A, k_even, r)).transpose()
        if not steady_state_check(x[local_community_index]):
            raise ValueError("One of the population has not reach a steady-value, increase the maximum time.")

    steady_state_densities = x[:, :, -1]

    return steady_state_densities


def p_value_spearman(steady_state_densities, common_species_list, nb_local_community, nb_resampling=1000):

    ## list of all the possible couple of species present in the local communities
    couple_species = [(specie_1, specie_2) for specie_1 in common_species_list
                      for specie_2 in common_species_list if specie_2 > specie_1]

    p_value_spearman = np.zeros(len(couple_species))

    for index, (specie_1, specie_2) in enumerate(couple_species):

        null_distrib_rho = np.zeros(nb_resampling * nb_local_community)

        ## Computation of Spearman coefficient for all pairs

        density_specie_1 = steady_state_densities[local_comm_species == specie_1]  # We obtain the density of specie_1
        # for each local community in an array of length nb_local_community
        density_specie_2 = steady_state_densities[local_comm_species == specie_2]
        spearman_rho, _p_value = stats.spearmanr(density_specie_1, density_specie_2)

        spy = 0

        for resampling in xrange(nb_resampling):
            np.random.shuffle(density_specie_1)
            ## Computation of the Spearman coefficient for the null distribution
            null_distrib_rho[spy], _p_value = stats.spearmanr(density_specie_1, density_specie_2)
            spy += 1

        ## Computation of the p-value

        p_value_spearman[index] = len(null_distrib_rho[null_distrib_rho >= spearman_rho]) / len(null_distrib_rho)

        ## Correction for multiple comparison by Benjamini and Hochberg (1995):
        ## http://statsmodels.sourceforge.net/devel/generated/statsmodels.sandbox.stats.multicomp.multipletests.html
        # #statsmodels.sandbox.stats.multicomp.multipletests

    ## Computation of the correction for multiple comparison
    rejects, p_value_corrected,\
    _alpha_1, _alpha_2 = statsmodels.sandbox.stats.multicomp.multipletests(p_value_spearman, method="fdr_bh")

    p_value_spearman_corrected = np.zeros((N, N))

    for index, (specie_1, specie_2) in enumerate(couple_species):
        p_value_spearman_corrected[specie_1, specie_2] = p_value_spearman[index]
        p_value_spearman_corrected[specie_2, specie_1] = p_value_spearman[index]

    return p_value_spearman_corrected


N = 40  # Number of distinct species
M = 20  # number of distinct species in the local community (M < N)

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

# Interaction matrix A
## Random Erdös-Renyi model

p = 2 * 2 / (N * (N - 1))  # Here, average of 2 interactions per species
# Probability that a link exist between two random nodes. Here, 2 interactions for each species in average

A_ER = graph.generate_random_graph(N, p)

NB_LOCAL_COMMUNITY = 10  # Number of local communities
FRACTION_SHARED = 0.80  # fraction of species that need to be shared between each pair of local community.
#FRACTION_SHARED * NB_LOCAL_COMMUNITY must be an integer
NB_COMMON_SPECIES = int(FRACTION_SHARED * M)

local_comm_species, common_species_list = subsample_local_pop(N, NB_LOCAL_COMMUNITY, FRACTION_SHARED)

# Initial abundance of species x_0
x_0 = stats.uniform.rvs(loc=10, scale=90, size=(NB_LOCAL_COMMUNITY, M))  # Uniform distribution between 10 and 100
steady_state_densities = get_steady_state_densities(NB_LOCAL_COMMUNITY, M, local_comm_species, x_0, A_ER, k_even, r, t_max=2000., t_min=0, ts=1.)


### Computation of the correlation coefficient (Spearman rho here)

p_value_spearman = p_value_spearman(steady_state_densities, common_species_list, NB_LOCAL_COMMUNITY)
plt.imshow(p_value_spearman)
plt.show()
#np.save("p_value", p_value_spearman)
#p_value_spearman = np.load("p_value.npy")