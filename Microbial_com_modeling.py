# -*- coding: utf8 -*-

from __future__ import division
import scipy.stats as stats
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import random
import statsmodels
import seaborn as sns

sns.set_style("white")
sns.set(font="Liberation Sans")


def subsample_local_pop(total_nb_species, nb_species, nb_local_community, nb_common_species):
    """"""
    # We construct 'nb_local_community' communities composed of 'nb_species' species each, with the following constraint:
    # Each local community has to share a fraction 'fraction_shared' of species with the rest of the local communities.

    common_species_list = np.random.choice(total_nb_species, nb_common_species, replace=False)  # List of species that
    # need to be shared between each local community. xrange(total_nb_species): list of the species, without repetition
    # (replace=False).
    # NB_COMMON_SPECIES: number of species to be chosen

    local_comm_species = np.zeros((nb_local_community, nb_species), dtype=int)  # Matrix representing the species (in the form
    # of integers) chosen for each local population
    local_comm_species[:, 0:nb_common_species] = common_species_list

    remaining_species = [x for x in xrange(total_nb_species) if
                         x not in common_species_list]  # List of species that have not be chosen yet

    for comm in xrange(nb_local_community):
        local_comm_species[comm, nb_common_species:nb_species] = random.sample(remaining_species, nb_species - nb_common_species)
        # We sample the rest of the species for each local community

    return local_comm_species, common_species_list


def derivative(x, t0, A, k, r):
    return r * x * (1 - (np.dot(A, x) / k))


def steady_state_check(population_density, epsilon=0.5, time_range_percent=10):
    """Check if all the populations of the community have reach a steady-state value."""

    time_range = int(population_density.shape[1] / time_range_percent)  # We select only the last timepoints for each
    # population
    population_density_reduced = population_density[:, -time_range:-1]

    for specie in xrange(len(population_density)):
        steady_state_value = population_density_reduced[specie, -1]
        steady_state_range = np.array([steady_state_value - epsilon, steady_state_value + epsilon])

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


def p_value_spearman(steady_state_densities, couple_species, total_nb_species, local_comm_species,
                     nb_resampling=1000):

    p_value_spearman = np.zeros(len(couple_species))
    p_value_spearman_corrected = np.zeros((total_nb_species, total_nb_species))
    spearman_rho = np.zeros((total_nb_species, total_nb_species))

    for index, (specie_1, specie_2) in enumerate(couple_species):
        print(index)

        ## Computation of Spearman coefficient for all pairs

        density_specie_1 = steady_state_densities[local_comm_species == specie_1]  # We obtain the density of specie_1
        # for each local community in an array of length nb_local_community
        density_specie_2 = steady_state_densities[local_comm_species == specie_2]

        rho, _p_value = stats.spearmanr(density_specie_1, density_specie_2)

        spearman_rho[specie_1, specie_2] = rho
        spearman_rho[specie_2, specie_1] = rho

        null_distrib_rho = np.zeros(nb_resampling)

        for resampling in xrange(nb_resampling):
            np.random.shuffle(density_specie_1)
            ## Computation of the Spearman coefficient for the null distribution
            null_distrib_rho[resampling], _p_value = stats.spearmanr(density_specie_1, density_specie_2)

        ## Computation of the p-value

        left_tail_event = len(null_distrib_rho[null_distrib_rho < rho]) / len(null_distrib_rho)
        right_tail_event = len(null_distrib_rho[null_distrib_rho >= rho]) / len(null_distrib_rho)

        p_value_spearman[index] = 2 * min(left_tail_event, right_tail_event)

        ## Correction for multiple comparison by Benjamini and Hochberg (1995):
        ## http://statsmodels.sourceforge.net/devel/generated/statsmodels.sandbox.stats.multicomp.multipletests.html
        # #statsmodels.sandbox.stats.multicomp.multipletests

    ## Computation of the correction for multiple comparison
    rejects, p_value_corrected, _alpha_1, _alpha_2 = statsmodels.sandbox.stats.multicomp.multipletests(p_value_spearman,
                                                                                                       method="fdr_bh")

    for index, (specie_1, specie_2) in enumerate(couple_species):
        p_value_spearman_corrected[specie_1, specie_2] = p_value_corrected[index]
        p_value_spearman_corrected[specie_2, specie_1] = p_value_corrected[index]

    return p_value_spearman_corrected, spearman_rho


def sensibility_sensitivity_analysis(co_occurrence_matrix, A):
    nb_false_pos = 0
    nb_false_neg = 0
    nb_true_pos = 0
    nb_true_neg = 0

    N = co_occurrence_matrix.shape[0]

    for i in xrange(N):
        for j in xrange(i+1, N):
            if A[i, j] > 0.:
                if co_occurrence_matrix[i, j] > 0.:
                    nb_true_pos += 1
                elif co_occurrence_matrix[i, j] == 0.:
                    nb_false_neg += 1

            elif A[i, j] < 0.:
                if co_occurrence_matrix[i, j] < 0.:
                    nb_true_pos += 1
                elif co_occurrence_matrix[i, j] == 0.:
                    nb_false_neg += 1

            elif A[i, j] == 0.:
                if co_occurrence_matrix[i, j] == 0.:
                    nb_true_neg += 1
                elif co_occurrence_matrix[i, j] != 0.:
                    nb_false_pos += 1
            else:
                raise ValueError

    return nb_true_pos, nb_true_neg, nb_false_pos, nb_false_neg