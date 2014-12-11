# -*- coding: utf8 -*-

from __future__ import division
import graph
from Microbial_com_modeling import *

old_settings = np.seterr(all='raise')

N = 400  # Number of distinct species
M = 100  # number of distinct species in the local community (M < N)

r = stats.uniform.rvs(loc=0, scale=1, size=M)  # growth rate, uniform distribution between 0 and 1, vector of size M
while np.any(r == 0.):
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

p = 2 / (N - 1)  # Here, average of 2 interactions per species
# Probability that a link exist between two random nodes. Here, 2 interactions for each species in average

A_ER = graph.generate_random_graph(N, p)

NB_LOCAL_COMMUNITY = 100  # Number of local communities
FRACTION_SHARED = 0.80  # fraction of species that need to be shared between each pair of local community.
#FRACTION_SHARED * NB_LOCAL_COMMUNITY must be an integer
NB_COMMON_SPECIES = int(FRACTION_SHARED * M)

local_comm_species, common_species_list = subsample_local_pop(N, M, NB_LOCAL_COMMUNITY, NB_COMMON_SPECIES)

# Initial abundance of species x_0
x_0 = stats.uniform.rvs(loc=10, scale=90, size=(NB_LOCAL_COMMUNITY, M))  # Uniform distribution between 10 and 100
steady_state_densities = get_steady_state_densities(NB_LOCAL_COMMUNITY, M, local_comm_species, x_0, A_ER, k_even, r,
                                                    t_max=5000., t_min=0, ts=1.)

#np.save("densities", steady_state_densities)
#steady_state_densities = np.load("densities.npy")

# ### Computation of the correlation coefficient (Spearman rho here)

## list of all the possible couple of species present in the local communities
couple_species = [(specie_1, specie_2) for specie_1 in common_species_list
                  for specie_2 in common_species_list if specie_2 > specie_1]

p_value_spearman, spearman_rho = p_value_spearman(steady_state_densities, couple_species, N, local_comm_species)

# np.save("p_value", p_value_spearman)
# np.save("spearman_rho", spearman_rho)
# np.save("common_species_list", common_species_list)
# np.save("A_ER", A_ER)


# p_value_spearman = np.load("p_value.npy")
# spearman_rho = np.load("spearman_rho.npy")

np.save("spearman", spearman_rho)
np.save("p_value", p_value_spearman)

co_occurrence_matrix = np.copy(spearman_rho)
co_occurrence_matrix[p_value_spearman > 0.05] = 0.

# List of all species that are not in common_species_list
non_common = [specie for specie in xrange(N) if specie not in common_species_list]

A_refactored = np.copy(A_ER)

A_refactored[:, non_common] = 0.
A_refactored[non_common, :] = 0.


# We keep from the interaction matrix only the strongest interaction coefficient if two species both interact with
# each other, and we make the matrix symmetric

for i in xrange(N):
    for j in xrange(i+1, N):
        if A_refactored[i, j] != A_refactored[j, i]:
            if abs(A_refactored[j, i]) > abs(A_refactored[i, j]):
                A_refactored[i, j] = A_refactored[j, i]
            else:
                A_refactored[j, i] = A_refactored[i, j]


nb_true_pos, nb_true_neg, nb_false_pos, nb_false_neg = sensibility_sensitivity_analysis(co_occurrence_matrix, A_ER)

prompt = "Number of true positive: {}\nTrue negative: {}\nFalse positive: {}\nFalse negative: {}"

print(prompt.format(nb_true_pos, nb_true_neg, nb_false_pos, nb_false_neg))

np.save("co_occurence_matrix", co_occurrence_matrix)

plt.imshow(A_ER)
plt.colorbar()
plt.savefig("A_ER.svg")

plt.imshow(A_refactored)
plt.savefig("A_refactored.svg")

plt.imshow(co_occurrence_matrix)
plt.savefig("co_occurrence.svg")