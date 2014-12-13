# -*- coding: utf8 -*-

from __future__ import division
import graph
import Microbial_com_modeling as mcm
import multiprocess
import numpy as np
import scipy.stats as stats
import statsmodels
import multiprocessing as mp
import time
import logging
from utils import start_logging, save_json
import json
import matplotlib.pyplot as plt
import seaborn as sns

logger = start_logging()
old_settings = np.seterr(all='raise')
start_time = time.time()


def repeat_simulation(varying_parameter, value_range):

    specificity = np.zeros(len(value_range))
    sensitivity = np.zeros(len(value_range))

    json_dict = {"parameters": None, "specificity": None, "sensitivity": None,
                 "varying_parameter": varying_parameter, "value_range": value_range.tolist()}

    save_json(json_dict, varying_parameter)

    for i, parameter_value in enumerate(value_range):
        specificity[i], sensitivity[i], parameters = start_simulation(**{varying_parameter: parameter_value})

        if i % 3 == 0:
            time.sleep(30)

        with open("data/{}.json".format(varying_parameter)) as json_file:
            output = json.load(json_file)

        output["specificity"] = specificity.tolist()
        output["sensitivity"] = sensitivity.tolist()
        output["parameters"] = parameters

        save_json(output, varying_parameter)


def start_simulation(**kwargs):

    N = 200  # Number of distinct species
    M = 100  # number of distinct species in the local community (M < N)
    NB_LOCAL_COMMUNITY = 100  # Number of local communities
    FRACTION_SHARED = 0.80  # fraction of species that need to be shared between each pair of local community.
    p = 2 / (N - 1)  # Here, average of 2 interactions per species
    # Probability that a link exist between two random nodes. Here, 2 interactions for each species in average
    carrying_capacity_b = 1.
    graph_model = "ER"

    for arg, value in kwargs.items():
        if arg == "N":
            N = value
        elif arg == "M":
            M = value
        elif arg == "NB_LOCAL_COMMUNITY":
            NB_LOCAL_COMMUNITY = value
        elif arg == "FRACTION_SHARED":
            FRACTION_SHARED = value
        elif arg == "p":
            p = value
        elif arg == "carrying_capacity_b":
            carrying_capacity_b = value
        elif arg == "graph_model":
            graph_model = value
        else:
            raise ValueError("Unknown argument: {}".format(arg))

    NB_COMMON_SPECIES = int(FRACTION_SHARED * M)

    r = stats.uniform.rvs(loc=0, scale=1, size=M)  # growth rate, uniform distribution between 0 and 1, vector of size M
    while np.any(r == 0.):
        for index, value in enumerate(r):
            if value == 0.:
                r[index] = stats.uniform.rvs(loc=0, scale=1)

    # k: carrying capacity
    k = stats.beta.rvs(a=1, b=carrying_capacity_b, loc=0, scale=1, size=M)

    # Scaling of carrying capacity k between 1 and 100
    k = 1. + k * 100

    # Interaction matrix A
    ## Random Erdös-Renyi model

    if graph_model == "ER":
        A = graph.generate_random_graph(N, p)

    local_comm_species, common_species_list = mcm.subsample_local_pop(N, M, NB_LOCAL_COMMUNITY, NB_COMMON_SPECIES)

    # Initial abundance of species x_0
    x_0 = stats.uniform.rvs(loc=10, scale=90, size=(NB_LOCAL_COMMUNITY, M))  # Uniform distribution between 10 and 100
    steady_state_densities = mcm.get_steady_state_densities(NB_LOCAL_COMMUNITY, M, local_comm_species, x_0, A, k, r,
                                                            t_max=5000., t_min=0, ts=1.)


    # ### Computation of the correlation coefficient (Spearman rho here)

    ## list of all the possible couple of species present in the local communities
    couple_species = [(specie_1, specie_2) for specie_1 in common_species_list
                      for specie_2 in common_species_list if specie_2 > specie_1]


    if mp.cpu_count() > len(couple_species):
        nb_thread = len(couple_species)
    else:
        nb_thread = mp.cpu_count()

    couple_species_splitted = np.array_split(np.array(couple_species), nb_thread)

    logging.info("Starting multiprocessing with {} threads.".format(nb_thread))
    args = [steady_state_densities, None, local_comm_species]
    results = multiprocess.apply_async_with_callback(mcm.p_value_spearman, args, couple_species_splitted, 1, nb_thread)
    logging.info("Multiprocessing computations done.")

    p_value_spearman_list = []
    spearman_rho_list = []

    for index, result in enumerate(results):
        p_value_spearman_list += list(result[0])
        spearman_rho_list += list(result[1])


    ## Computation of the correction for multiple comparison
    rejects, p_value_corrected, _, __ = statsmodels.sandbox.stats.multicomp.multipletests(p_value_spearman_list,
                                                                                          method="fdr_bh")

    p_value_spearman, spearman_rho = mcm.fill_matrices(p_value_corrected, spearman_rho_list,
                                                       couple_species, N)

    co_occurrence_matrix = np.copy(spearman_rho)
    co_occurrence_matrix[p_value_spearman > 0.05] = 0.

    # List of all species that are not in common_species_list
    non_common = [specie for specie in xrange(N) if specie not in common_species_list]

    A_filtered = np.copy(A)

    A_filtered[:, non_common] = 0.
    A_filtered[non_common, :] = 0.

    # We keep from the interaction matrix only the strongest interaction coefficient if two species both interact with
    # each other, and we make the matrix symmetric

    for i in xrange(N):
        for j in xrange(i+1, N):
            if A_filtered[i, j] != A_filtered[j, i]:
                if abs(A_filtered[j, i]) > abs(A_filtered[i, j]):
                    A_filtered[i, j] = A_filtered[j, i]
                elif abs(A_filtered[j, i]) < abs(A_filtered[i, j]):
                    A_filtered[j, i] = A_filtered[i, j]
                elif abs(A_filtered[j, i]) == abs(A_filtered[i, j]):
                    A_filtered[i, j] = A_filtered[j, i]

    nb_true_pos, nb_true_neg, nb_false_pos, nb_false_neg = mcm.sensibility_sensitivity_analysis(co_occurrence_matrix,
                                                                                                A_filtered)

    sensitivity = nb_true_pos / (nb_true_pos + nb_false_neg)
    specificity = nb_true_neg / (nb_true_neg + nb_false_pos)

    prompt = "Number of true positive: {}\nTrue negative: {}\nFalse positive: {}\nFalse negative: {}"
    prompt += "\nSensitivity: {}\nSpecificity: {}"
    logging.info(prompt.format(nb_true_pos, nb_true_neg, nb_false_pos, nb_false_neg, sensitivity, specificity))
    logging.info("Elapsed time: {}".format(time.time() - start_time))

    saved_data = {}

    parameters = {"N": N, "M": M, "NB_LOCAL_COMMUNITY": NB_LOCAL_COMMUNITY,
                                "FRACTION_SHARED": FRACTION_SHARED, "p": p,
                                "carrying_capacity_b": carrying_capacity_b, "graph_model": graph_model}

    return specificity, sensitivity, parameters

    # plt.imshow(A_ER)
    # plt.colorbar()
    # plt.savefig("A_ER.svg")
    #
    # plt.imshow(A_filtered)
    # plt.savefig("A_filtered.svg")
    #
    # plt.imshow(co_occurrence_matrix)
    # plt.savefig("co_occurrence.svg")

    ## Save of important variables

    # np.save("densities", steady_state_densities)
    # np.save("A_ER", A_ER)
    # np.save("A_filtered", A_filtered)
    # np.save("common_species_list", common_species_list)
    # np.save("spearman_rho", spearman_rho)
    # np.save("p_value", p_value_spearman)
    # np.save("co_occurrence_matrix", co_occurrence_matrix)

    # steady_state_densities = np.load("densities.npy")
    # A_ER = np.load("A_ER.npy")
    # A_filtered = np.load("A_filtered.npy")
    # common_species_list = np.load("common_species_list.npy")
    # spearman_rho = np.load("spearman_rho.npy")
    # p_value_spearman = np.load("p_value.npy")
    # co_occurrence_matrix = np.load("co_occurrence_matrix.npy")


if __name__ == "__main__":
    varying_parameter = "NB_LOCAL_COMMUNITY"
    value_range = np.arange(10, 310, 10)
    repeat_simulation(varying_parameter, value_range)