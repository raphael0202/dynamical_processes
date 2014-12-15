import numpy as np
import scipy.stats as stats
import random
import networkx as nx


def generate_random_graph(N, p):
    A_ER = np.zeros((N, N))
    for i in xrange(N):
        for j in xrange(N):
            if i == j:
                A_ER[i, j] = 1
            else:
                if random.random() <= p:
                    A_ER[i, j] = stats.uniform.rvs(loc=-1, scale=2)
    return A_ER


def generate_watts_strogatz_graph(N, p):

    k = int((N - 1) * p)
    graph_WS = nx.watts_strogatz_graph(N, k, p)
    A = np.array(nx.adjacency_matrix(graph_WS).todense(), dtype=float)

    for i in xrange(N):
        for j in xrange(N):
            if A[i, j] == 1. and i != j:
                A[i, j] = stats.uniform.rvs(loc=-1, scale=2)

    for diag in xrange(N):
        A[diag, diag] = 1.

    return A