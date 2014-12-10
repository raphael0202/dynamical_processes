import numpy as np
import scipy.stats as stats
import random


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