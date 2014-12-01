# -*- coding: utf8 -*-

# In[1]:

from __future__ import division
import scipy.stats as stats
from scipy.integrate import odeint, ode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

sns.set(font="Liberation Sans")

# In[2]:

N = 400  # Number of distinct species
M = 100  # number of distinct species in the local community (M < N)

r = stats.uniform.rvs(loc=0, scale=1, size=M)  # growth rate, uniform distribution between 0 and 1, vector of size M
while np.any(r == 0.):
    print("0 value")
    for index, value in enumerate(r):
        if value == 0.:
            r[index] = stats.uniform.rvs(loc=0, scale=1)


# In[3]:

# k: carrying capacity
k_even = stats.beta.rvs(a=1, b=1, loc=0, scale=1, size=M) # uniform distribution
k_uneven = stats.beta.rvs(a=1, b=1.5, loc=0, scale=1, size=M) #uneven distribution


# In[131]:

# Scaling of carrying capacity k between 1 and 100
k_even = 1. + k_even * 100
k_uneven = 1. + k_uneven * 100


# In[7]:

### Interaction matrix A

## Random Erdös-Renyi model

p = 1 / (N * (N - 1))  # Probability that a link exist between two random nodes. Here, 2 interactions for each species in average

A_ER = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            A_ER[i][j] = 1
        else:
            if random.random() <= p:
                A_ER[i][j] = stats.uniform.rvs(loc=-1, scale=2)


# In[9]:

## Subsampling
# We construct 'nb_local_com' communities composed of 'M' species each, with the following constraint:
# Each local community has to share a fraction 'frac_shared' of species with the rest of the local communities.

nb_local_com = 300  # Number of local communities
frac_shared = 0.80  # fraction of species that need to be shared between each pair of local community.
#frac_shared * nb_local_com must be an integer
nb_common_species = int(frac_shared * M)

common_species_list = random.sample(xrange(N), nb_common_species)  # List of species that need to be shared between
# each local community. xrange(N): list of the species, nb_common_species: number of species to be chosen

local_comm_species = np.zeros((nb_local_com, M), dtype=int)  # Matrix representing the species (in the form of integers)
# chosen for each local population
local_comm_species[:, 0:nb_common_species] = common_species_list

remaining_species = [x for x in xrange(N) if x not in common_species_list]  # List of species that have not be chosen yet

for comm in xrange(nb_local_com):
    local_comm_species[comm, nb_common_species:M] = random.sample(remaining_species, M - nb_common_species)
    # We sample the rest of the species for each local community


# In[14]:

# Initial abundance of species x_0
x_0 = stats.uniform.rvs(loc=10, scale=90, size=(nb_local_com, M))  # Uniform distribution between 10 and 100

# In[21]:


def derivative(t0, x, A, r, k):
    return r * x * (1 - (np.dot(A, x) / k))


def integrate(M, x_0, A, k, r, t_start, t_end, t_step):
  equation = ode(derivative)
  equation.set_integrator('lsoda', nsteps=500, method='bdf')
  equation.set_initial_value(x_0, 0)  # initial x value, initial time value
  equation.set_f_params(A, r, k)

  ts = np.zeros(t_end - t_start + 1)
  x = np.zeros((M, t_end - t_start + 1))
  x[:, 0] = x_0

  time_index = 1
  while equation.successful() and equation.t < t_end:
    equation.integrate(equation.t + t_step)
    ts[time_index] = equation.t
    x[:, time_index] = equation.y
    time_index += 1

  return x


# In[22]:

### Test
## We extract from the A_ER matrix the A matrix corresponding only to the species present in local_comm_species[0],
# in order to speed up computation (it avoids unecessary calculus)


A = A_ER[:, local_comm_species[0, :]]
A = A[local_comm_species[0, :], :]  # We get the interaction matrix with only species present in the local population

# In[23]:

## Alternative integration method through the ode function

t_end = 200.
t_start = 0.
t_step = 1.

x = np.zeros((nb_local_com, M, t_end - t_start + 1))

for local_community_index in xrange(nb_local_com):
  x[local_community_index] = integrate(M, x_0[local_community_index], A, k_even, r, t_start, t_end, t_step)



plt.plot(x[0].transpose())


