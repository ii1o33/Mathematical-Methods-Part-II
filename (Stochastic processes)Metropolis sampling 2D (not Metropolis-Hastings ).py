# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:22:07 2024

@author: socce
"""

"""
Directly from:
https://github.com/abdulfatir/sampling-methods-numpy/blob/master/Metropolis-Hastings.ipynb
"""
#The important bit: For dimension > 1, sampling from bivariate conditional pdf!

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

n_sample = 100000



#The target density
def density1(z):
    z = np.reshape(z, [z.shape[0], 2])
    z1, z2 = z[:, 0], z[:, 1]
    norm = np.sqrt(z1 ** 2 + z2 ** 2)
    exp1 = np.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
    exp2 = np.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
    u = 0.5 * ((norm - 2) / 0.4) ** 2 - np.log(exp1 + exp2)
    return np.exp(-u)

#target density plotting
r = np.linspace(-5, 5, 1000)
z = np.array(np.meshgrid(r, r)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])

q0 = density1(z)
plt.figure(figsize=(12,8))
plt.hexbin(z[:,0], z[:,1], C=q0.squeeze(), cmap='rainbow')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.show()



#Metropois (not Metropolis-Hastings) algorithm for sampling
def metropolis_hastings(target_density, size=n_sample):
    burnin_size = 10000
    size += burnin_size
    x0 = np.array([[0, 0]])
    xt = x0
    samples = []
    for i in tqdm(range(size)):
        xt_candidate = np.array([np.random.multivariate_normal(xt[0], np.eye(2))])     #For dimension > 1, sampling from bivariate conditional pdf!
        accept_prob = (target_density(xt_candidate))/(target_density(xt))
        if np.random.uniform(0, 1) < accept_prob:
            xt = xt_candidate
        samples.append(xt)
    samples = np.array(samples[burnin_size:])
    samples = np.reshape(samples, [samples.shape[0], 2])
    return samples


#Metroplis sampling and plotting
plt.figure(figsize=(12,8))
samples = metropolis_hastings(density1)
plt.hexbin(samples[:,0], samples[:,1], cmap='rainbow')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.show()
