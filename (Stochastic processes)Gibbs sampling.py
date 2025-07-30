# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:54:25 2024

@author: socce
"""

"""
Note this is a toy example where the target is bivariate normal, and conditional probabilities are simply normal distribution
Partially from: https://github.com/abdulfatir/sampling-methods-numpy/blob/master/Gibbs-Sampling.ipynb
and : https://numeryst.com/gibbs-sampling-an-introduction-with-python-implementation/
And see scipy.stats.multrivariate_normal: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA



#User input
n_sample = 200000
burn_in = [10,100,1000,10000]
Xinit = [0,0]

mu_x = 5          #Bivariate normal distribution properties
mu_y = 7
sigma_x = 4
sigma_y = 1
rho = 0.9


#Multivariate normal pdf (original pdf)
mu = np.array([[mu_x,mu_y]])
def multivariate_normal(X, mu, sig):  #where X = column vector of x1,x2,...,xn   (If X = [n,m], this function calculates pdf at M different points)
    sqrt_term = np.sqrt(2*np.pi*LA.det(sig))
    sig_inv = LA.inv(sig)
    X = X[:, None, :] - mu[None, :, :]
    return np.exp(-np.matmul(np.matmul(X,np.expand_dims(sig_inv,0)),(X.transpose(0,2,1)))/2)/sqrt_term   #np.matmul: matrix multiplication

sig = np.array([[pow(sigma_x,2),rho*sigma_x*sigma_y],[rho*sigma_x*sigma_y,pow(sigma_y,2)]])            
x = np.linspace(-15,15,1000)
X = np.array(np.meshgrid(x,x)).transpose(1,2,0)
X = np.reshape(X, [X.shape[0]*X.shape[1],-1])
z = multivariate_normal(X,mu,sig)



#Gibbs sampling
def gibbs_sampler(num_samples, mu_x, mu_y, sigma_x, sigma_y, rho, Xinit):
    # Initialize x and y to random values
    x = Xinit[0]
    y = Xinit[1]

    # Initialize arrays to store samples
    samples_x = np.zeros(num_samples)
    samples_y = np.zeros(num_samples)

    # Run Gibbs sampler
    for i in range(num_samples):
        # Sample from P(x|y)
        x = np.random.normal(mu_x + rho * (sigma_x / sigma_y) * (y - mu_y), np.sqrt((1 - rho ** 2) * sigma_x ** 2))

        # Sample from P(y|x)
        y = np.random.normal(mu_y + rho * (sigma_y / sigma_x) * (x - mu_x), np.sqrt((1 - rho ** 2) * sigma_y ** 2))

        # Store samples
        samples_x[i] = x
        samples_y[i] = y

    return samples_x, samples_y


samples_x, samples_y = gibbs_sampler(n_sample, mu_x, mu_y, sigma_x, sigma_y, rho, Xinit)

# Plot the samples
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,8))
axs = [ax1,ax2,ax3,ax4]
for i in range(4):
    unburnt_x = samples_x[burn_in[i]:] 
    unburnt_y = samples_y[burn_in[i]:]
    axs[i].scatter(unburnt_x,unburnt_y,s=5)
    axs[i].contour(x, x, z.squeeze().reshape([x.shape[0], -1]),10, cmap='cool')
    axs[i].set_title("burn-in period of {} samples".format(burn_in[i]))
    
