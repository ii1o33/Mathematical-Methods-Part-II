# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:36:23 2024

@author: socce
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math
import statsmodels.api as sm


#User input
n_sample = 30000
n_sample_best = 100000    #number of samples for the best sig trial

subsam = 10  #subsampling for the best sig trial
burn_in = 5000
x_init = -1

sig_set = [0.4, 2, 5]


#target pdf(up to the normalisation const)
p = lambda x: 3*np.exp(-pow(x,2)/2) + np.exp(-pow(x-4,2)/2)
nor = integrate.quad(p,-math.inf,math.inf)

x = np.linspace(-4,8,1000)
y_p = p(x)/nor[0]


#Calculation and plot for different standard deviation of proposal distribution
fig1, ((ax1,ax4),(ax2,ax5),(ax3,ax6)) = plt.subplots(3,2,figsize=(15,10))
fig1.tight_layout()
axs = [ax1,ax2,ax3,ax4,ax5,ax6]

for j in range(len(sig_set)):
    sig = sig_set[j]
    #Proposal distribution (normal)
    q1 = lambda x, x_given:  1/(np.sqrt(2*np.pi)*sig)*np.exp(pow(x-x_given,2)/((-2)*pow(sig,2)))
    
    
    #Main calculation of Metropolis_Hastings algorithm
    X=[x_init]
    for i in range(n_sample):
        X_star = np.random.normal(X[-1],sig)
        accept_ratio = p(X_star)*q1(X[-1],X_star)/(p(X[-1])*q1(X_star,X[-1]))
        u = np.random.uniform(0,1)
        if u <min(1,accept_ratio):
            X.append(X_star)
            
    X = X[burn_in:]
    axs[j].plot(x,y_p)
    axs[j].hist(X, density=True,bins=100)
    axs[j].set_title("sig = {}, # of samples = {}, burn-in = {}".format(sig_set[j],n_sample,burn_in))
    axs[j].set_ylabel("Probability")
    axs[j].set_xlabel("X")
    
    
    a = sm.tsa.acf(X)
    xx = np.linspace(0,len(a),len(a))
    axs[j+3].plot(xx,a)
    axs[j+3].set_title("autocorelation")
    axs[j+3].set_ylabel("Autocorrelation")
    axs[j+3].set_xlabel("lag")






#after testing parametes: sig, autocorrelation
#the best possible(?)
sig = 0.4

    
#Main calculation of Metropolis_Hastings algorithm
X=[x_init]

for i in range(n_sample_best):
    X_star = np.random.normal(X[-1],sig)
    accept_ratio = p(X_star)*q1(X[-1],X_star)/(p(X[-1])*q1(X_star,X[-1]))
    u = np.random.uniform(0,1)
    if u <min(1,accept_ratio):
        X.append(X_star)
            
X = X[burn_in:]
X = X[::subsam]
plt.figure(figsize=(12,8))
plt.plot(x,y_p)
plt.hist(X, density=True,bins=100)
plt.title("sig = {}, # of samples = {}, burn-in = {}, subsampling = {}".format(0.4,n_sample,burn_in,subsam))
plt.ylabel("Probability")
plt.xlabel("X")



