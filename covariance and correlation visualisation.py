# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:51:22 2024

@author: socce
"""

import matplotlib.pyplot as plt
import numpy as np
import statistics

nPoints = 10000 #Create 10,000 points
slope = 2
randomness = 3


X = np.zeros((2,nPoints))
X[0] = np.linspace(-5, 5,nPoints)
X[1] = X[0]*slope
X = X + np.random.randn(2,nPoints)*randomness

#Plot random data to overlay PCA
ax2 = plt.figure()
plt.plot(X[0,:],X[1,:],'.',color='k')   #plot data to overlay PCA
plt.grid()
lim = 30
plt.xlim((-lim,lim))
plt.ylim((-lim,lim))

varx = statistics.variance(X[0])
vary = statistics.variance(X[1])

ans = np.cov(X)

corr = ans[0][1]/np.sqrt(varx*vary)
print("Correlation: {}".format(corr))
print("Covariance: {}".format(ans[0][1]))
print("Var_x: {}".format(ans[0][0]))
print("Var_y: {}".format(ans[1][1]))


