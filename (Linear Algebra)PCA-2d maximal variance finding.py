# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:59:29 2024

@author: socce
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [16,8]


#Creating random data set with mean, rotation and stretch values known
xC = np.array([2,1])  #Centre of data (mean)
sig = np.array([2,0.5])   #Pincipal axes
theta = np.pi/3  #Rotate cloud by pi/3

R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])   #Rotation matix

nPoints = 10000 #Create 10,000 points
X = R @ np.diag(sig) @ np.random.randn(2,nPoints) + np.diag(xC) @ np.ones([2,nPoints])   #random data stretched and rotated. Then, translated to the mean.

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(X[0,:], X[1,:], '.', color='k')
ax1.grid()
plt.xlim((-6,8))
plt.ylim((-6,8))



#Find Principal components
Xavg = np.mean(X,axis=1)       #compute the mean
B = X - np.tile(Xavg,(nPoints,1)).T  #Mean-subtracted data
U,S,VT = np.linalg.svd(B/np.sqrt(nPoints),full_matrices=0)

#Plot random data to overlay PCA
ax2 = fig.add_subplot(122)
ax2.plot(X[0,:],X[1,:],'.',color='k')   #plot data to overlay PCA
ax2.grid()
plt.xlim((-6,8))
plt.ylim((-6,8))

#Plot principal components and elopsoid of 1,2,3 Standard devidations
theta = 2*np.pi*np.arange(0,1,0.01)
Xstd = U @ np.diag(S) @ np.array([np.cos(theta),np.sin(theta)])

ax2.plot(Xavg[0] + Xstd[0,:], Xavg[1] + Xstd[1,:],'-',color='r',linestyle='solid')
ax2.plot(Xavg[0] + 2*Xstd[0,:], Xavg[1] + 2*Xstd[1,:],'-',color='r',linestyle='solid')
ax2.plot(Xavg[0] + 3*Xstd[0,:], Xavg[1] + 3*Xstd[1,:],'-',color='r',linestyle='solid')


ax2.plot(np.array([Xavg[0], Xavg[0]+U[0,0]*S[0]]),
         np.array([Xavg[1], Xavg[1]+U[1,0]*S[0]]),'-',color='cyan',linestyle='solid')
ax2.plot(np.array([Xavg[0], Xavg[0]+U[0,1]*S[1]]),
         np.array([Xavg[1], Xavg[1]+U[1,1]*S[1]]),'-',color='cyan',linestyle='solid')

plt.show

if np.shape(X)[0] < 3:
    dominantDIM = 2
else:
    dominantDIM = 3
for i in range(dominantDIM):
    print(S[i])