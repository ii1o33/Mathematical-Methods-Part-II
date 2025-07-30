# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:54:35 2024

@author: socce
"""



"""
GO TO: https://plos.figshare.com/articles/dataset/The_microarray_dataset_of_ovarian_cancer_in_csv_format_/13658802
THEN DOWNLOAD THE CSV FILE IN THIS CURRENT FOLDER AND NAME IT AS 'ovarianCancer.csv'
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read the data
data = pd.read_csv('ovarianCancer.csv')
data = np.array(data)
n1 = np.shape(data)[1]-1
result = data[:,n1]
data = np.delete(data, (n1), axis=1)
data = data.astype(np.float64)

#calculate svd
U, S, VT = np.linalg.svd(data, full_matrices=0)

#plot variance and cumulative value of variance 
fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax1.semilogy(S,'-o',color='k')
ax2 = fig1.add_subplot(122)
ax2.plot(np.cumsum(S)/np.sum(S),'-o',color='k')

#Scattered plot - dot product of the dataset with VT to reduce dimension to 3
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
for j in range(data.shape[0]):
    x = VT[0,:] @ data[j,:].T
    y = VT[1,:] @ data[j,:].T
    z = VT[2,:] @ data[j,:].T
    
    if result[j] == 'Cancer':
        ax.scatter(x,y,z,marker='x', color='r', s=50)
    else:
        ax.scatter(x,y,z,marker='o', color='b', s=50)
        
ax.view_init(25,60)




