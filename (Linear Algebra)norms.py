# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 22:21:38 2024

@author: socce
"""


"""For detailed explanation visit:
https://github.com/garth-wells/notebooks-3M1/blob/master/01-Fundamentals.ipynb
"""


import numpy as np
np.random.seed(2)

#Vector norms
x= np.random.rand(10) +1j*np.random.rand(10)                             #Generate random vector
for p in range(1,5):                                                     #1~4-norm of x
    x_norm = np.linalg.norm(x,p)
    print("The l_{} norm of x is : {}".format(p, x_norm))
l_inf = np.linalg.norm(x,np.inf)                                         #inf-norm of x
print("The l_inf norm of x is : {}".format(l_inf))


#matrix norm
A = np.random.rand(5,5) +1j*np.random.rand(5,5)                          #Generate random matrix
for p in range(1,3):                                                     #1~2-norm
    x_norm = np.linalg.norm(A,p)
    print("The l_{} norm of A is : {}".format(p, x_norm))
print("The l_inf norm os A is : {}".format(np.linalg.norm(A,np.inf)))    #inf-norm
print("The Frobenius norm of A is : {}".format(np.linalg.norm(A,'fro'))) #Frobenius norm (vector-like norm)