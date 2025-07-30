# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:12:16 2024

@author: socce
"""


"""For detailed explanation visit:
https://github.com/garth-wells/notebooks-3M1/blob/master/01-Fundamentals.ipynb
"""

import scipy.linalg as la
import numpy as np


#Effects of ill-conditioned matrix on perturbed system
H = la.hilbert(10)                    #Creater a Hilbert matix of size 10x10
print("Condition number of the Hilber Matrix is : {}".format(np.linalg.cond(H,2)))

"""The condtion number of the matrix is very large.
So, it is expected that the x of the perturbed system (A[x+dx] = b+db) is
 significantly different to the original x (Ax = b) as proven below.
"""

b = np.ones(H.shape[0])
b_delta = 1.0e-6*np.random.rand(H.shape[0])
b_perturbed = b + b_delta

x = np.linalg.solve(H,b)                        #Solving for original x
x_perturbed = np.linalg.solve(H,b_perturbed)    #Solvinf for perturbed x

error_x = np.linalg.norm(x_perturbed - x, 2)/np.linalg.norm(x,2)  #=(2-norm of dx)/(2-norm of x)
error_b = np.linalg.norm(b_delta, 2)/np.linalg.norm(b,2)          #=(2-norm of db)/(2-norm of b)
print("Relative error in x and b: {}, {}".format(error_x,error_b))

"""
As can be seen above, large condition number(ill-conditioned) allows
small relative error in b produce large relative error in x
"""
