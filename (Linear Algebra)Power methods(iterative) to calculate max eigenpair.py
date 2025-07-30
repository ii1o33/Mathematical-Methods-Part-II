# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:12:09 2024

@author: socce
"""

"""
explanation of how it works
https://velog.io/@cleansky/EVD-%EC%97%86%EC%9D%B4-Eigenvalue-%EC%B0%BE%EA%B8%B0-Power-Iteration-Power-Method
https://ergodic.ugr.es/cphys/LECCIONES/FORTRAN/power_method.pdf
"""


import numpy as np


def create_stiffness_matrix(n):
    "Create a stiffness matrix of size n x n"

    # Create a zero matrix
    A = np.zeros((n + 1, n + 1))

    # Add components of small matrix k to A
    k = np.array([[1.0, -1.0], [-1.0, 1.0]])
    for i in range(len(A) - 1):
        A[i:i+2, i:i+2] += k

    # We will remove the last row and column of the matrix. This will
    # make the matrix invetible (SPD)
    A = np.delete(A, -1, 0)    
    A = np.delete(A, -1, 1)   
    return A



def max_eigenpair(A, print_values=False):
    "Compute the eigenpair for the largest eigenvalue. Matrix A is assumed symmetric"
    
    # Compute eigenpairs
    evals, evecs = np.linalg.eig(A)

    # Get index of largest absolute eigenvalue
    index = np.argmax(np.abs(evals))

    # Get largest eigenvalue and corresponding eigenvector
    eval_max = evals[index]
    evec_max = evecs[:, index]/np.linalg.norm(evecs[:, index])

    if print_values:
        print("  Largest eigenvalue:        {}".format(eval_max))

        # Get second largest eigenvalue to compare to largest
        eval1 = evals[np.argsort(abs(evals))[-2]]
        print("  Second largest eigenvalue: {}".format(eval1))
    
    return eval_max, evec_max



# Create 10x10 matrix
A = create_stiffness_matrix(10)

# Get exact eigenpair to compute errors
lambda_max, u_max = max_eigenpair(A)

# Create random starting vector and normalise
np.random.seed(3)
u0 = np.random.rand(A.shape[1])
u0 = u0/np.linalg.norm(u0)

# Perform power iteration
for k in range(10):
    print("Step: {}".format(k))

    # Compute u_{k+1} = A u_{k}
    u1 = A.dot(u0)

    # Estimate eigenvalue (from scaling of each entry and by Rayleigh quotient)
    lambda_est   = np.divide(u1, u0)
    rayleigh_est = u1.dot(A.dot(u1))/(u1.dot(u1))

    # Normalise estimated eigenvector and assign to u0
    u0 = u1/np.linalg.norm(u1)

    # Print errors in eigenvalue
    print("  Relative errors")    
    print("    lambda (scaling):  {}".format(np.abs(lambda_max - np.average(lambda_est))/lambda_max))    
    print("    lambda (Rayleigh): {}".format(np.abs(lambda_max - rayleigh_est)/lambda_max))    

    # Get signs on eigenvectors (could be pointing in opposite directions) and print error
    dir0, dir_est = abs(u_max[0])/u_max[0], abs(u0[0])/u0[0]
    print("    u (l2):            {}".format(np.linalg.norm(dir0*u_max - dir_est*u0)))