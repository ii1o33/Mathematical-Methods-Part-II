# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:20:19 2024

@author: socce
"""


import numpy as np
import scipy.linalg as LA




#Matrix creation
def create_mass_matrix(n):
    "Create a mass matrix of size n x n"
    A = np.zeros((n, n))
    m = np.array([[1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0]])
    for i in range(len(A) - 1):
        A[i:i+2, i:i+2] += m
    return A

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




#Iterative solvers
def Jacobi(A,b,tolerance,x='default'):   #Jacobi solver
    D = np.diag(np.diag(A))
    Dinv = np.diag(1.0/np.diag(A))
    R = A - D
    if x == 'default':
        x = np.zeros(A.shape[1])
    rnor = r0_norm = np.linalg.norm(b-A.dot(x),2)
    
    k = 0
    while rnor > tolerance:
        x = Dinv.dot(b - R.dot(x))
        r = b - A.dot(x)
        rnor = np.linalg.norm(r,2)/r0_norm
        k += 1
    return x, k

def GS(A,b,tolerance,x='default'):  #Gauss-Seidel solver
    L = np.tril(A)
    U = A - L
    if x == 'default':
        x = np.zeros(A.shape[1])
    rnor = r0_norm = np.linalg.norm(b-A.dot(x),2)
    
    k = 0
    while rnor > tolerance:
        x = np.linalg.inv(L).dot(b-U.dot(x))
        r = b-A.dot(x)
        rnor = np.linalg.norm(r,2)/r0_norm
        k += 1
    return x, k 

def SOR(A, b, convergence_criteria, x = 'default'):
    """
    This is an implementation of the pseudo-code provided in the Wikipedia article.
    Arguments:
        A: nxn numpy matrix.
        b: n dimensional numpy vector.
        omega: relaxation factor.
        x: An initial solution guess for the solver to start with.
        convergence_criteria: The maximum discrepancy acceptable to regard the current solution as fitting.
    Returns:
        phi: solution vector of dimension n.
    """
    evals, evecs = np.linalg.eig(A)
    maxEvalABS = abs(max(evals))
    omega = 2/(1+np.sqrt(1-pow(maxEvalABS,2)))
    if x == 'default':
        phi = np.zeros(A.shape[1])
    residual = np.linalg.norm(A @ phi - b)  # Initial residual
    
    k = 0
    while residual > convergence_criteria:
        for i in range(A.shape[0]):
            sigma = 0
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i, j] * phi[j]
            phi[i] = (1 - omega) * phi[i] + (omega / A[i, i]) * (b[i] - sigma)
        residual = np.linalg.norm(A @ phi - b)
        k += 1
    return phi, k

def cg(A,b,tolerance,x = 'default'):          #Conjugate gradient solver
    if x == 'default':
        x0 = np.zeros(A.shape[1])

    # Create starting vectors
    r0 = b - A.dot(x0)
    rnorm = r0_norm = np.linalg.norm(r0)
    p0 = r0.copy()

    k = 0
    # Start iterations    
    while rnorm > tolerance:
        alpha = r0.dot(r0)/(p0.dot(A.dot(p0)))
        x1 = x0 + alpha*p0
        r1 = r0 - alpha*A.dot(p0)
        rnorm = np.linalg.norm(r1,2)/r0_norm 
        beta = r1.dot(r1)/r0.dot(r0)
        p1 = r1 + beta*p0    
    
        # Update for next step
        p0, r0, x0 = p1, r1, x1
        k += 1
        
    return x0 , k

"""
def SOR(A,b,Niter, x='default'):   #Successive Over-Relaxation for Gauss-Seidel
    D = np.diag(np.diag(A))
    L = np.tril(A,-1)
    U = np.triu(A,1)
    evals, evecs = np.linalg.eig(A)
    maxEvalABS = abs(max(evals))
    w = 2/(1+np.sqrt(1-pow(maxEvalABS,2)))
    if x == 'default':
        x = np.zeros(A.shape[1])
    r0_norm = np.linalg.norm(b-A.dot(x),2)
    
    for k in range(Niter):
        x = np.linalg.inv(D+w*L)@((1-w)*D+w*U)@x + w*np.linalg.inv(D-w*L)@b
        r = b-A.dot(x)
        print("Step: {}, relative norm of residual (l2): {}".format(k+1,np.linalg.norm(r,2)/r0_norm))
    return x

def CG(A,b,Niter,x='default'):    #Conjugate Gradient method with preconditioning
    C = np.sqrt(np.diag(np.diag(A)))
    if x == 'default':
        x = np.ones(A.shape[1])*10
    r = b - A@x
    w = np.linalg.inv(C)@r
    v = np.linalg.inv(C.T)@w
    
    for k in range(Niter):
        tau = np.dot(w,w)/np.dot(v,A@v)
        x = x + tau*v
        r = r-tau*A@v
        wprev = w
        w = np.linalg.inv(C)@r
        s = np.dot(w,w)/np.dot(wprev,wprev)
        v = np.linalg.inv(C.T)@w + s*v
        
        
    return np.linalg.inv(C.T)@x
"""

#User input
tolerance = pow(10,-10)
A = create_mass_matrix(50)
b = np.ones(A.shape[1])

xexact = np.linalg.solve(A,b)

spectralRadius = max(np.linalg.eigvals(A))

xCG, kCG = cg(A,b,tolerance)
errorCG = np.linalg.norm(xCG-xexact,2)
if spectralRadius < 1:
    xJ, kJ = Jacobi(A, b, tolerance)
    errorJ = np.linalg.norm(xJ-xexact,2)
    xGS, kGS = GS(A, b, tolerance)
    errorGS = np.linalg.norm(xGS-xexact,2)
    xSOR, kSOR = SOR(A,b,tolerance)
    errorSOR = np.linalg.norm(xSOR-xexact,2)
    
    import tabulate as t
    table = [['Method', 'Jacobi', 'Gauss-Seidel', 'Successive Over-Relaxation for GS', 'Conjugate Gradient'], ['Error', errorJ, errorGS, errorSOR, errorCG], ['Number of iterations', kJ, kGS, kSOR, kCG]]
    print(t.tabulate(table, headers='firstrow'))   




