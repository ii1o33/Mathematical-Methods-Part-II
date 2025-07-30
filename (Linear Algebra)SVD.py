# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 23:47:30 2024

@author: socce
"""
import numpy as np 
#User Input
X = np.random.randn(9, 6) + 1j*np.random.randn(9, 6) #create random data matrix
EconOption = False

#Calculation
n, m = X.shape
U, S, V = np.linalg.svd(X, full_matrices=not EconOption)   #Calculates the full SVD
if EconOption:
    Smat = np.zeros((m,m),dtype = complex)
else:
    Smat = np.zeros((n,m),dtype=complex)
Smat[:m,:m] = np.diag(S)
Xcal = U@Smat@V
residual = X - Xcal