# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 01:16:22 2024

@author: socce
"""

import numpy as np
import matplotlib.pyplot as plt
x = 5 #True slope

a = np.linspace(-3,3,80)   #factors
b = a*x + np.random.randint(-3,7,a.shape)  #Adding noise to the outcome data

#Calculation
xtilde = (a.T@b)/pow(np.linalg.norm(a,2),2)   #xtilde calculated from SVD (xtilde = V*inv(S)*inv(U) where V = 1, S = norm(a,2) and  U = a/norm(a,2))

plt.figure()
plt.plot(a,b,'r.')
plt.plot(a,a*x,'k-')
plt.plot(a,a*xtilde,'g--')
plt.show