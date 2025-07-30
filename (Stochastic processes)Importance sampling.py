# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:07:14 2024

@author: socce
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math
import tabulate as t


#User input
scale = 12  #For scaling and plotting |f|p with normalised proposal distributions
npoint = [100,1000,3000,10000]
plot_on = False   #Whether to plot |f|p and q to see fittness of q chosen

std1=2.6
mu1=0.8

std2=5
mu2=-5



#target pdf (x~p, f=f(x): E_p(f(x)) = integral of f*p from -inf to inf)
#p = lambda x: 3*np.exp(-pow(x,2)/2) + np.exp(-pow(x-4,2)/2)
#f1 = lambda x: np.sin(x)
#f2 = lambda x: x/x
p_f1 = lambda x: (3*np.exp(-pow(x,2)/2) + np.exp(-pow(x-4,2)/2))*np.sin(x)
p_f2 = lambda x: (3*np.exp(-pow(x,2)/2) + np.exp(-pow(x-4,2)/2))
E_p_f1 = integrate.quad(p_f1,-math.inf,math.inf)[0]   #True but not exact solution for comparion
E_p_f2 = integrate.quad(p_f2,-math.inf,math.inf)[0]   #True but not exact solution for comparion

#proposal pdf 1(normal)
q1 = lambda x: (1/(std1*np.sqrt(2*np.pi)))*np.exp(pow((x-mu1)/std1,2)/(-2)) 
#proposal pdf 2(noraml)
q2 = lambda x: (1/(std2*np.sqrt(2*np.pi)))*np.exp(pow((x-mu2)/std2,2)/(-2)) 




E = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
#for the matrix above elements in order: "integrand = f1*p, proposal pdf = q1", "integrand = f1*p, proposal pdf = q2", "integrand = f2*p, proposal pdf = q1", "integrand = f2*p, proposal pdf = q2"
#Main algorithm
k = 0
for i in npoint:
    for j in range(i):
        x1 = np.random.normal(mu1,std1)
        x2 = np.random.normal(mu2,std2)
        E[0][k] += p_f1(x1)/q1(x1)
        E[1][k] += p_f1(x2)/q2(x2)
        E[2][k] += p_f2(x1)/q1(x1)
        E[3][k] += p_f2(x2)/q2(x2)
    E[0][k] = E[0][k]/j
    E[1][k] = E[1][k]/j
    E[2][k] = E[2][k]/j
    E[3][k] = E[3][k]/j
    k += 1
    

#Percentage error calculation
P_error = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
for j in range(4):
    for i in range(2):
        P_error[i][j] = abs(E[i][j]-E_p_f1)/abs(E_p_f1)
    for i in range(2,4):
        P_error[i][j] = abs(E[i][j]-E_p_f2)/abs(E_p_f2)
        
   
#Plot
if plot_on == True:
    x = np.linspace(-20,10,500)
    y_p_f1 = abs(p_f1(x))
    y_p_f2 = abs(p_f2(x))
    y_q1 = q1(x)
    y_q2 = q2(x)
    
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,y_p_f1/scale, label='|f1|p')
    ax.plot(x,y_p_f2/scale, label='|f2|p')
    ax.plot(x,y_q1, label='q1')
    ax.plot(x,y_q2, label='q2')
    ax.legend()
    ax.set_title("Check the fitness of q: good q is s.t q is high where |f|p is high")


#Output as a table
table = [["# points","percentage error: f1*p by q1", "percentage error: f1*p by q2", "percentage error: f2*p by q1", "percentage error: f2*p by q2"]]
for i in range(4):
    append = [npoint[i]]
    for j in range(4):
        append.append(P_error[j][i])
    table.append(append)
print(t.tabulate(table, headers='firstrow',tablefmt="fancy_grid"))   



