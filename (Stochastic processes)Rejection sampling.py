# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:34:22 2024

@author: socce
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math


#User input
npoint = [70,100,1000,10000]  

M1 = 20
std1=2.4
mu1=0.8

M2 = 45
std2=4
mu2=-1.5



#target pdf
F = lambda x: 3*np.exp(-pow(x,2)/2) + np.exp(-pow(x-4,2)/2)
normalisationConst = integrate.quad(F,-math.inf,math.inf)[0]
#proposal pdf 1(normal)
q1 = lambda x: (1/(std1*np.sqrt(2*np.pi)))*np.exp(pow((x-mu1)/std1,2)/(-2)) 
#proposal pdf 2(noraml)
q2 = lambda x: (1/(std2*np.sqrt(2*np.pi)))*np.exp(pow((x-mu2)/std2,2)/(-2)) 



#The main rejection sampling algorithm and plotting
fig1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,8))
axs = [ax1,ax2,ax3,ax4]
x = np.linspace(-15,15,1000)
y=F(x)


for j in range(4):
    naccepted1 = 0
    accepted1 = []
    rv1 = np.random.normal(mu1,std1,npoint[j])
    naccepted2 = 0
    accepted2 = []
    rv2 = np.random.normal(mu2,std2,npoint[j])
    for i in range(npoint[j]):
        u = np.random.uniform(0,1)
        if u*M1*q1(rv1[i])<F(rv1[i]):
            accepted1.append(rv1[i])
            naccepted1 += 1
        if u*M2*q2(rv2[i])<F(rv2[i]):
            accepted2.append(rv2[i])
            naccepted2 += 1
    rejRate1 = (npoint[j]-naccepted1)/npoint[j]
    rejRate2 = (npoint[j]-naccepted2)/npoint[j]
    
    axs[j].hist(accepted1,density=True,bins=200, label='proposal1, rejection rate = {}'.format(rejRate1))
    axs[j].hist(accepted2,density=True,bins=200, label='proposal2, rejection rate = {}'.format(rejRate2))
    axs[j].plot(x,y/normalisationConst, label='Target pdf')
    axs[j].set_title('# of points = {}'.format(i+1))
    axs[j].legend()
    


#Plotting target and proposal pdfs to see the fittness of proposal pdfs
y1 = q1(x)
y2 = q2(x)
fig2 = plt.figure()
ax5=fig2.add_subplot(1,1,1)
ax5.plot(x,y,label='target pdf')
ax5.plot(x,y1*M1,label='M1*q1')
ax5.plot(x,y2*M2,label='M2*q2')
ax5.set_title("Target and proposal distributuions")
ax5.legend()
ax5.set_ylim([0,5])
ax5.set_xlim([-15,15])



