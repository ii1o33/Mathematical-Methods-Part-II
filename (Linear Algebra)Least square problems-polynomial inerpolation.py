# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:40:40 2024

@author: socce
"""

"""For detailed explanation visit:
https://github.com/garth-wells/notebooks-3M1/blob/master/01-Fundamentals.ipynb
"""
import matplotlib.pyplot as plt
import numpy as np

N = 30   #number of data points
noise = 0.1    #size of noise
x = np.linspace(-1,1,N)         #x positions of data points (equispaced)
x_clusteredEND = np.polynomial.legendre.leggauss(N)[0]  #x positions of data points (clustered ends) 


#Runge function
def runge(x):
    return 1 /(25 * (x**2) + 1)

#Noisy Runge function
def Nrunge(x, noise):
    return 1 /(25 * (x**2) + 1) + np.random.uniform(-noise, noise, len(x))

#Sine function
def sin(x):
    return np.sin(2*np.pi*x)

#Noisy Sine function
def Nsin(x,noise):
    return np.sin(2*np.pi*x) + np.random.uniform(-noise, noise, len(x))

#Polynomial fitting using monomial basis
def monomial(x,y,N):
    A = np.vander(x,N)
    c = np.linalg.solve((A.T).dot(A), (A.T).dot(y))
    p = np.poly1d(c)
    return p(x)

#Polynomial fitting using Legengre basis
def regendre(x,y,N):
    from numpy.polynomial import Polynomial as P 
    return P.fit(x,y,N)

#fig1 = plot of Runge function without noise
fig1 = plt.figure(figsize=(7,10), dpi=150)
ax11 = fig1.add_subplot(211)
ax11.plot(x,runge(x),'*', color='k', label='Original data points')
ax11.plot(x,monomial(x,runge(x),N),'-', color='r', label='monomial interpolation')
ax11.plot(*regendre(x,runge(x),N).linspace(200),'-', color='b', label='regendre interpolation')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Runge function - equispaced data points')
plt.legend()

ax12 = fig1.add_subplot(212)
ax12.plot(x_clusteredEND,runge(x_clusteredEND),'*', color='k', label='Original data points')
ax12.plot(x_clusteredEND,monomial(x_clusteredEND,runge(x_clusteredEND),N),'-', color='r', label='monomial interpolation')
ax12.plot(*regendre(x_clusteredEND,runge(x_clusteredEND),N).linspace(200),'-', color='b', label='regendre interpolation')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Runge function - clustered ends data points')
plt.legend()


#fig2 = plot of sine function without noise
fig2 = plt.figure(figsize=(7,10), dpi=150)
ax21 = fig2.add_subplot(211)
ax21.plot(x,sin(x),'*', color='k', label='Original data points')
ax21.plot(x,monomial(x,sin(x),N),'-', color='r', label='monomial interpolation')
ax21.plot(*regendre(x,sin(x),N).linspace(200),'-', color='b', label='regendre interpolation')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('sine function - equispaced data points')
plt.legend()

ax22 = fig2.add_subplot(212)
ax22.plot(x_clusteredEND,sin(x_clusteredEND),'*', color='k', label='Original data points')
ax22.plot(x_clusteredEND,monomial(x_clusteredEND,sin(x_clusteredEND),N),'-', color='r', label='monomial interpolation')
ax22.plot(*regendre(x_clusteredEND,sin(x_clusteredEND),N).linspace(200),'-', color='b', label='regendre interpolation')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('sine function - clustered ends data points')
plt.legend()


#fig3 = plot of Runge function with noise
fig3 = plt.figure(figsize=(7,10), dpi=150)
ax31 = fig3.add_subplot(211)
ax31.plot(x,runge(x),'*', color='k', label='Original data points')
ax31.plot(x,monomial(x,Nrunge(x,noise),N),'-', color='r', label='monomial interpolation')
ax31.plot(*regendre(x,Nrunge(x,noise),N).linspace(200),'-', color='b', label='regendre interpolation')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Runge function with noise - equispaced data points')
plt.legend()

ax32 = fig3.add_subplot(212)
ax32.plot(x_clusteredEND,Nrunge(x_clusteredEND, noise),'*', color='k', label='Original data points')
ax32.plot(x_clusteredEND,monomial(x_clusteredEND,Nrunge(x_clusteredEND, noise),N),'-', color='r', label='monomial interpolation')
ax32.plot(*regendre(x_clusteredEND,Nrunge(x_clusteredEND, noise),N).linspace(200),'-', color='b', label='regendre interpolation')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Runge function with noise - clustered ends data points')
plt.legend()


#fig4 = plot of sine function with noise
fig4 = plt.figure(figsize=(7,10), dpi=150)
ax41 = fig4.add_subplot(211)
ax41.plot(x,sin(x),'*', color='k', label='Original data points')
ax41.plot(x,monomial(x,Nsin(x,noise),N),'-', color='r', label='monomial interpolation')
ax41.plot(*regendre(x,Nsin(x,noise),N).linspace(200),'-', color='b', label='regendre interpolation')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('sine function with noise - equispaced data points')
plt.legend()

ax42 = fig4.add_subplot(212)
ax42.plot(x_clusteredEND,sin(x_clusteredEND),'*', color='k', label='Original data points')
ax42.plot(x_clusteredEND,monomial(x_clusteredEND,Nsin(x_clusteredEND, noise),N),'-', color='r', label='monomial interpolation')
ax42.plot(*regendre(x_clusteredEND,Nsin(x_clusteredEND, noise),N).linspace(200),'-', color='b', label='regendre interpolation')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('sine function with noise - clustered ends data points')
plt.legend()
