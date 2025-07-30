# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:52:54 2024

@author: socce
"""

import numpy as np 
import matplotlib.pyplot as plt

# 1d random walk
def RW1d(nsteps=1000, p=0.5, stepsize=1):
    steps = [1*stepsize if np.random.rand()<p else -1*stepsize for i in range(nsteps)]
    y = np.cumsum(steps)
    x = list(range(len(y)))
    return x, list(y)

def plotRW1d(npath = 10, nsteps=1000, p=0.5, stepsize=1):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(1,1,1)
    for i in range(npath):
        x,y = RW1d(nsteps, p, stepsize)
        ax1.plot(x,y,label='path{}'.format(i+1))
    ax1.set_xlabel("time")
    ax1.set_ylabel('distance')
    ax1.set_title('Random Walk 1d')
    ax1.legend()




#2d randdom walk
def RW2d(nsteps=5000, p=0.5, xstepsize=1, ystepsize=1):
    x = [0]
    y = [0]
    for i in range(nsteps-1):
        x.append(x[i]+xstepsize if np.random.rand() < p else x[i]-xstepsize)
        y.append(y[i]+ystepsize if np.random.rand() < p else y[i]-ystepsize) 
    return x,y

def plotRW2d(npath = 3, nsteps=1000, p=0.5, xstepsize=1, ystepsize=1):
    fig = plt.figure(figsize=(8,5))
    ax1 = fig.add_subplot(1,1,1)
    for i in range(npath):
        x,y = RW2d(nsteps,p,xstepsize,ystepsize)
        ax1.plot(x,y,label='path{}'.format(i+1))
    ax1.set_xlabel('coordinate 1')
    ax1.set_ylabel('coordinate 2')
    ax1.set_title('Random Walk 2d')
    ax1.legend()
    


#3d randdom walk
def RW3d(nsteps=5000, p=0.5, xstepsize=1, ystepsize=1, zstepsize=1):
    x = [0]
    y = [0]
    z = [0]
    for i in range(nsteps-1):
        x.append(x[i]+xstepsize if np.random.rand() < p else x[i]-xstepsize)
        y.append(y[i]+ystepsize if np.random.rand() < p else y[i]-ystepsize)
        z.append(z[i]+zstepsize if np.random.rand() < p else z[i]-zstepsize)
    return x,y,z

def plotRW3d(npath = 1, nsteps=1000, p=0.5, xstepsize=1, ystepsize=1, zstepsize=1):
    fig = plt.figure(figsize=(8,5))
    ax1 = fig.add_subplot(1,1,1,projection='3d')
    for i in range(npath):
        x,y,z = RW3d(nsteps,p,xstepsize,ystepsize,zstepsize)
        ax1.plot(x,y,z,label='path{}'.format(i+1))
    ax1.set_xlabel('coordinate 1')
    ax1.set_ylabel('coordinate 2')
    ax1.set_zlabel('coordinate 3')
    ax1.set_title('Random Walk 3d')
    ax1.legend()
    
    
    
#1d Brownian motion
def BM1d(nsteps = 1000, dt=0.1, diffusionCoeff=1, drift = 0):
    x=[0]
    for i in range(nsteps):
        x.append(x[i]+np.random.normal(0,1)*np.sqrt(dt)*diffusionCoeff+drift*dt)
    return x

def plotBM1d(npath = 10, nsteps=1000,dt=0.1,diffusioncoeff=1,drift=0):
    fig = plt.figure(figsize=(8,5))
    ax1=fig.add_subplot(1,1,1)
    time = np.linspace(0,nsteps*dt,nsteps+1)
    for i in range(npath):
        x=BM1d(nsteps,dt,diffusioncoeff,drift)
        ax1.plot(time,x,label='path{}'.format(i+1))
    ax1.set_xlabel('time')
    ax1.set_ylabel('coordinate1')
    ax1.set_title('Brownian Motion 1d')
    ax1.legend()
    
    
    
#2d Brownian Motion
def BM2d(nsteps = 1000, dt=0.1, diffusionCoeff=1,xdrift=0,ydrift=0):
    x=[0]
    y=[0]
    for i in range(nsteps):
        x.append(x[i]+np.random.normal(0,1)*np.sqrt(dt)*diffusionCoeff+xdrift*dt)
        y.append(y[i]+np.random.normal(0,1)*np.sqrt(dt)*diffusionCoeff+ydrift*dt)
    return x,y

def plotBM2d(npath = 3, nsteps=1000,dt=0.1,diffusioncoeff=1,xdrift=0,ydrift=0):
    fig = plt.figure(figsize=(8,5))
    ax1=fig.add_subplot(1,1,1)
    for i in range(npath):
        x,y=BM2d(nsteps,dt,diffusioncoeff,xdrift,ydrift)
        ax1.plot(x,y,label='path{}'.format(i+1))
    ax1.set_xlabel('coordinate1')
    ax1.set_ylabel('coordinate2')
    ax1.set_title('Brownian Motion 2d')
    ax1.legend()
    
    
    
#3d Brownian Motion
def BM3d(nsteps = 1000, dt=0.1, diffusionCoeff=1,xdrift=0,ydrift=0,zdrift=0):
    x=[0]
    y=[0]
    z=[0]
    for i in range(nsteps):
        x.append(x[i]+np.random.normal(0,1)*np.sqrt(dt)*diffusionCoeff+xdrift*dt)
        y.append(y[i]+np.random.normal(0,1)*np.sqrt(dt)*diffusionCoeff+ydrift*dt)
        z.append(z[i]+np.random.normal(0,1)*np.sqrt(dt)*diffusionCoeff+zdrift*dt)
    return x,y,z

def plotBM3d(npath = 2, nsteps=1000,dt=0.1,diffusioncoeff=1,xdrift=0,ydrift=0,zdrift=0):
    fig = plt.figure(figsize=(8,5))
    ax1=fig.add_subplot(1,1,1,projection='3d')
    for i in range(npath):
        x,y,z=BM3d(nsteps,dt,diffusioncoeff,xdrift,ydrift,zdrift)
        ax1.plot(x,y,z,label='path{}'.format(i+1))
    ax1.set_xlabel('coordinate1')
    ax1.set_ylabel('coordinate2')
    ax1.set_zlabel('coordinate3')
    ax1.set_title('Brownian Motion 3d')
    ax1.legend()

plotRW1d()
plotRW2d()
plotRW3d()
plotBM1d(drift=0.1)
plotBM2d(ydrift=0.01)
plotBM3d(xdrift=2)
