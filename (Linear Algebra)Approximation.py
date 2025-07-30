# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:36:45 2024

@author: socce
"""


#User option
EconOption = False
filename = ('Trump.jpg')
approximations = (3,5,10,30,150)  #How many eigenmodes to approximate? (number of trials = odd)


import numpy as np 
import matplotlib.pyplot as plt
import cv2
X = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

#Calculation
n, m = X.shape
U, S, V = np.linalg.svd(X, full_matrices=not EconOption)   #Calculates the full SVD
if EconOption:
    Smat = np.zeros((m,m))
else:
    Smat = np.zeros((n,m))
Smat[:m,:m] = np.diag(S)

'''
j = 0
import matplotlib.pyplot as plt
for i in (3,10,30,150):
    Xapprox = U[:,0:i]@Smat[0:i, 0:i]@V[0:i, :]
    plt.figure(j+1)
    j += 1
    img = plt.imshow(Xapprox, cmap='gray')
    plt.axis('off')
    plt.title('i = ' + str(i) )
    plt.show()
'''

#Plot approximated images
plt.figure(figsize=(6,5), dpi=150)
j = 1
l = int(np.ceil(len(approximations)/2))
plt.subplot(l,l,1)
img = plt.imshow(X, cmap='gray')
plt.axis('off')
plt.title('original')
for i in (3,5,10,30,150): #Try odd number of approximations
    Xapprox = U[:,0:i]@Smat[0:i, 0:i]@V[0:i, :]
    plt.subplot(l,l,j+1)
    j += 1
    img = plt.imshow(Xapprox, cmap='gray')
    plt.axis('off')
    plt.title('i = ' + str(i) )  
plt.show()


#log plot and cumulative sum of singular values
plt.figure(figsize=(10, 6), dpi=100)
plt.subplot(2,1,1)
plt.yscale("log")
plt.plot(S)
plt.yscale("log")
plt.title('logplot of singular values')

Scum = np.cumsum(S)
Scum = Scum/Scum[-1]
plt.subplot(2,1,2)
plt.plot(Scum)
plt.title('Percentage cumulative sum of singular values')
plt.subplots_adjust(bottom=0.01, right=0.8, top=0.9)
plt.show()


'''    
u = m
Xapprox = U[:,0:u]@Smat[0:u, 0:u]@V[0:u, :]
plt.imshow(Xapprox, cmap='gray')
    
cv2.imshow('ImageCalculated', Xcal)# waits for user to press any key 
# (this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 
'''