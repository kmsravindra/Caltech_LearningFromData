# -*- coding: utf-8 -*-
"""
Created on Sun May 21 10:57:46 2017

@author: Ravindra kompella
"""

import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np

fig, ax = plt.subplots(1)

# initializing the input space X with random uniform values between -1 and 1

def initializeX(examples):
    X = np.zeros([examples, 2])
    X[:,0] = np.random.uniform(-1, 1, examples)
    X[:,1] = np.random.uniform(-1, 1, examples)
    return X

X = initializeX(10)
print('this is input X', X)


# initializing a line to be drawn. This forms the target function for PLA
x1,y1,x2,y2 = np.random.uniform(-1,1,4)
line_eqn = lambda x : ((y2-y1)/(x2-x1)) * (x - x1) + y1
xrange = np.arange(-1,1,0.1)
ax.plot(xrange, [line_eqn(x) for x in xrange], color='k', linestyle='-', linewidth=2)

# determine if y=1 or y =-1 depending on the x in X above/below the line
def initializeY(X):
    y = np.zeros([10,1])
    [m,n] = X.shape
    print(X.shape)
    
    for i in range(0,m):
        px = X[i,0]
        py = X[i,1]
        
        if (px*(y2-y1)-py*(x2-x1)>(x1*y2-x2*y1)):
            ax.scatter(X[i,0],X[i,1], marker = 'o')
            y[i] = 1
            print("point x above", X[i,0])
            print("point y above", X[i,1])
        else:
            ax.scatter(X[i,0],X[i,1], marker = 'x')
            y[i] = -1
            print("point x below", X[i,0])
            print("point y below", X[i,1])
    
    return y

y = initializeY(X)
print('this is target y', y)

# initialize weight vector
initW = np.zeros([1,X.shape[1]-1])

# add bias column to X
v = np.ones([X.shape[0],1])
X = np.c_[v,X]

# determine sign of a given value
def sign(a):
    if a > 0:
        return 1
    else:
        return -1

# determine final weights
def computeWeights(iter,w):   
    for n in range(0,iter):
        # find the index of the row where hypothesis doesnt match target output.
        # This gives the index of the first misclassified point by hypothesis.    
        er=np.where(np.all(y!= h, axis=1))
        if er[0].size:
            w = w + y[er[0]][0]*X[er[0],:][0]
            Hyp = np.dot(X,np.transpose(w))
            for i in range(0,X.shape[0]):
                h[i,0] = sign(Hyp[i,0])
        else:
            break
    return w

w = computeWeights(15,initW)

le = lambda x : (-w[0,0]-w[0,1]*x)/w[0,2]
xrange = np.arange(-1,1,0.1)
ax.plot(xrange, [le(x) for x in xrange], color='r', linestyle='-', linewidth=2)
plt.show()

print('This is target output', y)
print('This is the hypothesis', h)
print('These are final weights', w)