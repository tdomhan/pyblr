'''
Created on May 1, 2013

@author: tdomhan

Baysian linear regression.

'''

import numpy as np
from numpy import linalg as LA

from numpy.random import multivariate_normal

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm
from matplotlib.colors import LogNorm


sigma = 5.
lmbd = 20.


Xtest = np.loadtxt("../test-data/Xtest.txt")
Xtrain = np.loadtxt("../test-data/Xtrain.txt")

y_test = np.loadtxt("../test-data/Ytest.txt")
y_train = np.loadtxt("../test-data/Ytrain.txt")


n_samples = [2,10,100]


def beta_ml(X,y):
    """
        returns b_ml = max_b p(y|X,b)
    """
    return LA.inv(X.T.dot(X)).dot(X.T.dot(y))

def beta_map(X,y):
    """
        MAP linear regression
    """
    l = sigma*sigma/lmbd * np.eye(X.shape[1],X.shape[1])
    return LA.inv(X.T.dot(X)+l).dot(X.T.dot(y))

def beta_mu_sigma_posterior(X,y):
    """
        P(beta|D,sigma,lambda)

        returns mu and sigma of a multivariate gaussian posterior of beta.
    """
    l = sigma*sigma/lmbd * np.eye(X.shape[1],X.shape[1])
    Sigma = LA.inv(X.T.dot(X)+l)
    mu = Sigma.dot(X.T.dot(y))
    Sigma = sigma*sigma*Sigma
    return mu,Sigma

def beta_mu_sigma_veriational(X,y):
    """
        P(beta|D,sigma,lambda)

        returns mu and sigma of a multivariate gaussian posterior of beta.
    """
    l = sigma*sigma/lmbd * np.eye(X.shape[1],X.shape[1])
    SigmaOld = LA.inv(X.T.dot(X)+l)
    mu = SigmaOld.dot(X.T.dot(y))
    Sigma = sigma*sigma*LA.inv(np.eye(X.shape[1],X.shape[1])*(X.T.dot(X)+l))
    return mu,Sigma

def mse(X,y,beta):
    d = y-X.dot(beta)
    return 1/float(len(y)) * d.T.dot(d)

def mv_normal(x, mu, sigma):
    pass

errors = []

for n in n_samples:
    X = Xtrain[:n]
    y = y_train[:n]
    beta = beta_map(X,y)
    print "samples: %d" % n
    print beta
    
    errors.append(mse(Xtest,y_test,beta))

    fig = plt.figure(n)
    ax=plt.subplot(111)
    ax.set_xlim(0,1)
    ax.set_ylim(-10,40)
    ax.scatter(X[:,1],y)
    ax.plot([0, 1.0], [beta[0], beta[0]+1.0*beta[1]], color='k', linestyle='-', linewidth=1)
    plt.title("training cases: %d mse %f" % (n, mse(Xtest,y_test,beta)))
    
    
    
#plot of error vs n_samples
plt.figure(99)
plt.title("number of samples vs the MSE")
ax=plt.subplot(111)
ax.set_xlim(0,110)
#ax.set_ylim(-10,40)
plt.plot(n_samples,errors)
    
    

#for n in n_samples:
#    X_samples = Xtrain[:n]
#    y_samples = y_train[:n]
#    #mu,Sigma = beta_mu_sigma_posterior(X_samples,y_samples)
#    mu,Sigma = beta_mu_sigma_veriational(X_samples,y_samples)
#    
#    print "mu"
#    print mu
#    print "sigma"
#    print Sigma
#    
#    Sigma_inv = LA.inv(Sigma)
#    
#    plt.figure(n)
#    
#    delta = 0.1
#    #defining the grid
#    x = np.arange(0., 20.0, delta)
#    y = np.arange(0., 20.0, delta)
#    z = np.zeros((len(x),len(y)))
#    xx,yy = np.meshgrid(x, y)
#    q = np.log(np.power(LA.det(2 * np.pi * Sigma),-0.5))
#    for i in range(len(x)):
#        for j in range(len(y)):
#            # treat xv[i,j], yv[i,j]
#            beta = np.array([xx[i,j], yy[i,j]])
#            d = beta - mu
#            r = 1
#            # d is a row vector -> therefore the transpose is the other way around
#            r += -0.5 * d.dot(Sigma_inv).dot(d[:,np.newaxis])
#            z[i,j] = r
#            
#    cs = plt.contour(xx, yy, z, 20, colors='k')
#    #cs = plt.contourf(xx, yy, z,20,cmap=plt.cm.jet)#,norm = LogNorm()
#    plt.clabel(cs, inline=1, fontsize=10)
#    plt.scatter(mu[0],mu[1])
#    plt.xlabel("beta 0")
#    plt.ylabel("beta 1")
#    #print mu
#    plt.title('log posterior with %d samples' % n)
#    
#    plt.figure(n+1)
#    ax=plt.subplot(111)
#    ax.set_xlim(0,1)
#    ax.set_ylim(-10,40)
#    ax.scatter(X_samples[:,1],y_samples)
#    for i in range(0,10):
#        #draw beta:
#        beta = multivariate_normal(mu, Sigma)
#        ax.plot([0, 1.0], [beta[0], beta[0]+1.0*beta[1]], color='k', linestyle='-', linewidth=1)
#    plt.title("training cases: %d" % n)
#    
#    print "done with %d" % n
mlmmmm
#    
plt.show()
    


