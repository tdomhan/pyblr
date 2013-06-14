"""
  Run some tests on the Bayesian Linear Regression.
"""

import blr
import numpy as np
from numpy.random import multivariate_normal
from numpy import linalg as LA

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm
from matplotlib.colors import LogNorm

def mse(X,y,clf):
    d = y-clf.predict(X)
    return 1/float(len(y)) * d.T.dot(d)

#read some test data
Xtest = np.loadtxt("../test-data/Xtest.txt")
Xtrain = np.loadtxt("../test-data/Xtrain.txt")

y_test = np.loadtxt("../test-data/Ytest.txt")
y_train = np.loadtxt("../test-data/Ytrain.txt")

n_samples = [2,10,100]

def run_clf(clf):
  for n in n_samples:
      X = Xtrain[:n]
      y = y_train[:n]
      clf.fit_map(X,y)
      error_map = mse(Xtest,y_test,clf)
      clf.fit_ml(X,y)
      error_ml = mse(Xtest,y_test,clf)

      print "samples: %d, ML mse: %f MAP mse: %f" % (n,error_ml, error_map)


# test different initializations of the classifier:

clf = blr.BayesianLinearRegression(fit_intercept=False)
print "without arguments"
run_clf(clf)

print "vector arguments"
run_clf(clf)
clf = blr.BayesianLinearRegression(
        lamb=np.array([20.,20.]),
        sigma=5, fit_intercept=False,
        beta_mu=[0,0])

print "scalar arguments"
run_clf(clf)
clf = blr.BayesianLinearRegression(
        lamb=20,
        sigma=5, fit_intercept=False,
        beta_mu=0)

#let's do some interesting plots:


fig = plt.figure(figsize=(20,8))
for i,n in enumerate(n_samples):
    X = Xtrain[:n]
    y = y_train[:n]

    clf.fit_map(X,y)
    beta_map = clf.beta
    error_map = mse(Xtest,y_test,clf)
    Sigma = clf.Sigma
#    Sigma_inv = LA.inv(Sigma)

    clf.fit_ml(X,y)
    beta_ml = clf.beta
    error_ml = mse(Xtest,y_test,clf)

    ax=plt.subplot(1,3,i+1)
    ax.set_xlim(0,1)
    ax.set_ylim(-10,40)
   #draw from posterior:
    for i in range(0,100):#10000):
      beta = multivariate_normal(beta_map, Sigma)
      ax.plot([0, 1.0], [beta[0], beta[0]+1.0*beta[1]], color='k', linestyle='-', linewidth=1,zorder=1, alpha=0.03)
    ax.plot([0, 1.0], [beta_ml[0], beta_ml[0]+1.0*beta_ml[1]], linestyle='-', linewidth=2, label="ML",zorder=100)
    ax.plot([0, 1.0], [beta_map[0], beta_map[0]+1.0*beta_map[1]], linestyle='-', linewidth=2, label="MAP",zorder=100)

    ax.scatter(X[:,1],y, marker="o", color='r', edgecolor="white", s=100,zorder=10)
 
    plt.legend()
    plt.title("training cases: %d" % n)

fname = "ml_vs_map.png"
fig.savefig(fname)

#some plots of the beta posterior distribution:

fig = plt.figure(figsize=(20,8))
plt.clf()
for i,n in enumerate(n_samples):
    X = Xtrain[:n]
    y = y_train[:n]

    clf.fit_map(X,y)
    beta_map = clf.beta
    error_map = mse(Xtest,y_test,clf)
    Sigma = clf.Sigma
    Sigma_inv = LA.inv(Sigma)

    clf.fit_ml(X,y)
    beta_ml = clf.beta
    error_ml = mse(Xtest,y_test,clf)

    mu = beta_map
    ax=plt.subplot(1,3,i+1)
    delta = 0.1
    x = np.arange(0., 20.0, delta)
    y = np.arange(0., 20.0, delta)
    z = np.zeros((len(x),len(y)))
    xx,yy = np.meshgrid(x, y)
    q = np.log(np.power(LA.det(2 * np.pi * Sigma),-0.5))
    for i in range(len(x)):
      for j in range(len(y)):
        beta = np.array([xx[i,j], yy[i,j]])
        d = beta - mu
        r = 1
        # d is a row vector -> therefore the transpose is the other way around
        r += -0.5 * d.dot(Sigma_inv).dot(d[:,np.newaxis])
        z[i,j] = r
    #cs = plt.contour(xx, yy, z, 20, colors='k')
    #plt.clabel(cs, inline=1, fontsize=10)
    plt.pcolor(xx,yy,z,vmin=0.,vmax=1)
    plt.scatter(mu[0],mu[1],c='k')
    plt.xlabel("beta[0]")
    plt.ylabel("beta[1]")
    plt.xlim(0,20)
    plt.ylim(0,20)

    plt.title("beta posterior distribution")

fname = "posterior.png"
fig.savefig(fname)







