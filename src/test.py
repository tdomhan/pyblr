"""
  Run some tests on the Bayesian Linear Regression.
"""

import blr
import numpy as np

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







