'''
Created on May 1, 2013

@author: tdomhan

Baysian linear regression.

Depends on numpy (and optionally matplotlib for plotting).

'''

import numpy as np
from numpy import linalg as LA

from numpy.random import multivariate_normal


class BayesianLinearRegression:
  """
    Linear regression model: y = z beta[1] + beta[0]

    beta ~ N(0,Lambda)
    Lambda = I * lambda

    P(y|x,beta) ~ N(y|x.dot(beta),sigma**2)
  """
  def __init__(self, lamb=[20.,20.], beta_mu=[0.,0.], sigma=5, fit_intercept=True):
    """
      lamb: variance of the prior for each of the feature dimensions.
      beta_mu: the mean of the prior
      sigma: variance of the prediction error.

      TODO: don't expect the data dimension to be 2 for the default args
    """
    self.inv_lamb = 1./np.asarray(lamb)
    self.beta_mu  = np.asarray(beta_mu)

    self.sigma = sigma
    self.fit_intercept = fit_intercept

  def add_intercept(self, X):
    X_new = np.ones((X.shape[0],X.shape[1]+1), dtype=X.dtype)
    X_new[:,1:] = X[:,:]
    return X_new

  def fit_ml(self, X, y):
    "Fit a Maximum Likelihood estimate. (not Bayesian)"
    if self.fit_intercept:
      X = self.add_intercept(X)
    self.beta = LA.inv(X.T.dot(X)).dot(X.T.dot(y))

  def fit_map(self, X, y, full_posterior=True):
    """
      Fit a MAP estimate
    """
    if self.fit_intercept:
      X = self.add_intercept(X)
    sigma = self.sigma
    l = sigma**2 * np.diagflat(self.inv_lamb)
    s = LA.inv(X.T.dot(X)+l)
    b0 = sigma**2 * np.diagflat(self.inv_lamb).dot(self.beta_mu)

    self.beta = s.dot(X.T.dot(y) + b0)
    if full_posterior:
      self.Sigma = sigma * sigma * s


  def predict(self, X):
    if self.fit_intercept:
      X = self.add_intercept(X)
    return X.dot(self.beta)







