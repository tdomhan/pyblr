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
  def __init__(self, lamb=20,sigma=5):
    self.lamb = lamb
    self.sigma = sigma

  def fit_ml(self, X, y):
    "Fit a Maximum Likelihood estimate. (not Bayesian)"
    self.beta = LA.inv(X.T.dot(X)).dot(X.T.dot(y))

  def fit_map(self, X, y, full_posterior=True):
    """
      Fit a MAP estimate
    """
    sigma = self.sigma
    lmbd  = self.lamb
    l = sigma*sigma/lmbd * np.eye(X.shape[1],X.shape[1])
    s = LA.inv(X.T.dot(X)+l)
    self.beta = s.dot(X.T.dot(y))
    if full_posterior:
      self.Sigma = sigma * sigma * s


  def predict(self, X):
    return X.dot(self.beta)







