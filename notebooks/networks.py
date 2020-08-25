from torch import nn
import torch.nn.functional as F

import torch

class Ridge:
    def __init__(self, alpha = 0, fit_intercept = True,):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        X = X.rename(None)
        y = y.rename(None).view(-1,1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1), X], dim = 1)
        # Solving Xw = y with Normal equations:
        # X^{T}Xw = X^{T}y 
        lhs = X.T @ X 
        rhs = X.T @ y
        if self.alpha == 0:
            self.w, _ = torch.lstsq(rhs, lhs)
        else:
            ridge = self.alpha*torch.eye(lhs.shape[0])
            self.w, _ = torch.lstsq(rhs, lhs + ridge)
    def predict(self, X):
        X = X.rename(None)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1), X], dim = 1)
        return X @ self.w