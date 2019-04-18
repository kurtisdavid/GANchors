import torch
import torch.autograd as autograd
import pandas as pd
import numpy as np

# from this: https://github.com/activatedgeek/stein-gradient
class RBF(torch.nn.Module):
  def __init__(self, sigma=None):
    super(RBF, self).__init__()

    self.sigma = sigma

  def forward(self, X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

    # Apply the median heuristic (PyTorch does not give true median)
    if self.sigma is None:
      np_dnorm2 = dnorm2.detach().cpu().numpy()
      h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
      sigma = np.sqrt(h).item()
    else:
      sigma = self.sigma

    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()

    return K_XY

class SVGD:
  def __init__(self, P, K, optimizer):
    # reference distribution 
    self.P = P
    # kernel
    self.K = K
    # optimizer (for particles)
    self.optim = optimizer

  def phi(self, X):
    X = X.detach().requires_grad_(True)
    
    # assumes reference distribution can calculate log[p(X)]
    log_prob = self.P.log_prob(X)
    score_func = autograd.grad(log_prob.sum(), X)[0]

    K_XX = self.K(X, X.detach())
    grad_K = -autograd.grad(K_XX.sum(), X)[0]

    phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

    return phi

  def step(self, X):
    self.optim.zero_grad()
    # manually sets gradient
    X.grad = -self.phi(X)
    self.optim.step()
    
  def train(self, X, num_iter = 100):
    for i in range(num_iter):
        self.step(X)
    return X 
