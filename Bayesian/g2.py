import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

def oscillator(d, w0, x):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    return y

np.random.seed(122)

x = torch.linspace(0,1,1000).view(-1,1)
d, w0=1.5, 15
y = oscillator(d, w0, x).view(-1,1)

# slice out a small number of points from the LHS of the domain
index=np.random.choice(np.arange(len(x)), replace=0, size=10)
x_data1 = x[index]
y_data1 = y[index]

# ## Gaussian kernel 2
def GaussianKernel(X1, X2, sig=1.):
    dist_sqs = np.sum(X1**2, axis=1).reshape([-1,1]) + \
        np.sum(X2**2, axis=1).reshape([1,-1]) - \
        2*np.matmul(X1, X2.T)
    K = np.exp(-.5*dist_sqs/sig**2)
    return K

# # Collection of functions
gp_sample_n = 100     # number of functions
xs = np.linspace(0, 1, gp_sample_n).reshape([-1,1])
sigma=.1

tr_xs = x_data1.numpy()#.T
tr_ys = y_data1.numpy()#.T

k = GaussianKernel(tr_xs, xs, sigma)  # covariances
K = GaussianKernel(tr_xs, tr_xs, sigma)
invK = np.linalg.inv(K)

m_fun = np.matmul(np.matmul(k.T, invK), tr_ys).T[0]
k_fun = GaussianKernel(xs, xs, sigma) - np.matmul(np.matmul(k.T, invK), k)

ys = np.random.multivariate_normal(m_fun, k_fun, gp_sample_n)

# plt.scatter(tr_xs, tr_ys, s=1000)
for i in range(gp_sample_n):
    plt.plot(xs.T[0], ys[i], alpha=.1, c='k')
plt.plot(xs.T[0], np.mean(ys, axis=0), c='r', linewidth=2, alpha=.5, label='mean')
plt.plot(x.cpu(), y.cpu(), c='b', linewidth=2, alpha=.5, label='solution')
plt.scatter(tr_xs, tr_ys, s=20, c='r', zorder=5, label='data')

plt.xlabel("Time")
plt.ylabel("Displacement")

plt.title(r"Bayesian (N=100, $\sigma$=0.1)")

plt.legend()
plt.savefig("bayesian", dpi=300)
# plt.show()

