import numpy as np
import matplotlib.pyplot as plt
# Gaussian kernel
def GaussianKernel(X1, X2, sig=1.):
    dist_sqs = np.sum(X1**2, axis=1).reshape([-1,1]) + \
        np.sum(X2**2, axis=1).reshape([1,-1]) - \
        2*np.matmul(X1, X2.T)
    K = np.exp(-.5*dist_sqs/sig**2)
    return K

# # Collection of functions
gp_sample_n = 50     # number of functions
xs = np.linspace(0, 5, gp_sample_n).reshape([-1,1])

# Posterior function generation
tr_xs = np.array([[.2, .6, 2.7, 3.5, 4.7]]).T
tr_ys = np.array([[.95, .7, 1.11, 1.1, 0.7]]).T

k = GaussianKernel(tr_xs, xs)  # covariances
K = GaussianKernel(tr_xs, tr_xs)
invK = np.linalg.inv(K)

m_fun = np.matmul(np.matmul(k.T, invK), tr_ys).T[0]
k_fun = GaussianKernel(xs, xs) - np.matmul(np.matmul(k.T, invK), k)

ys = np.random.multivariate_normal(m_fun, k_fun, gp_sample_n)

# plt.scatter(tr_xs, tr_ys, s=1000)
for i in range(gp_sample_n):
    plt.plot(xs.T[0], ys[i], alpha=.3, c='k')
plt.scatter(tr_xs, tr_ys, s=30, c='k', zorder=5)
plt.show()


# var_test = np.diag(k_fun)

# plt.scatter(tr_xs, tr_ys)
# plt.plot(xs, m_fun)

# plt.fill_between(
#     xs.ravel(),
#     m_fun.ravel() - np.sqrt(var_test),
#     m_fun.ravel() + np.sqrt(var_test),
#     alpha=0.5
# )
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()