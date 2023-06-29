import numpy as np
import matplotlib.pyplot as plt

# Data generation
xs = np.array([-10, -5, 9, 15, 25])
ys = np.array([.95, 1, 1.11, 1.1, 1.05])
x_test = np.array([5.])

def k_fun(X_1, X_2, gamma = .005):
    return np.exp(-gamma*(
        np.diag(np.matmul(X_1, X_1.T)).reshape([-1,1]) + np.diag(np.matmul(X_2, X_2.T)).reshape([1,-1]) - \
        2*np.matmul(X_1, X_2.T)))

tr_num = len(xs)
K = k_fun(xs.reshape([tr_num,-1]), xs.reshape([tr_num,-1]))
k = k_fun(x_test.reshape([1,-1]), xs.reshape([-1,1]))
# K

y_test = np.matmul(np.matmul(k, np.linalg.inv(K)), ys.reshape([-1,1]))
# print(y_test)

test_num = 50
xs_test = np.linspace(-12, 30, num=test_num)
k = k_fun(xs_test.reshape([test_num,-1]), xs.reshape([tr_num,-1]))

ys_pred = np.matmul(np.matmul(k, np.linalg.inv(K)), ys.reshape([-1,1]))

k_test = k_fun(xs_test.reshape([test_num,-1]), xs_test.reshape([test_num,-1]))
sig_sq = .0005
var_test = sig_sq + np.diag(k_test) - np.diag(np.matmul(np.matmul(k, np.linalg.inv(K)), k.T))

plt.plot(xs_test, ys_pred)
plt.scatter(xs, ys)

plt.fill_between(
    xs_test.ravel(),
    ys_pred.ravel() - np.sqrt(var_test),
    ys_pred.ravel() + np.sqrt(var_test),
    alpha=0.5
)
plt.xlabel('x')
plt.ylabel('y')
plt.show()