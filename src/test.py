import numpy as np
from model import filter, smooth, EM

from nose import with_setup


def test_vr():
    N = 10
    M = np.random.randn(N, 3, 4)
    x = np.random.randn(N, 4)

    prod = M @ x[..., np.newaxis]

    v1 = M[1] @ x[1]
    np.testing.assert_almost_equal(np.squeeze(prod[1]), v1)



def gen_data_for_filtering_and_smoothing():
    dim = 2
    T = 500
    mu_init =  np.ones((1,dim))
    S_init = np.eye(dim)
    A = np.eye(dim)
        # np.array([[np.cos(25), np.sin(25)], [-np.sin(25), np.cos(25)]])
    B = np.zeros_like(A)
    C = np.eye(dim)
    D = np.eye(dim)
    d = np.zeros(dim)
    E =  np.eye(dim)
    e = np.zeros(dim)

    params = (A, B, C, D, d, E, e, S_init, mu_init)

    def step(y, u):
        y = y @ A + u @ B + np.random.multivariate_normal(d, D)
        return y, y @ C + np.random.multivariate_normal(e, E)

    y = 10 * np.ones((1, dim))
    u = np.zeros((1, dim))
    X = np.empty((1, T, dim))
    Y = np.empty((1, T, dim))
    for t in range(T):
        y, x = step(y, u)
        X[:, t, :] = x
        Y[:, t, :] = y

    U = np.tile(u, (T, 1))[np.newaxis, ...]

    return X, U, Y, params

X, U, Y_true, params = gen_data_for_filtering_and_smoothing()


# def test_filter(X, U, params):
#     return filter(X, U, params)
#
# means, covs, loglik = test_filter(X, U, params)
# print(loglik)
#
import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(X[0, :, 0], X[0, :, 1],'+k')
# plt.plot(means[:, 0, 0], means[:, 0, 1], '+r')
# plt.plot(Y_true[0, :, 0], Y_true[0, :, 1], '+g')
# plt.title('filtering')
# plt.show()
#
# plt.figure()
# plt.plot(X[0, :, 0], 'k+')
# plt.plot(Y_true[0, :, 0], 'g+')
# plt.plot(means[:, :, 0], 'r+')
# plt.title('filtering')
# plt.show()

def test_smoother(X, U, params):
    return smooth(X, U, params)

#
means, covs, crsses, loglik = test_smoother(X, U, params)
print(loglik)

plt.figure()
plt.plot(X[0, :, 0], X[0, :, 1],'k+')
plt.plot(means[:, 0, 0], means[:, 0, 1], '+r')
plt.plot(Y_true[0, :, 0], Y_true[0, :, 1], '+g')
plt.title('smoothing')
plt.show()

plt.figure()
plt.plot(X[0, :, 0], 'k+')
plt.plot(Y_true[0, :, 0], 'g+')
plt.plot(means[:, :, 0], 'b+')
plt.title('smoothing')
plt.show()

#
# def callback(ll, params):
#     print(ll)
#
#
# learned_params = EM(params, X, U, callback=callback, tol=1e-1)
