import numpy as np
from functools import partial


def EM(init_params, obs, acts, callback=None, tol=1e-3):

    def EM_update(params):
        means, covs, crsses, ll = smooth(obs, acts, params)
        if callback: callback(ll, params)
        return M_step(means, covs, crsses, obs, acts, params)

    def fixed_point(f, x0):
        x1 = f(x0)
        while different(x0, x1):
            x0, x1 = x1, f(x1)

        return x1

    def different(params1, params2):
        allclose = partial(np.allclose, atol=tol, rtol=tol)
        return not all(map(allclose, params1, params2))

    return fixed_point(EM_update, init_params)


def M_step(means, covs, crosses, obs, actions, params):
    """

    :param means: T x N x D array
    :param covs: T x N x D x D array
    :param crosses: T x N x D x D array
    :return: Params
    """

    N, T, dim = obs.shape
    obs = obs.swapaxes(0, 1)
    A, B, C, D, d, E, e, S_init, mu_init = params

    # distribution over initial hidden state
    mu_init = np.mean(means[0], axis=0, keepdims=True)
    S_init = covs[0] - mu_init.T @ mu_init

    # Transition Matrices
    A = np.sum(crosses[:-1], axis=0) @ np.linalg.inv(np.sum(covs[:-1], axis=0))
    C = np.sum(obs[..., np.newaxis] @ means.reshape((T, N, 1, dim)), axis=0)
    C = np.sum(C @  np.linalg.inv(np.sum(covs, axis=0)), axis=0)

    # Noise Matrices
    D = np.mean(covs[1:] - A @ crosses[:-1], axis=0)
    E = np.mean(obs[..., np.newaxis] @ obs.reshape((T, N, 1, dim)), axis=0)
    temp = (obs[..., np.newaxis] @ means.reshape((T, N, 1, dim))) @ C.swapaxes(-2, -1)
    E = np.mean(E - temp, axis=0)

    params = A, B, C, D, d, E, e, S_init, mu_init
    params = tuple((np.squeeze(param) for param in params))


    return params


def log_liklihood(mean, prec, obs):
    s, absZ = np.linalg.slogdet(prec)
    Z = 0.5 * s * absZ
    delta_x = obs - mean
    half = prec @ delta_x[:, :, np.newaxis]
    mahalob = - 0.5 * np.expand_dims(delta_x, 1) @ half
    return np.squeeze(Z + mahalob)


def filter(observations, actions, params):
    """ Computes the means and covariances on the latent
    states y_1:t given observations x_1:t-1

    :param observations: np array of shape (num_time_series, time_series length, time_series_dim)
    :param actions: np array of shape (num_time_series, time_series length, action_dim)
    :param params: tupple of np arrays
    :return: means and covariances for each time-step
    """
    N, T, dim = observations.shape
    A, B, C, D, d, E, e, S_init, mu_init = params

    # checkshapes(params, N, T , dim)

    def step(f,  F, x, u, ll):

        # find the means and covariances of the joint p(v_t, h_t|x_:t)
        mu_h = f @ A + u @ B + d
        mu_x = mu_h @ C + e
        S_hh = A @ F @ A.swapaxes(-2, -1) + D
        S_vv = C @ S_hh @ C.swapaxes(-2, -1) + E
        S_vv_inv = np.linalg.inv(S_vv)
        S_vh = C @ S_hh

        # Use Guassian conditioning to get the filtered posterior on h_t
        K = S_vh.swapaxes(-2, -1) @ S_vv_inv  # not the Kalman gain
        f = (mu_h + (K @ (x - mu_x)[:, :, np.newaxis])[:, :, 0])
        F = S_hh - S_vh.swapaxes(-2, -1) @ S_vv_inv @ S_vh

        # Calculate the probability of this observation
        ll = log_liklihood(mu_x, S_vv_inv, x)

        return f, F, ll

    means = np.empty((T, N, dim))
    covs = np.empty((T, N, dim, dim))
    f = mu_init
    F = S_init
    loglik = 0.0
    for t in range(T):
        f, F , ll = step(f, F, observations[:, t, :], actions[:, t, :], loglik)
        means[t] = f
        covs[t] = F
        loglik += ll

    return means, covs, loglik


def smooth(observations, actions, params):
    """

    :param observations: np array of shape (num_time_series, time_series length, time_series_dim)
    :param actions: np array of shape (num_time_series, time_series length, action_dim)
    :param params: tupple of np arrays
    :param means: list of filtered means
    :param covs: list of filtered covariances
    :return:
    """

    N, T, dim = observations.shape
    A, B, C, D, d, E, e, _, _ = params

    def backward_step(g, G, f, F, u):

        # Get moments of join to then condition
        mu = f @ A + u @ B + d
        S_hth = A @ F
        S_hh = S_hth @ A.swapaxes(-2, -1) + D

        # Set up backwards dynamics
        Shh_inv = np.linalg.inv(S_hh)
        S_back = F - S_hth.swapaxes(-2, -1) @ Shh_inv @ S_hth
        A_back = S_hth.swapaxes(-2, -1) @ Shh_inv
        m_back = f - (A_back @ mu[..., np.newaxis])[:,:,0]

        # Do the backwards updates
        g_new = (A_back @ g[..., np.newaxis])[:,:,0] + m_back
        G_new = A_back @ G @ A_back.swapaxes(-2, -1) + S_back
        crss = A_back @ G_new + g @ g_new.swapaxes(-2, -1)  # this cross moment is needed for learning during the M-step of EM

        return g_new, G_new, crss

    means, covs, loglik = filter(observations, actions, params)

    G = covs[-1]
    g = means[-1]
    new_means = np.empty_like(means)
    new_covs = np.empty_like(covs)
    crsses = np.empty_like(covs)
    for t in range(T):
        g, G, cs = backward_step(g, G, means[T-(t+1)], covs[T-(t+1)], actions[:, T-(t+1),:])
        new_means[T-(t+1)] = g
        new_covs[T-(t+1)] = G
        crsses[T-(t+1)] = cs

    return new_means, new_covs, crsses, loglik

def checkshapes(params, N, T, dim):
    A, B, C, D, d, E, e, S_init, mu_init = params
    assert(A.shape == (dim, dim))

def initialise_params(X, U, h_dim):
    return A, B, C, D, E, d, e