import numpy as np


def EM(init_params, data, callback=None):
    pass


def E_step(params, X):
    """

    :param params: tuple of numpy arrays
    :param X: numpy array of shape (batch_size, num_steps, obs_dim)
    :return: sufficient stats needed for the M-STEP
    """
    return sufficient_stats


def M_step(means, covs, crosses):

    # distribution over initial hidden state
    mu_init = np.mean(np.sum(means[-1], axis=0), axis=0)
    S_init = covs[-1] - means[-1][..., np.newaxis] @ means[-1][:, np.newaxis, :]

    # Transition Matrices


    # Noise Matrices



    return mu_init, S_init


def log_liklihood(obs, params):
    return ll


def filter(observations, actions, params):
    """ Computes the means and covariances on the latent
    states y_1:t given observations x_1:t-1

    :param observations: np array of shape (num_time_series, time_series length, time_series_dim)
    :param actions: np array of shape (num_time_series, time_series length, action_dim)
    :param params: tupple of np arrays
    :return: means and covariances for each time-step
    """
    N, T, D = observations.shape
    A, B, C, D, E, d, e = params

    def step(f,  F, x, u):

        # find the means and covariances of the joint p(v_t, h_t|x_:t)
        mu_h = f @ A + u @ B + d
        mu_x = mu_h @ C + e
        S_hh = A @ F @ A.swapaxes(-2, -1) + D
        S_vv_inv = np.linalg.inv(B @ S_hh @ B.swapaxes(-2, -1) + E)
        S_vh = B @ S_hh

        # Use Guassian conditioning to get the filtered posterior on h_t
        K = S_vh.swapaxes(-2, -1) @ S_vv_inv  # Kalman gain
        f = mu_h + K @ (x - mu_x)[..., np.newaxis]
        half = (np.eye(K.shape[0]) - K @ C)
        F = half @ S_hh @ half.swapaxes(-2, -1) + K @ E @ K.swapaxes(-2, -1)

        return f, F

    means = []
    covs = []
    f = np.zeros((N, D))
    F = np.zeros((N, D, D))
    for t in range(T):
        f, F = step(f, F, observations[:, t, :], actions[:, t, :])
        means.append(f)
        covs.append(F)

    return means, covs


def smooth(observations, actions, params):
    """

    :param observations: np array of shape (num_time_series, time_series length, time_series_dim)
    :param actions: np array of shape (num_time_series, time_series length, action_dim)
    :param params: tupple of np arrays
    :param means: list of filtered means
    :param covs: list of filtered covariances
    :return:
    """

    N, T, D = observations.shape
    A, B, C, D, E, d, e = params

    def backward_step(g, G, f, F, u):

        # Get moments of join to then condition
        mu = f @ A + u @ B + d
        S_hth = A @ F
        S_hh = S_hth @ A.swapaxes(-2, -1) + D

        # Set up backwards dynamics
        Shh_inv = np.linalg.inv(S_hh)
        S_back = F - S_hth.swapaxes(-2, -1) @ Shh_inv @ S_hth
        A_back = S_hth.swapaxes(-2, -1) @ Shh_inv
        m_back = f - A_back @ mu[..., np.newaxis]

        # Do the backwards updates
        crss = A_back @ G + g @ g.swapaxes(-2, 1)  # this cross moment is needed for learning during the M-step of EM
        g_new = A_back @ g + m_back
        G_new = A_back @ G @ A_back.swapaxes(-2, -1) + S_back

        return g_new, G_new, crss

    means, covs = filter(observations, actions, params)

    G = covs[-1]
    g = means[-1]
    new_means = [g]
    new_covs = [G]
    crsses = []
    for t in range(T):
        g, G, cs = backward_step(g, G, means[-1-t], covs[-1-t], actions[:, -1-t,:])
        new_means.append(g)
        new_covs.append(G)
        crsses.append(cs)

    return new_means, new_covs, crsses



def init_params(X, U, h_dim):
    return A, B, C, D, E, d, e