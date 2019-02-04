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

def M_step(sufficient_stats):
    return params

def log_liklihood(obs, params):
    return ll

def forward_pass(observations, params):

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
        f = mu_h + K @ (x - mu_x)
        half = (np.eye(K.shape[0])[None,:, :] - K @ C)
        F = half @ S_hh @ half.swapaxes(-2, -1) + K @ E @ K.swapaxes(-2, -1)

        return f, F


def init_params(X, U, h_dim):
    return A, B, C, D, E, d, e