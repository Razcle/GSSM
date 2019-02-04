import numpy as np


def test_vr():
    N = 10
    M = np.random.randn(N, 3, 4)
    x = np.random.randn(N, 4)

    prod = M @ x[..., np.newaxis]

    v1 = M[1] @ x[1]
    print(v1, prod)

    np.testing.assert_almost_equal(np.squeeze(prod[1]), v1)


test_vr()