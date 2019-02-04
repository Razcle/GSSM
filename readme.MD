# GSSM

A very simple implementation of the EM algorithm for Gaussian State Space Models.

The implementation uses the Rauch-Tung-Streibel smoother with symmetrised updates in the Kalman filter
forward pass.

# Notation

The latent states are given by {h_{1...t}}, control inputs {u_{1..t}} and observations {x_{1..t}}

h_{t} = Ah_{t-1} + Bu_{t-1} + D

x_t = Ch_t + E

where D \sim N(d, D) and E \sim N(e, E)

# Usage

The api mirrors that of the scikitlean api with methods fit(X) and predict(x) but with additional
methods, filter(X) and smooth(X).

If you know the dynamics of your model a-priori (as is often the case in engineering) then you
must initialise the parameters A, B, C, D, E, d, e by hand. Otherwise the parameters are initialsed
automatically and learned using the EM algorithm.

e.g 

```
from src.model import GSSM
import numpy as np

my_time_series_data = np.load('data.npy')

model = GSSM(latent_dim=10)

post_ltnt_means, pst_ltnt_covs, mle_params = model.fit(my_time_series_data, control_inputs=None)

```


 
