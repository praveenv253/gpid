#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np


def get_params():
    lamda_m = np.array([2., 2.])
    lamda_x = np.array([1.,])      # Noise to be added to X: shape (d_X,)
    lamda_y = np.array([1.,])

    w_x1_vals = np.arange(11) / 10
    w_x2 = 0.5
    w_y1 = 0.5
    w_y2 = 0.5

    return lamda_m, lamda_x, lamda_y, w_x1_vals, w_x2, w_y1, w_y2


def sample_mult_poisson(n, lamda_m, w_x, w_y, lamda_x, lamda_y):
    """
    XXX: not yet fully general - X and Y have no shared randomness except M.
    """

    d_M = lamda_m.size
    d_X = lamda_x.size
    d_Y = lamda_y.size

    rng = np.random.default_rng()
    m = rng.poisson(lam=lamda_m, size=(n, 2)).T

    x = np.zeros((d_X, n))
    for j in range(d_X):
        x[j, :] = (sum(rng.binomial(m[i], w_x[j, i]) for i in range(d_M))
                   + rng.poisson(lam=lamda_x[j], size=n))

    y = np.zeros((d_Y, n))
    for j in range(d_Y):
        y[j, :] = (sum(rng.binomial(m[i], w_y[j, i]) for i in range(d_M))
                   + rng.poisson(lam=lamda_y[j], size=n))

    return m, x, y


def poisson_dist(lamda, x):
    return (np.exp(-lamda[..., None]) * lamda[..., None] ** x
            / np.array([np.math.factorial(i) for i in x]))


def binomial_dist(n, p):
    x = np.arange(n + 1)
    return np.array([np.math.comb(n, i) * p**i * (1 - p)**(n - i) for i in x])


def mult_poisson_dist(lamda_m, w_x, w_y, lamda_x, lamda_y, D=None):
    """
    XXX: not yet fully general - X and Y have no shared randomness except M.
    """

    d_M = lamda_m.size
    d_X = lamda_x.size
    d_Y = lamda_y.size

    if D is None:
        D = int(np.rint(np.max(lamda_m) * 3))  # Size of domain
    #p = np.ones([D,] * (d_M + d_X + d_Y))

    d = np.arange(D)
    pm = poisson_dist(lamda_m, d)

    # XXX: Dimension-specific code starts here
    assert (d_M == 2 and d_X == 1 and d_Y == 1)

    pmm = pm[[0], :].T * pm[[1], :]
    pmmx = np.zeros((D, D, D))
    pmmy = np.zeros((D, D, D))

    for i in range(D):
        for j in range(D):
            pmmx[i, j, :] = np.convolve(
                np.convolve(poisson_dist(lamda_x, d).squeeze(),
                            binomial_dist(d[i], w_x[0, 0])),
                binomial_dist(d[j], w_x[0, 1])
            )[:D]
            pmmy[i, j, :] = np.convolve(
                np.convolve(poisson_dist(lamda_y, d).squeeze(),
                            binomial_dist(d[i], w_y[0, 0])),
                binomial_dist(d[j], w_y[0, 1])
            )[:D]
    #lamda_eff_x = (w_x[0, 0] * d[:, None]) + (w_x[0, 1] * d) + lamda_x[0]
    #lamda_eff_y = (w_y[0, 0] * d[:, None]) + (w_y[0, 1] * d) + lamda_y[0]
    #pmmx = poisson_dist(lamda_eff_x, d)
    #pmmy = poisson_dist(lamda_eff_y, d)

    p = pmm[:, :, None, None] * pmmx[:, :, :, None] * pmmy[:, :, None, :]
    p_sum = p.sum()
    if p_sum < 0.95:
        warnings.warn('Total probability of truncated distribution < 0.95: p = %g' % p_sum)
    p /= p_sum  # Renormalize
    return p
