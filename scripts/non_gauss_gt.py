#!/usr/bin/env python3

from __future__ import print_function, division

import sys
import numpy as np
import pandas as pd

from gpid.tilde_pid import exact_gauss_tilde_pid
from mult_poisson_gt import compute_qstar_admui, pid


def binomial_marginal_system(n, alpha):
    rng = np.random.default_rng()
    m = rng.binomial(4, 0.5, size=(2, n))

    x = (rng.binomial(m[0], alpha)
         + rng.binomial(m[1], 0.5)
         + rng.binomial(2, 0.5, size=(n,)))
    y = (rng.binomial(m[0], 0.5)
         + rng.binomial(m[1], 0.5)
         + rng.binomial(2, 0.5, size=(n,)))

    return m, x, y


def poisson_marginal_system(n, alpha):
    rng = np.random.default_rng()
    m = rng.poisson(2.0, size=(2, n))

    x = (rng.binomial(m[0], alpha)
         + rng.binomial(m[1], 0.5)
         + rng.poisson(1.0, size=(n,)))
    y = (rng.binomial(m[0], 0.5)
         + rng.binomial(m[1], 0.5)
         + rng.poisson(1.0, size=(n,)))

    return m, x, y


def zero_inflation_channel(m, x, y, inflation_factor):
    """
    Pass m, x, and y through a Z-channel that occasionally zeroes out values.
    """
    rng = np.random.default_rng()

    m_zi = m * rng.binomial(1, 1-inflation_factor, size=m.shape)
    x_zi = x * rng.binomial(1, 1-inflation_factor, size=x.shape)
    y_zi = y * rng.binomial(1, 1-inflation_factor, size=y.shape)

    return m_zi, x_zi, y_zi


if __name__ == '__main__':
    bins = np.arange(9) - 0.5

    n = 1000000        # Sample size
    inflation_factor = 0.3
    alpha_vals = np.arange(11) / 10
    dm, dx, dy = 2, 1, 1
    #dm, dx, dy = 1, 1, 1

    pid_table = pd.DataFrame()

    config_cols = ['desc', 'id', 'dm', 'dx', 'dy', 'w_x1']
    pid_cols = ['imxy', 'uix', 'uiy', 'ri', 'si']

    systems = {
        'binom': binomial_marginal_system,
        'poiss': poisson_marginal_system,
        'binom_zi': (lambda n, alpha: zero_inflation_channel(*binomial_marginal_system(n, alpha),
                                                             inflation_factor)),
        'poiss_zi': (lambda n, alpha: zero_inflation_channel(*poisson_marginal_system(n, alpha),
                                                             inflation_factor)),
    }

    sysname = sys.argv[1]
    system = systems[sysname]

    i = int(sys.argv[2])
    alpha = alpha_vals[i]

    print(sysname, end=': ', flush=True)
    print(i, end='', flush=True)

    cols = [(col, '') for col in config_cols]
    vals = [sysname, i, dm, dx, dy, alpha]

    m, x, y = system(n, alpha)
    #m = m[0]
    mxy = np.vstack((m, x, y))

    # Compute ground truth

    p = np.histogramdd(mxy.T, bins=[bins,]*4)[0]
    p0 = p.sum(axis=(1, 2, 3))
    p1 = p.sum(axis=(0, 2, 3))
    p0_indices = np.where(p0 != 0)[0]
    p1_indices = np.where(p1 != 0)[0]
    assert (p0_indices == p1_indices).all()

    p = p[np.ix_(p0_indices, p1_indices)]

    #p = np.histogramdd(mxy.T, bins=[bins,]*3)[0]
    p /= p.sum()
    p += 1e-5     # Regularize zero-probability values
    p /= p.sum()

    #print(((mxy[0] == 0) & (mxy[1] == 0)).sum())
    #print(p.sum(axis=(0, 1, 2)))
    #import sys
    #sys.exit()

    s = p.shape
    p = p.reshape((s[0]**2, s[2], s[3]))
    qstar = compute_qstar_admui(p, p.shape[0], [], maxiter=500)
    uix, ri, si, imxy = pid(p, qstar)  # UI corresponds to X
    uiy = imxy - uix - ri - si

    cols.extend([('gt', col) for col in pid_cols])
    vals.extend([imxy, uix, uiy, ri, si])
    cols.append(('gt_exists', ''))
    vals.append(True)
    print('.', end=' ', flush=True)

    row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
    pid_table = pd.concat((pid_table, row), ignore_index=True)

    print()

    pid_table.to_pickle('../results/non_gauss_gt-%s-%d.pkl.gz' % (sysname, i))
