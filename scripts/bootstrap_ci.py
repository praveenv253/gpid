#!/usr/bin/env python3

from __future__ import print_function, division

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as la

from gpid.generate import swap_x_and_y, merge_covs
from gpid.tilde_pid import exact_gauss_tilde_pid


def gen_adjacency_matrix(N, M, p, q, r, mode):
    rng = np.random.default_rng()
    A = np.zeros((N, N))

    # XXX: Need a mode which has lots of synergy
    # This could also potentially have a lot of UI_Y
    # Combine this with zero synergy which has UI_X and RI for bit of all

    if mode == 'both_unique':
        B = np.array([[1, 0, 2, 0],
                      [0, 1, 0, 2],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        subnet_sizes = np.array([M//2, M//2, M, M])
    elif mode == 'fully_redundant':
        B = np.array([[1, 2, 2],
                      [0, 1, 0],
                      [0, 0, 1]])
        subnet_sizes = np.array([M, M, M])
    elif mode == 'unique_plus_redundant':
        B = np.array([[1, 0, 0, 2, 0],
                      [0, 1, 0, 0, 2],
                      [0, 0, 1, 2, 2],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])
        subnet_sizes = np.array([3, 3, M-6, M, M])
    elif mode == 'zero_synergy':
        B = np.array([[1, 2, 0],
                      [0, 1, 2],
                      [0, 0, 1]])
        subnet_sizes = np.array([M, M, M])
    elif mode == 'high_synergy':
        B = np.array([[1, 2, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        subnet_sizes = np.array([M, M, M])
    else:
        raise ValueError('Unrecognized mode %s' % mode)

    assert subnet_sizes.sum() == N
    L = subnet_sizes.size
    for i in range(L):
        for j in range(L):
            subnet_margins = np.r_[0, np.cumsum(subnet_sizes)]
            Mi = subnet_margins[i]
            Ki = subnet_margins[i + 1]
            Mj = subnet_margins[j]
            Kj = subnet_margins[j + 1]

            m = subnet_sizes[i]
            k = subnet_sizes[j]

            if B[i, j] == 1:
                A[Mi : Ki, Mj : Kj] = (rng.uniform(size=(m, k)) <= p).astype(int)
            elif B[i, j] == 2:
                A[Mi : Ki, Mj : Kj] = (rng.uniform(size=(m, k)) <= q).astype(int)
            else:
                A[Mi : Ki, Mj : Kj] = (rng.uniform(size=(m, k)) <= r).astype(int)

    if mode == 'fully_redundant':
        A[:M, 2*M:] = A[:M, M:2*M]
    elif mode == 'unique_plus_redundant':
        A[M//2:M, 2*M:] = A[M//2:M, M:2*M]

    np.fill_diagonal(A, 0)

    return A, B


def gauss_weighted_adj(A):
    """Adds random gaussian weights to an adjacency matrix"""

    rng = np.random.default_rng()
    return A * rng.normal(0, 1, size=A.shape)


def gen_cov_matrix(N, M, p, q, r, mode):
    if mode == 'bit_of_all':
        cov1, dm1, dx1, dy1 = gen_cov_matrix(N//2, M//2, p, q, r, mode='zero_synergy')
        cov1, dm1, dx1, dy1 = swap_x_and_y(cov1, dm1, dx1, dy1)
        cov2, dm2, dx2, dy2 = gen_cov_matrix(N//2, M//2, p, q, r, mode='high_synergy')

        return merge_covs(cov1, cov2, dm1, dx1, dy1, dm2, dx2, dy2, random_rotn=True)

    A, _ = gen_adjacency_matrix(N, M, p, q, r, mode)
    A = gauss_weighted_adj(A)
    #plt.matshow(A)
    #plt.show()

    dm, dx, dy = M, M, M

    if mode in ['both_unique', 'fully_redundant', 'high_synergy']:
        sigm = np.eye(dm)
        sigx_m = np.eye(dx)
        sigy_m = np.eye(dy)
        hx = A[:dm, dm:dm+dx].T
        if mode == 'fully_redundant':
            hy = hx
            sigw = 0.9 * np.eye(dx)   # Assumes dx == dy
        elif mode == 'high_synergy':
            hy = np.zeros((dy, dm))
            sigw = 0.8 * np.eye(dx)  # Assumes dx == dy
        else:
            hy = A[:dm, dm+dx:].T
            sigw = np.zeros((dx, dy))
        cov = np.block([[sigm, sigm @ hx.T, sigm @ hy.T],
                        [hx @ sigm, hx @ sigm @ hx.T + sigx_m, hx @ sigm @ hy.T + sigw],
                        [hy @ sigm, hy @ sigm @ hx.T + sigw.T, hy @ sigm @ hy.T + sigy_m]])

    elif mode == 'zero_synergy':
        hx = A[:dm, dm:dm+dx].T
        hyx = A[dm:dm+dx, dm+dx:].T
        hy = hyx @ hx
        sigm = np.eye(dm)
        sigx_m = np.eye(dx)
        covx = hx @ sigm @ hx.T + sigx_m
        sigy_x = np.eye(dx)
        cov = np.block([[sigm, sigm @ hx.T, sigm @ hy.T],
                        [hx @ sigm, covx, covx @ hyx.T],
                        [hy @ sigm, hyx @ covx, hyx @ covx @ hyx.T + sigy_x]])

    else:
        raise ValueError('Unrecognized mode %s' % mode)

    return cov, dm, dx, dy


if __name__ == '__main__':
    M_vals = [10, 20]
    modes = ['both_unique', 'fully_redundant', 'zero_synergy', 'high_synergy',
             'bit_of_all']

    T = 100            # Number of trials
    T_bootstrap = 100  # Number of bootstrap samples

    pid_table = pd.DataFrame()

    config_cols = ['desc', 'sample_size', 'trial_id', 'dm', 'dx', 'dy', 'mode',
                   'M', 'bootstrap']
    pid_cols = ['imxy', 'uix', 'uiy', 'ri', 'si']

    p = 0.1
    q = 0.1
    r = 0

    for M in M_vals:
        print('M = %d' % M)
        #M = 20
        N = M * 3

        for mode in modes:
            print(mode, end=': ', flush=True)

            cov, dm, dx, dy = gen_cov_matrix(N, M, p, q, r, mode)

            cols = [(col, '') for col in config_cols]
            vals = ['bootstrap_ci', None, None, dm, dx, dy, mode, M, None]

            ret = exact_gauss_tilde_pid(cov, dm, dx, dy)
            imxy, uix, uiy, ri, si = ret[2], *ret[-4:]

            cols.extend([('tilde', col) for col in pid_cols])
            vals.extend([imxy, uix, uiy, ri, si])

            row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
            pid_table = pd.concat((pid_table, row), ignore_index=True)

            # Compute PID on samples of different sample sizes, each T times
            rng = np.random.default_rng()
            sample_sizes = np.r_[100, 200, 300, 600, 1000]
            for sample_size in sample_sizes:
                print('%d' % sample_size, end='', flush=True)

                # Compute PID values for true confidence interval
                for i in range(T):
                    # Sample from the covariance matrix
                    z = rng.multivariate_normal(np.zeros(cov.shape[0]), cov,
                                                size=sample_size)
                    cov_hat = np.cov(z.T)

                    # Compute unbiased PID estimates
                    try:
                        ret = exact_gauss_tilde_pid(cov_hat, dm, dx, dy, unbiased=True,
                                                    sample_size=sample_size)
                        imxy, uix, uiy, ri, si = ret[2], *ret[-4:]
                    except:  # Mainly to catch LinAlgWarning's
                        imxy, uix, uiy, ri, si = [np.nan,] * 5

                    cols = [(col, '') for col in config_cols]
                    vals = ['bootstrap_ci', sample_size, i, dm, dx, dy, mode, M, False]

                    cols.extend([('tilde', col) for col in pid_cols])
                    vals.extend([imxy, uix, uiy, ri, si])

                    row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
                    pid_table = pd.concat((pid_table, row), ignore_index=True)

                print('.', end=' ', flush=True)

                # Sample a new random initialization and generate bootstrap
                # samples to compute the bootstrap confidence interval
                z = rng.multivariate_normal(np.zeros(cov.shape[0]), cov,
                                            size=sample_size)

                for i in range(T_bootstrap):
                    # Sample from the covariance matrix
                    bootstrap_indices = rng.choice(sample_size, size=sample_size,
                                                   replace=True)
                    eff_sample_size = np.unique(bootstrap_indices).size

                    z_bs = z[bootstrap_indices, :]
                    cov_hat = np.cov(z_bs.T)

                    # Compute unbiased PID estimates
                    try:
                        ret = exact_gauss_tilde_pid(cov_hat, dm, dx, dy, unbiased=True,
                                                    sample_size=eff_sample_size)
                        imxy, uix, uiy, ri, si = ret[2], *ret[-4:]
                    except:  # Mainly to catch LinAlgWarning's
                        imxy, uix, uiy, ri, si = [np.nan,] * 5

                    cols = [(col, '') for col in config_cols]
                    vals = ['bootstrap_ci', sample_size, i, dm, dx, dy, mode, M, True]

                    cols.extend([('tilde', col) for col in pid_cols])
                    vals.extend([imxy, uix, uiy, ri, si])

                    row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
                    pid_table = pd.concat((pid_table, row), ignore_index=True)

            print()

        print()

    pid_table.to_pickle('../results/bootstrap_ci_v3.pkl.gz')
    print(pid_table)
