#!/usr/bin/env python3

from __future__ import print_function, division

import warnings

import numpy as np
import pandas as pd
import numpy.linalg as la

from gpid.utils import whiten


def gen_adjacency_matrix(N, M, p, q, r, mode):
    rng = np.random.default_rng()
    A = np.zeros((N, N))

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


def compute_imxy_1(cov, dm, dx, dy):
    """
    Compute the mutual information between M and (X, Y) using the conditional
    entropy formula:
        I(M ; (X, Y)) = h(M) - h(M | X, Y)
    """
    ret = whiten(cov, dm, dx, dy, ret_channel_params=True)
    sig_mxy, hx, hy, hxy, sigxy = ret
    imxy = la.slogdet(np.eye(dm) +
                      hxy.T @ la.solve(sigxy + 1e-7 * np.eye(*sigxy.shape), hxy)
                     )[1] / np.log(2) / 2
    return imxy


def compute_imxy_2(cov, dm, dx, dy):
    """
    Compute the mutual information between M and (X, Y) using the joint
    entropy formula:
        I(M ; (X, Y)) = h(M) + h(X, Y) - h(M, X, Y)
    """
    return 0.5 / np.log(2) * (la.slogdet(cov[:dm, :dm])[1]
                              + la.slogdet(cov[dm:, dm:])[1]
                              - la.slogdet(cov)[1])


def compute_bias(dm, dx, dy, n):
    """
    Compute the bias in the mutual information estimate based on the work of
    Cai et al. (J. Mult. Anal., 2015).

    This value needs to be subtracted from the mutual information estimate to
    recover the unbiased mutual information.
    """
    #bias = lambda d: d * (d + 1) / (4 * n * np.log(2))
    bias = lambda d: sum(np.log(1 - k / n)
                         for k in range(1, d+1)) / np.log(2) / 2
    return bias(dm) + bias(dx + dy) - bias(dm + dx + dy)


def debias(imxy, bias):
    """Remove bias while ensuring non-negativity."""

    return np.maximum(imxy - bias, 0)


if __name__ == '__main__':
    # Make warnings raise an error: used in the try-except block, when we get
    # a LinAlgWarning because of poorly conditioned matrices (happens when
    # sample size is too low for the given dimensionality).
    warnings.filterwarnings('error')

    M_vals = [10, 20, 50]
    #M_vals = [10, 20]
    modes = ['both_unique', 'fully_redundant', 'zero_synergy']

    pid_table = pd.DataFrame()

    config_cols = ['desc', 'sample_size', 'trial_id', 'dm', 'dx', 'dy', 'mode', 'M']
    value_cols = ['imxy_1', 'imxy_2', 'bias', 'gt_1', 'gt_2']

    p = 0.1
    q = 0.1
    r = 0

    for M in M_vals:
        print('M = %d' % M)
        #M = 20
        N = M * 3

        dm, dx, dy = M, M, M

        for mode in modes:
            print(mode, end=': ', flush=True)

            A, _ = gen_adjacency_matrix(N, M, p, q, r, mode)
            A = gauss_weighted_adj(A)
            #plt.matshow(A)
            #plt.show()

            if mode in ['both_unique', 'fully_redundant']:
                sigm = np.eye(dm)
                sigx_m = np.eye(dx)
                sigy_m = np.eye(dy)
                hx = A[:dm, dm:dm+dx].T
                if mode == 'fully_redundant':
                    hy = hx
                    sigw = 0.9 * np.eye(dx)   # Assumes dx == dy
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

            gt_1 = compute_imxy_1(cov, dm, dx, dy)
            gt_2 = compute_imxy_2(cov, dm, dx, dy)
            #try:
            #    gt_1 = compute_imxy_1(cov, dm, dx, dy)
            #except:
            #    gt_1 = np.nan
            #try:
            #    gt_2 = compute_imxy_2(cov, dm, dx, dy)
            #except:
            #    gt_2 = np.nan

            # Compute PID on samples of different sample sizes, each T times
            rng = np.random.default_rng()
            T = 100  # Number of trials
            sample_sizes = np.r_[100, 200, 300, 600, 1000]
            for sample_size in sample_sizes:
                print('%d' % sample_size, end=' ', flush=True)
                try:
                    bias = compute_bias(dm, dx, dy, sample_size)
                except:
                    bias = np.nan

                for i in range(T):
                    # Sample from the covariance matrix
                    z = rng.multivariate_normal(np.zeros(cov.shape[0]), cov,
                                                size=sample_size)
                    cov_hat = np.cov(z.T)
                    try:
                        imxy_1 = compute_imxy_1(cov_hat, dm, dx, dy)
                    except:  # Mainly to catch LinAlgWarning's
                        imxy_1 = np.nan
                    try:
                        imxy_2 = compute_imxy_2(cov_hat, dm, dx, dy)
                    except:
                        imxy_2 = np.nan

                    cols = config_cols + value_cols
                    vals = ['imxy_bias', sample_size, i, dm, dx, dy, mode, M]
                    vals.extend([imxy_1, imxy_2, bias, gt_1, gt_2])

                    row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
                    pid_table = pd.concat((pid_table, row), ignore_index=True)

            print()

        print()

    pid_table.to_pickle('../results/mutual_info_bias.pkl.gz')
    print(pid_table)
