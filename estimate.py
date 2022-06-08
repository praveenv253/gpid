#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Gabe Schamberg
Date: Jan 5, 2021
Description: Functions for approximating a Gaussian PID
"""

import numpy as np
import scipy
import cvxpy as cp
import warnings

from .utils import lin_tf_params_from_cov


def heuristic_channel(di, do, hi, ho, sigi, sigo, sigm,
                      verbose=False, maxiter=5000, eps=1e-10):
    """
    Approximate the deficiency-minimizing channel from a input to
    a output conditioned on M. For example, when considering
    \delta(M:X\Y), we are estimating a channel from Y to X, so we call
    Y the input and X the output

    Param
    -----------
    di : int
            dimension of input
    do : int
            dimension of output
    hi : np.ndarray [di,dm]
            channel gain matrix for M->input
    ho : np.ndarray [do,dm]
            channel gain matrix for M->output
    sigi : np.ndarray [di,di]
            channel covariance matrix for M->input
    sigo : np.ndarray [do,do]
            channel covariance matrix for M->output
    sigm : np.ndarray [dm,dm]
            covariance matrix for M
    verbose : bool
            whether or not to print

    Returns
    -------
    t : np.ndarray [do,di]
            channel gain matrix for input->output
    sigt : np.ndarray [do,do]
            channel covariance matrix for input->output
    """

    t = cp.Variable((do, di))
    A = sigo + ho @ sigm @ ho.T
    C = np.linalg.inv(sigi + hi @ sigm @ hi.T)
    B = t
    M = cp.bmat([[A, B], [B.T, C]])
    Ai = scipy.linalg.sqrtm(np.linalg.inv(sigo + ho @ sigm @ ho.T))
    #Ai = np.eye(do, do)
    sqrtm = scipy.linalg.sqrtm(sigm)
    objective = cp.Minimize(cp.norm(Ai @ t @ hi @ sqrtm - Ai @ ho @ sqrtm, 'fro'))
    constraints = [M >> 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver='SCS', verbose=verbose,
                        alpha=1, max_iters=maxiter, eps=eps)

    if 'unbounded' in prob.status or 'infeasible' in prob.status:
        raise ValueError('Convex problem was %s' % prob.status)
    if 'inaccurate' in prob.status:
        warnings.warn('Inaccurate solution')

    t = t.value

    #sigt = sigo + ho.dot(sigm).dot(ho.T) - t.dot(sigi + hi.dot(sigm).dot(hi.T)).dot(t.T)
    sigt = sigo + ho @ sigm @ ho.T - t @ (sigi + hi @ sigm @ hi.T) @ t.T

    w, v = np.linalg.eigh(sigt)
    sigt = v.dot(np.diag(w)).dot(v.T)
    wclip = w.clip(min=0)
    sigt = v.dot(np.diag(wclip)).dot(v.T)
    if verbose:
        if((wclip != w).any()):
            print('Negative eigenvalues:')
            print(w)

    return t, sigt


def exp_mvar_kl(h1, sig1, h2, sig2, sigm):
    """
    Compute the KL-divergence between two multivariate
    Gaussian channel outputs, taking an expectation w.r.t. the channel
    input

    Param
    -----------
    h1 : np.ndarray
            KLD first argument channel gain matrix
    sig1 : np.ndarray
            KLD first argument channel covariance matrix
    h2 : np.ndarray
            KLD second argument channel gain matrix
    sig2 : np.ndarray
            KLD secondt argument channel covariance matrix
    sigm : np.ndarray
            covariance matrix for channel input

    Returns
    -------
    expected kl divergence : float
    """
    #t1 = np.log(np.linalg.det(sig2)/np.linalg.det(sig1))
    t1 = np.linalg.slogdet(sig2)[1] - np.linalg.slogdet(sig1)[1]
    t2 = h1.shape[0]
    #t3 = np.trace(np.linalg.inv(sig2).dot(sig1))
    t3 = np.trace(scipy.linalg.solve(sig2, sig1, assume_a='pos'))
    #t4 = np.trace(sigm.dot((h2-h1).T).dot(np.linalg.inv(sig2)).dot(h2-h1))
    t4 = np.trace(sigm @ (h2-h1).T @ scipy.linalg.solve(sig2, h2-h1, assume_a='pos'))
    #print('\n\n\n%g\n\n\n' % (0.5 * (t1 + t3 + t4 - t2) / np.log(2)))
    return 0.5 * (t1 - t2 + t3 + t4) / np.log(2)


def approx_pid(hx, hy, hxy, sigx, sigy, sigxy, covxy, sigm, maxiter=5000, eps=1e-10, verbose=True, ret_t_sigt=False):
    """
    Approximate a PID using the outputs from the generate function

    Param
    -----------
    hx : np.ndarray [dx,dm]
            channel gain matrix for M->X (i.e. X|M has mean hx.dot(M))
    hy : np.ndarray [dy,dm]
            channel gain matrix for M->Y
    hxy : np.ndarray [dx+dy,m]
            channel gain matrix for M->(X,Y)
    sigx : np.ndarray [dx,dx]
            channel covariance matrix for M->X (i.e. X|M has cov sigx)
    sigy : np.ndarray [dy,dy]
            channel covariance matrix for M->Y
    sigxy : np.ndarray [dx+dy,dx+dy]
            channel covariance matrix for M->(X,Y)
    covxy : np.ndarray [dx+dy,dx+dy]
            covariance matrix for (X,Y)
    sigm : np.ndarray [dm,dm]
            covariance matrix for M

    Returns
    -------
    imx : float
            mutual information between M and X
    imy : float
            mutual information between M and Y
    imxy : float
            mutual information between M and (X,Y)
    defx : float
            deficiency of X w.r.t. Y
    defy : float
            deficiency of Y w.r.t. X
    uix : float
            unique information in X (about M, unique w.r.t Y)
    uiy : float
            unique information in Y
    ri : float
            redundant information in X and Y
    si : float
            synergistic information in X and Y
    """

    dx = hx.shape[0]
    dy = hy.shape[0]

    # compute mutual informations
    imx = exp_mvar_kl(hx, sigx, np.zeros_like(hx),
                      hx.dot(sigm).dot(hx.T) + sigx, sigm)
    imy = exp_mvar_kl(hy, sigy, np.zeros_like(hy),
                      hy.dot(sigm).dot(hy.T) + sigy, sigm)
    imxy = exp_mvar_kl(hxy, sigxy, np.zeros_like(hxy), covxy, sigm)

    # approximate channel Y-> X (unique info in X)
    t, sigt = heuristic_channel(dy, dx, hy, hx, sigy, sigx, sigm,
                                verbose=verbose, maxiter=maxiter, eps=eps)
    # get deficiency of Y w.r.t. X
    def_x_minus_y = exp_mvar_kl(hx, sigx, t @ hy, t @ sigy @ t.T + sigt, sigm)
    tx, sigtx = t.copy(), sigt.copy()

    # approximate channel X->Y
    t, sigt = heuristic_channel(dx, dy, hx, hy, sigx, sigy, sigm,
                                verbose=verbose, maxiter=maxiter, eps=eps)
    # get deficiency of X w.r.t. Y (unique info in Y)
    def_y_minus_x = exp_mvar_kl(hy, sigy, t @ hx, t @ sigx @ t.T + sigt, sigm)
    ty, sigty = t.copy(), sigt.copy()

    # compute PID
    ri = min(imx - def_x_minus_y, imy - def_y_minus_x)
    uix = imx - ri
    uiy = imy - ri
    si = imxy - uix - uiy - ri

    ret = (imx, imy, imxy, def_y_minus_x, def_x_minus_y, uix, uiy, ri, si)
    if ret_t_sigt:
        ret = (*ret, tx, sigtx, ty, sigty)

    return ret


def approx_pid_from_cov(cov, dm, dx, dy, verbose=False, ret_t_sigt=False):
    """
    Compute the approximate Gaussian PID from a covariance matrix.
    """

    params = lin_tf_params_from_cov(cov, dm, dx, dy)
    return approx_pid(*params, verbose=verbose, ret_t_sigt=ret_t_sigt)
