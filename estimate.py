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
    A = sigo + ho@sigm@ho.T
    C = np.linalg.inv(sigi + hi@sigm@hi.T)
    B = t
    M = cp.bmat([[A, B], [B.T, C]])
    Ai = scipy.linalg.sqrtm(np.linalg.inv(sigo + ho@sigm@ho.T))
    sqrtm = scipy.linalg.sqrtm(sigm)
    objective = cp.Minimize(cp.norm(Ai@t@hi@sqrtm-Ai@ho@sqrtm, 'fro'))
    constraints = [M >> 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver='SCS', verbose=True,
                        alpha=1, max_iters=maxiter, eps=eps)
    t = t.value

    sigt = sigo + ho.dot(sigm).dot(ho.T) - t.dot(sigi +
                                                 hi.dot(sigm).dot(hi.T)).dot(t.T)

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
    t1 = (np.linalg.slogdet(sig2)[1] - np.linalg.slogdet(sig1)[1]) / np.log(2)
    t2 = h1.shape[0]
    #t3 = np.trace(np.linalg.inv(sig2).dot(sig1))
    t3 = np.trace(scipy.linalg.solve(sig2, sig1, assume_a='pos'))
    #t4 = np.trace(sigm.dot((h2-h1).T).dot(np.linalg.inv(sig2)).dot(h2-h1))
    t4 = np.trace(sigm @ (h2-h1).T @ scipy.linalg.solve(sig2, h2-h1, assume_a='pos'))
    return 0.5 * (t1 - t2 + t3 + t4)


def approx_pid(hx, hy, hxy, sigx, sigy, sigxy, covxy, sigm, maxiter=5000, eps=1e-10):
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

    # approximate channel X->Y
    t, sigt = heuristic_channel(dx, dy, hx, hy, sigx, sigy, sigm)
    # get deficiency of X w.r.t. Y
    defx = exp_mvar_kl(hy, sigy, t.dot(hx), t.dot(sigx).dot(t.T) + sigt, sigm)

    # approximate channel Y->X
    t, sigt = heuristic_channel(dy, dx, hy, hx, sigy, sigx, sigm,
                                maxiter=maxiter, eps=eps)
    # get deficiency of Y w.r.t. X
    defy = exp_mvar_kl(hx, sigx, t.dot(hy), t.dot(sigy).dot(t.T) + sigt, sigm)

    # compute PID
    ri = min(imx-defy, imy-defx)
    uix = imx-ri
    uiy = imy-ri
    si = imxy-uix-uiy-ri

    return imx, imy, imxy, defx, defy, uix, uiy, ri, si
