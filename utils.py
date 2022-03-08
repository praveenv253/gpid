#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import scipy.linalg as la


def lin_tf_params_from_cov(cov, dm, dx, dy):
    """
    Utility function to extract linear transform parameters from a covariance
    matrix.

    Parameters
    ----------
    cov : np.ndarray [dm+dx+dy, dm+dx+dy]
            Covariance matrix to extract parameters from
    dm : int
            Dimension of M
    dx : int
            Dimension of X
    dy : int
            Dimension of Y

    Returns
    -------
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
    """
    covm = cov[:dm, :dm]
    covx = cov[dm:dm+dx, dm:dm+dx]
    covy = cov[dm+dx:, dm+dx:]
    covxy = cov[dm:, dm:]
    covx_m = cov[:dm, dm:dm+dx].T
    covy_m = cov[:dm, dm+dx:].T
    covxy_m = cov[:dm, dm:].T

    # channel gain matrices (using multivariate conditional mean)
    # Assumes that covm is invertible
    hx = covx_m.dot(la.inv(covm))
    hy = covy_m.dot(la.inv(covm))
    hxy = covxy_m.dot(la.inv(covm))

    # channel covariance matrices (using mvar conditional covariance)
    sigm = covm
    sigx = covx - covx_m.dot(la.inv(covm)).dot(covx_m.T)
    sigy = covy - covy_m.dot(la.inv(covm)).dot(covy_m.T)
    sigxy = covxy - covxy_m.dot(la.inv(covm)).dot(covxy_m.T)

    # whiten
    # Assumes that sigx and sigy are invertible
    hx = la.sqrtm(la.inv(sigx)).dot(hx)
    sigx = np.eye(dx)
    hy = la.sqrtm(la.inv(sigy)).dot(hy)
    sigy = np.eye(dy)

    return hx, hy, hxy, sigx, sigy, sigxy, covxy, sigm


def remove_lin_dep_comps(cov, dm, dx, dy):
    # XXX: Untested
    """
    Utility function to remove linearly dependent components from the
    covariance matrix for M, in order to make it invertible.

    Parameters
    ----------
    cov : np.ndarray [dm+dx+dy, dm+dx+dy]
            Covariance matrix of M, X and Y
    dm : int
            Dimension of M
    dx : int
            Dimension of X
    dy : int
            Dimension of Y

    Returns
    -------
    cov_new : np.ndarray [dm+dx+dy, dm+dx+dy]
            New joint covariance matrix for M, X and Y
    dm_new : int
            New dimension of M
    """

    covm = cov[:dm, :dm]
    #covx = cov[dm:dm+dx, dm:dm+dx]
    #covy = cov[dm+dx:, dm+dx:]

    wm, vm = la.eigh(covm)
    wm = wm[::-1]
    vm = vm[:, ::-1]
    #wx, vx = la.eigh(covx)
    #wy, vy = la.eigh(covy)

    wm_thresholded_mask = (wm > 1e-10)
    wm_cumsum = np.cumsum(wm) / wm.sum()
    wm_var_capture_index = np.where(wm_cumsum < 1 - 1e-3)[0].max() + 1
    wm_var_mask = (np.arange(wm.size) <= wm_var_capture_index)
    mask = wm_thresholded_mask & wm_var_mask

    wm_new = wm[mask]
    vm_new = vm[:, mask]
    dm_new = wm_new.size

    transform = np.tile([[vm_new.T, np.zeros((dm, dx + dy))],
                         [np.zeros((dx + dy, dm)), np.eye(dx + dy)]])
    cov_new = transform @ cov @ transform.T

    return cov_new, dm_new


def make_cov_m_equals_xy(covxy, dx, dy, epsilon=0.0):
    """
    Utility function to make a joint covariance matrix for M, X and Y
    where M = (X, Y), given the covariance matrix for (X, Y).

    Parameters
    ----------
    covxy : np.ndarray [dx+dy, dx+dy]
            Covariance matrix of X and Y
    dx : int
            Dimension of X
    dy : int
            Dimension of Y

    Returns
    -------
    cov : np.ndarray [dm+dx+dy, dm+dx+dy]
            Covariance matrix of M, X and Y
    """
    covx = covxy[:dx, :dx]
    covy = covxy[dx:, dx:]
    covx_y = covxy[:dx, dx:]

    covm_x = np.vstack((covx, covx_y.T))
    covm_y = np.vstack((covx_y, covy))
    cov = np.block([[covxy, covm_x, covm_y],
                    [covm_x.T, covx + epsilon * np.eye(dx), covx_y],
                    [covm_y.T, covx_y.T, covy + epsilon * np.eye(dy)]])

    return cov


