#!/usr/bin/env python3

from __future__ import print_function, division

import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.linalg as npla

from .utils import whiten


def objective(sig, hx, hy, dm, dx, dy, reg):
    S = (1 + reg) * np.eye(dx) - sig @ sig.T
    B = hx - sig @ hy
    obj = 0.5 / np.log(2) * npla.slogdet(
        np.eye(dm) + hy.T @ hy + B.T @ la.solve(S, B)
    )[1]
    return obj


def gradient(sig, hx, hy, dm, dx, dy, reg):
    S = (1 + reg) * np.eye(dx) - sig @ sig.T
    B = hx - sig @ hy
    S_inv_B = la.solve(S, B)
    g_sig = S_inv_B @ la.solve(np.eye(dm) + hy.T @ hy + B.T @ S_inv_B,
                               S_inv_B.T @ sig - hy.T)
    return g_sig


def project(sig_temp):
    """
    Returns a projection of sig_temp which satisfies:
        I - sig_temp @ sig_temp.T is positive semi-definite.
    Also returns a flag to indicate whether the projection changed the matrix.
    """
    dx, dy = sig_temp.shape

    # Project sig back onto the PSD cone
    covxy__m = np.block([[np.eye(dx), sig_temp],
                         [sig_temp.T, np.eye(dy)]])

    # Project covxy__m back onto the PSD cone
    lamda, V = la.eigh(covxy__m)
    lamda = lamda.real  # Real symmetric matrix should have real eigenvalues
    if (lamda >= 0).all():
        return sig_temp, False

    V = V.real
    lamda[lamda < 0] = 0
    covxy__m = V @ np.diag(lamda) @ V.T

    covx__m = covxy__m[:dx, :dx]
    covy__m = covxy__m[dx:, dx:]

    # Pull sig out of covxy and re-standardize
    sig_temp_proj = la.solve(la.sqrtm(covx__m).real,
                             la.solve(la.sqrtm(covy__m).real, covxy__m[:dx, dx:].T).T)

    return sig_temp_proj, True


def exact_tilde_union_info_minimizer(hx, hy, plot=False, ret_obj=False):
    dx, dm = hx.shape
    dy, dm_ = hy.shape
    if dm != dm_:
        raise ValueError('Incompatible shapes for Hx and Hy')

    # Initialize sig
    # XXX: Choice of which to pinv is arbitrary - can we average instead?
    # (based on the two equations: we can either write it out in terms of
    # hx - sig @ hy or hy - sig.T @ hx)
    sig_temp = hx @ la.pinv(hy)
    sig_temp_proj = project(sig_temp)[0]
    sig = sig_temp_proj.copy()

    # Gradient descent
    eta_sig = 1e-3 * np.ones((dx, dy))
    beta = 0.9       # Factor to increase or decrease LR for Rprop
    alpha = 0.999    # Slow decay of overall learning rate

    reg = 1e-7       # Regularization in the objective for matrix inverse
    noise_std = 0    # Standard deviation of noise to add to the gradient
    stop_threshold = 1e-6  # Absolute difference in objective for stopping
    max_iterations = 10000
    patience = 20    # Num iters with small gradient before stopping (min=1)
    extra_iters = 0  # Num of extra iters after stop criterion is attained

    minima = None
    g_sig_prev = None
    running_obj = []
    running_sig_pre_proj = [sig_temp,]
    running_sig_post_proj = [sig_temp_proj,]
    running_grad = []
    running_eta = []
    i = 1
    extra = 0
    while True:
        # Evaluate the objective
        obj = objective(sig, hx, hy, dm, dx, dy, reg)

        if minima is None or obj < min(running_obj):
            minima = (sig.copy(), obj)

        if len(running_obj) >= patience:
            if extra == 0:
                if (np.abs(np.array(running_obj[-patience:]) - obj) < stop_threshold).all() or i >= max_iterations:
                    if i >= max_iterations:
                        warnings.warn('Exceeded maximum number of iterations. May not have converged.')
                    if extra_iters == 0: break
                    extra += 1
            elif extra > extra_iters:
                break
            else:
                extra += 1

        if np.isnan(obj):
            running_obj.append(np.inf)
        else:
            running_obj.append(obj)
        i += 1

        g_sig = gradient(sig, hx, hy, dm, dx, dy, reg)
        g_sig = np.sign(g_sig).astype(int)

        # Backtracking with Rprop would have to work by ensuring that
        # gradients are moving in the right direction along all dimensions.
        #
        # This won't work well in conjunction with *projected* gradient
        # descent, because the projection step may be parallel and opposite
        # to the gradient step in one dimension. If so, that dimension can
        # never be made to move in the right direction, meaning
        # backtracking will fail before convergence.

        # Vanilla gradient descent
        sig_plus = sig - alpha**i * eta_sig * g_sig

        # Project sig back onto the PSD cone
        sig_proj, _ = project(sig_plus)

        # Learning rate update
        if g_sig_prev is not None:
            sign_changed = - g_sig * g_sig_prev  # -1 if sign did not change, +1 if sign changed
            eta_sig *= beta**sign_changed

        g_sig_prev = g_sig

        running_eta.append(eta_sig)
        running_grad.append(g_sig)
        running_sig_pre_proj.append(sig_plus)
        running_sig_post_proj.append(sig_proj)
        sig[:, :] = sig_proj

    running_sig_pre_proj = np.array(running_sig_pre_proj).squeeze()
    running_sig_post_proj = np.array(running_sig_post_proj).squeeze()
    running_grad = np.array(running_grad).squeeze()

    if plot:
        nrows = 2
        ncols = 2
        plt.figure(figsize=(10, 7))
        plt.subplot(nrows, ncols, 1)
        plt.semilogy(running_obj)
        plt.title('Convergence of objective')
        plt.ylabel('Objective')
        plt.xlabel('Iteration')

        if dx == 1 or dy == 1:
            plt.subplot(nrows, ncols, 2)
            plt.plot(running_sig_pre_proj)
            plt.plot(running_sig_post_proj)
            plt.plot(running_grad)
            plt.title('Convergence of minimizer')
            plt.ylabel('$\Sigma_i$')
            plt.xlabel('Iteration')
        elif dx == 2 and dy == 2:
            plt.subplot(nrows, ncols, 2)
            pre_proj = running_sig_pre_proj.reshape((running_sig_pre_proj.shape[0], -1))
            post_proj = running_sig_post_proj.reshape((running_sig_post_proj.shape[0], -1))
            plt.plot(post_proj)
            plt.title('Convergence of minimizer')
            plt.ylabel('$\Sigma_i$')
            plt.xlabel('Iteration')

        if dx == 1 and dy == 1:
            plt.subplot(nrows, ncols, 3)
            x_ = np.linspace(-1, 1, 100)
            #x_ = np.linspace(0.69, 0.71, 100)
            x = 0.5 * (x_[1:] + x_[:-1])
            t = np.arange(len(running_obj) + 1)
            objs = []
            for xi in x:
                sig = np.array([[xi]])
                if np.any(np.eye(dx) - sig @ sig.T <= 0):
                    objs.append(np.nan)
                    continue
                obj = objective(sig, hx, hy, dm, dx, dy, reg)
                objs.append(obj)
            objs = np.repeat(np.array(objs).reshape((1, -1)), len(running_obj), axis=0)
            plt.pcolormesh(t, x_, objs.T, cmap='jet')
            plt.colorbar()
            plt.plot(running_sig_post_proj, 'w-')
            #plt.ylim((0.69, 0.71))

        if (dx == 2 and dy == 1) or (dx == 1 and dy == 2):
            plt.subplot(nrows, ncols, 3)
            x, y = np.mgrid[-1:1:100j, -1:1:100j]
            sigs = np.moveaxis(np.array((x, y)), 0, 2)
            objs = []
            for sig in sigs.reshape((-1, 2)):
                sig = sig.reshape((dx, dy))
                if np.any(npla.eigvals(np.eye(dx) - sig @ sig.T) < 0):
                    objs.append(np.nan)
                    continue
                obj = objective(sig, hx, hy, dm, dx, dy, reg)
                objs.append(obj)
            objs = np.array(objs).reshape(sigs.shape[:2])
            plt.pcolormesh(x, y, objs[:-1, :-1], cmap='jet')
            plt.colorbar()
            #for pre, post in zip(running_sig_pre_proj, running_sig_post_proj):
            #    plt.plot([pre[0], post[0]], [pre[1], post[1]], 'k-')
            #for pre, post in zip(running_sig_pre_proj[1:], running_sig_post_proj[:-1]):
            #    plt.plot([pre[0], post[0]], [pre[1], post[1]], 'w-')
            #plt.plot(running_sig_pre_proj[:, 0], running_sig_pre_proj[:, 1], 'k-')
            plt.plot(running_sig_post_proj[:, 0], running_sig_post_proj[:, 1], 'w-')
            plt.plot(running_sig_post_proj[0, 0], running_sig_post_proj[0, 1], 'ko')

        if dx == 2 and dy == 2:
            plt.subplot(nrows, ncols, 3)
            x = np.linspace(-1, 1, 100)
            for i in range(2):
                for j in range(2):
                    post_proj = running_sig_post_proj[-1]
                    sigs = post_proj * np.ones((100, 2, 2))
                    sigs[:, i, j] = x
                    objs = []
                    for sig in sigs:
                        if np.any(npla.eigvals(np.eye(dx) - sig @ sig.T) < 0):
                            objs.append(np.nan)
                            continue
                        obj = objective(sig, hx, hy, dm, dx, dy, reg)
                        objs.append(obj)
                    plt.plot(x, objs, label=('$\Sigma_{%d%d}$' % (i, j)))
            plt.title('Objective around optima')
            plt.xlabel('$\Sigma_{ij}$')
            plt.ylabel('Objective')
            plt.legend()

        #plt.subplot(nrows, ncols, 4)
        #plt.semilogy(running_eta)

        plt.show()

    sig, obj = minima

    if ret_obj:
        return sig, obj
    return sig


def bias(d, n):
    return sum(np.log(1 - k / n) for k in range(1, d+1)) / np.log(2) / 2


def compute_bias(du, dv, n):
    """
    Compute the bias in the mutual information estimate based on the work of
    Cai et al. (J. Mult. Anal., 2015).

    This value needs to be subtracted from the mutual information estimate to
    recover the unbiased mutual information.
    """
    # Bias of differential entropy estimate
    return bias(du, n) + bias(dv, n) - bias(du + dv, n)


def debias(imxy, bias_):
    """Remove bias while ensuring non-negativity."""

    return np.maximum(imxy - bias_, 0)


def exact_gauss_tilde_pid(cov, dm, dx, dy, verbose=False, ret_t_sigt=False,
                          plot=False, unbiased=False, sample_size=None):

    # XXX: Debiasing has not been thoroughly tested.
    # Right now, we assume that the proportion of bias in the union information
    # is the same as the proportion of bias in I(M ; (X, Y)).

    if unbiased == True and sample_size is None:
        raise ValueError('Must supply sample_size when requesting unbiased estimates')

    ret = whiten(cov, dm, dx, dy, ret_channel_params=True)
    sig_mxy, hx, hy, hxy, sigxy = ret

    imx = 0.5 * npla.slogdet(np.eye(dm) + hx.T @ hx)[1] / np.log(2)
    imy = 0.5 * npla.slogdet(np.eye(dm) + hy.T @ hy)[1] / np.log(2)
    imxy = 0.5 * npla.slogdet(np.eye(dm) + hxy.T @ la.solve(sigxy + 1e-7 * np.eye(*sigxy.shape), hxy))[1] / np.log(2)

    if unbiased:
        imx = debias(imx, compute_bias(dm, dx, sample_size))
        imy = debias(imy, compute_bias(dm, dy, sample_size))
        imxy_debiased = debias(imxy, compute_bias(dm, dx + dy, sample_size))

        # But ensure that the debiased imxy does not go below the debiased imx
        # or the debiased imy, as this will make PID values negative
        imxy_debiased = max(imxy_debiased, imx, imy)
    else:
        imxy_debiased = imxy

    debias_factor = imxy_debiased / imxy

    #sig = exact_tilde_union_info_minimizer(hx, hy, plot=plot)
    sig, obj = exact_tilde_union_info_minimizer(hx, hy, plot=plot, ret_obj=True)
    covxy__m = np.block([[np.eye(dx), sig], [sig.T, np.eye(dy)]])
    #covxy = covxy__m + np.vstack((hx, hy)) @ np.vstack((hx, hy)).T

    #union_info = 0.5 / np.log(2) * npla.slogdet(
    #    np.eye(dm) + hxy.T @ la.solve(covxy__m + 1e-7 * np.eye(*covxy__m.shape), hxy))[1]
    #union_info = obj
    union_info = objective(sig, hx, hy, dm, dx, dy, reg=1e-7)

    union_info *= debias_factor

    # Union info is lower bounded by max{I(M; X), I(M; Y)} and upper bounded by
    # min{I(M; X) + I(M; Y), I(M; (X, Y))}: imposing this ensures positivity of
    # the PID terms
    union_info = max(union_info, imx, imy)
    union_info = min(union_info, imx + imy, imxy_debiased)

    uix = union_info - imy
    uiy = union_info - imx
    ri = imx + imy - union_info
    si = imxy_debiased - union_info

    #uix = (union_info - imy) * debias_factor
    #uiy = (union_info - imx) * debias_factor
    #ri = (imx + imy - union_info) * debias_factor
    #si = (imxy - union_info) * debias_factor

    # Return union_info and None in place of deficiency values to keep return signature consistent
    ret = (imx, imy, imxy_debiased, union_info, obj, uix, uiy, ri, si)
    if ret_t_sigt:
        ret = (*ret, None, None, None, sig)

    return ret
