#!/usr/bin/env python3

from __future__ import print_function, division

import sys
import copy
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.linalg as npla

import torch
import torch.linalg

from .utils import whiten, lin_tf_params_from_cov, lin_tf_params_bert
from .estimate import exp_mvar_kl


def objective(sig, hx, hy, dm, dx, dy, reg):
    S = (1 + reg) * torch.eye(dx) - sig @ sig.T
    B = hx - sig @ hy
    obj = 0.5 / np.log(2) * torch.logdet(
        torch.eye(dm) + hy.T @ hy + B.T @ torch.linalg.solve(S, B)
    )
    return obj


def objective_numpy(sig, hx_, hy_, dm, dx, dy, reg):
    S = (1 + reg) * np.eye(dx) - sig @ sig.T
    B = hx_ - sig @ hy_
    obj = 0.5 / np.log(2) * npla.slogdet(
        np.eye(dm) + hy_.T @ hy_ + B.T @ la.solve(S, B)
    )[1]
    return obj


def gradient_numpy(sig, hx_, hy_, dm, dx, dy, reg):
    S = (1 + reg) * np.eye(dx) - sig @ sig.T
    B = hx_ - sig @ hy_
    S_inv_B = la.solve(S, B)
    g_sig = S_inv_B @ la.solve(np.eye(dm) + hy_.T @ hy_ + B.T @ S_inv_B,
                               S_inv_B.T @ sig - hy_.T)
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
    sig_temp_proj = la.solve(la.sqrtm(covx__m),
                             la.solve(la.sqrtm(covy__m), covxy__m[:dx, dx:].T).T)

    return sig_temp_proj, True


def exact_bert_union_info_minimizer(hx_, hy_, plot=False):
    dx, dm = hx_.shape
    dy, dm_ = hy_.shape
    if dm != dm_:
        raise ValueError('Incompatible shapes for Hx and Hy')

    sig = torch.eye(dx, dy, requires_grad=True)

    hx = torch.from_numpy(hx_.astype(np.float32))
    hy = torch.from_numpy(hy_.astype(np.float32))

    # Initialize sig
    with torch.no_grad():
        # XXX: Choice of which to pinv is arbitrary - can we average instead?
        # (based on the two equations: we can either write it out in terms of
        # hx - sig @ hy or hy - sig.T @ hx)
        sig_temp = hx_ @ la.pinv(hy_)
        sig_temp_proj = project(sig_temp)[0]
        sig[:, :] = torch.from_numpy(sig_temp_proj.astype(np.float32))

    # Gradient descent
    eta_sig = 0.1       # Learning rate for sig
    bt_ls = True        # Whether to use backtracking line search
    beta_bt = 0.9       # Shrinking rate for backtracking line search
    max_bt_iters = 150  # Maximum number of iters to try backtracking

    # Adam parameters
    gamma = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    beta3 = 0.99
    eps = 1e-3
    m_sig = torch.zeros((dx, dy))
    v_sig = torch.zeros((dx, dy))

    reg = 1e-7       # Regularization in the objective for matrix inverse
    noise_std = 0    # Standard deviation of noise to add to the gradient
    stop_threshold = 1e-6  # Absolute difference in objective for stopping
    max_iterations = 10000
    patience = 20    # Num iters with small gradient before stopping (min=1)
    extra_iters = 0  # Num of extra iters after stop criterion is attained

    minima = None
    running_obj = []
    running_sig_pre_proj = [sig_temp,]
    running_sig_post_proj = [sig_temp_proj,]
    running_grad = []
    running_grad_numpy = []
    running_eta = []
    i = 1
    extra = 0
    while True:
        # Evaluate the objective
        obj = objective(sig, hx, hy, dm, dx, dy, reg)
        obj.backward()

        with torch.no_grad():
            if minima is None or obj.item() < min(running_obj):
                minima = copy.deepcopy(sig)

            if len(running_obj) >= patience:
                if extra == 0:
                    if (np.abs(np.array(running_obj[-patience:]) - obj.item()) < stop_threshold).all() or i >= max_iterations:
                        if extra_iters == 0: break
                        extra += 1
                elif extra > extra_iters:
                    break
                else:
                    extra += 1

            running_obj.append(copy.deepcopy(obj.item()))
            i += 1

            # Compute gradient
            g_sig = sig.grad #+ torch.abs(sig.grad).mean() * noise_std * torch.randn(dx, dy)

            # Clip element-wise gradient values at +/-1
            g_sig = torch.minimum(g_sig, torch.ones(g_sig.shape))
            g_sig = torch.maximum(g_sig, -torch.ones(g_sig.shape))

            sig_ = sig.detach().numpy()
            g_sig_numpy = gradient_numpy(sig_, hx_, hy_, dm, dx, dy, reg)
            g_sig_numpy = np.minimum(g_sig_numpy, 1)
            g_sig_numpy = np.maximum(g_sig_numpy, -1)
            running_grad_numpy.append(g_sig_numpy.copy())

            bt_i = 0  # Backtracking iteration number
            while True:
                # Vanilla gradient descent
                #sig_plus = sig - eta_sig * g_sig
                sig_plus_ = sig_ - eta_sig * g_sig_numpy

                # Vanilla gradient descent update with exponential lr decay
                #sig_plus = sig - beta3**i * eta_sig * g_sig

                # Adam update
                #m_sig = beta1 * m_sig + (1 - beta1) * g_sig
                #v_sig = beta2 * v_sig + (1 - beta2) * g_sig**2
                #sig_plus = sig - gamma * (m_sig / (1 - beta1**i)) / (torch.sqrt(v_sig / (1 - beta2**i)) + eps)

                # Project sig back onto the PSD cone
                #sig_temp = sig_plus.detach().numpy()
                sig_temp = sig_plus_.copy()
                sig_temp_proj, sig_changed = project(sig_temp)
                #sig_plus[:, :] = torch.from_numpy(sig_temp_proj.astype(np.float32))

                if not bt_ls:  # Not using backtracking
                    break

                # Backtracking line search
                #if sig_changed:
                #    # Projection took effect; we are at the boundary
                #    # So don't backtrack
                #    eta_sig *= beta3  # Shrink eta anyway
                #    break
                # Sigma didn't change: can use sig_plus to compute obj_plus
                obj_plus = objective_numpy(sig_temp_proj, hx_, hy_, dm, dx, dy, reg)
                #obj_plus = objective(sig_plus, hx, hy, dm, dx, dy, reg)
                if obj_plus < obj:
                    break
                elif bt_i > max_bt_iters:
                    # XXX: This warning will get issued if we have already converged
                    warnings.warn('Backtracking line search failed')
                    break
                else:
                    bt_i += 1
                    eta_sig *= beta_bt  # Change lr for vanilla GD
                    running_eta.append(eta_sig)

            running_grad.append(g_sig.detach().numpy().copy())
            running_sig_pre_proj.append(sig_temp.copy())
            running_sig_post_proj.append(sig_temp_proj.copy())
            sig[:, :] = torch.from_numpy(sig_temp_proj.astype(np.float32))

    running_sig_pre_proj = np.array(running_sig_pre_proj).squeeze()
    running_sig_post_proj = np.array(running_sig_post_proj).squeeze()
    running_grad = np.array(running_grad).squeeze()
    running_grad_numpy = np.array(running_grad_numpy).squeeze()

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
            plt.plot(running_grad_numpy)
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
                obj = objective_numpy(sig, hx_, hy_, dm, dx, dy, reg)
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
                obj = objective_numpy(sig, hx_, hy_, dm, dx, dy, reg)
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
                        obj = objective_numpy(sig, hx_, hy_, dm, dx, dy, reg)
                        objs.append(obj)
                    plt.plot(x, objs, label=('$\Sigma_{%d%d}$' % (i, j)))
            plt.title('Objective around optima')
            plt.xlabel('$\Sigma_{ij}$')
            plt.ylabel('Objective')
            plt.legend()

        plt.subplot(nrows, ncols, 4)
        plt.semilogy(running_eta)

        plt.show()

    sig = minima

    return sig.detach().numpy()


def exact_bert_pytorch(cov, dm, dx, dy, verbose=False, ret_t_sigt=False, plot=False):
    # NOTE: Not using lin_tf_params_bert
    #ret = lin_tf_params_from_cov(cov, dm, dx, dy)
    #hx, hy, sigx_y, sigm, sigx, sigy, hxy, sigxy = ret
    ret = whiten(cov, dm, dx, dy, ret_channel_params=True)
    sig_mxy, hx, hy, hxy, sigxy = ret

    imx = 0.5 * npla.slogdet(np.eye(dm) + hx.T @ hx)[1] / np.log(2)
    imy = 0.5 * npla.slogdet(np.eye(dm) + hy.T @ hy)[1] / np.log(2)
    imxy = 0.5 * npla.slogdet(np.eye(dm) + hxy.T @ la.solve(sigxy, hxy))[1] / np.log(2)

    sig = exact_bert_union_info_minimizer(hx, hy, plot=plot)
    covxy__m = np.block([[np.eye(dx), sig], [sig.T, np.eye(dy)]])
    #covxy = covxy__m + np.vstack((hx, hy)) @ np.vstack((hx, hy)).T

    union_info = 0.5 / np.log(2) * npla.slogdet(
        np.eye(dm) + hxy.T @ la.solve(covxy__m + 1e-5 * np.eye(*covxy__m.shape), hxy))[1]

    uix = union_info - imy
    uiy = union_info - imx
    ri = imx + imy - union_info
    si = imxy - union_info

    # Return None in place of deficiency values to keep return signature consistent
    ret = (imx, imy, imxy, None, None, uix, uiy, ri, si)
    if ret_t_sigt:
        ret = (*ret, None, None, None, sig)

    return ret
