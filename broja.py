#!/usr/bin/env python3

from __future__ import print_function, division

import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.linalg as npla

import torch
import torch.linalg

from .utils import whiten, lin_tf_params_from_cov, lin_tf_params_bert
from .estimate import exp_mvar_kl


def exact_bert_union_info_minimizer(hx_, hy_):
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

        covxy__m = np.block([[np.eye(dx), sig_temp],
                             [sig_temp.T, np.eye(dy)]])

        # Project covxy__m back onto the PSD cone
        lamda, V = la.eigh(covxy__m)
        lamda = lamda.real  # Real symmetric matrix should have real eigenvalues
        V = V.real
        lamda[lamda < 0] = 0
        covxy__m = V @ np.diag(lamda) @ V.T

        covx__m = covxy__m[:dx, :dx]
        covy__m = covxy__m[dx:, dx:]

        # Pull sig out of covxy and re-standardize
        sig_temp_proj = la.solve(la.sqrtm(covx__m), la.solve(la.sqrtm(covy__m), covxy__m[:dx, dx:].T).T)
        sig[:, :] = torch.from_numpy(sig_temp_proj.astype(np.float32))

    eta_sig = 1e-3  # Learning rate for sig
    # Adam parameters
    gamma = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    beta3 = 0.99
    eps = 1e-3
    m_sig = torch.zeros((dx, dy))
    v_sig = torch.zeros((dx, dy))

    noise_std = 0
    stop_threshold = 1e-4
    max_iterations = 10000
    patience = 20    # Number of iterations with small gradient before stopping
    extra_iters = 0

    minima = None
    running_obj = []
    running_sig_pre_proj = [sig_temp,]
    running_sig_post_proj = [sig_temp_proj,]
    i = 1
    extra = 0
    while True:
        # Evaluate the objective
        obj = 0.5 * torch.logdet(torch.eye(dm) + hy.T @ hy + (hx - sig @ hy).T @ torch.linalg.solve((1 + 1e-5) * torch.eye(dx) - sig @ sig.T, (hx - sig @ hy))) / np.log(2)
        obj.backward()

        with torch.no_grad():
            if minima is None or obj.item() < min(running_obj):
                minima = copy.deepcopy(sig)

            g_sig = sig.grad + torch.abs(sig.grad).mean() * noise_std * torch.randn(dx, dy)
            #m_sig = beta1 * m_sig + (1 - beta1) * g_sig
            #v_sig = beta2 * v_sig + (1 - beta2) * g_sig**2
            #sig -= gamma * (m_sig / (1 - beta1**i)) / (torch.sqrt(v_sig / (1 - beta2**i)) + eps)
            sig -= beta3**i * eta_sig * g_sig

            # Project sig back onto the PSD cone
            sig_temp = sig.detach().numpy()
            running_sig_pre_proj.append(sig_temp.copy())
            covxy__m = np.block([[np.eye(dx), sig_temp],
                                 [sig_temp.T, np.eye(dy)]])

            # Project covxy__m back onto the PSD cone
            lamda, V = la.eigh(covxy__m)
            lamda = lamda.real  # Real symmetric matrix should have real eigenvalues
            V = V.real
            lamda[lamda < 0] = 0
            covxy__m = V @ np.diag(lamda) @ V.T

            covx__m = covxy__m[:dx, :dx]
            covy__m = covxy__m[dx:, dx:]

            # Pull sig out of covxy and re-standardize
            sig_temp_proj = la.solve(la.sqrtm(covx__m), la.solve(la.sqrtm(covy__m), covxy__m[:dx, dx:].T).T)
            running_sig_post_proj.append(sig_temp_proj.copy())
            sig[:, :] = torch.from_numpy(sig_temp_proj.astype(np.float32))

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

    running_sig_pre_proj = np.array(running_sig_pre_proj).squeeze()
    running_sig_post_proj = np.array(running_sig_post_proj).squeeze()

    plt.semilogy(running_obj)
    plt.figure()
    plt.plot(running_sig_pre_proj)
    plt.plot(running_sig_post_proj)
    plt.figure()
    if dx == 2 and dy == 1:
        x, y = np.mgrid[-1:1:100j, -1:1:100j]
        sigs = np.moveaxis(np.array((x, y)), 0, 2)
        objs = []
        for sig in sigs.reshape((-1, 2)):
            sig = sig.reshape((dx, dy))
            if np.any(np.linalg.eigvals(np.eye(dx) - sig @ sig.T) < 0):
                objs.append(np.nan)
                continue
            obj = 0.5 * npla.slogdet(np.eye(dm) + hy_.T @ hy_ + (hx_ - sig @ hy_).T @ la.solve((1 + 1e-5) * np.eye(dx) - sig @ sig.T, (hx_ - sig @ hy_)))[1] / np.log(2)
            objs.append(obj)
        objs = np.array(objs).reshape(sigs.shape[:2])
        plt.pcolormesh(x, y, objs[:-1, :-1], cmap='jet')
        plt.colorbar()
        for pre, post in zip(running_sig_pre_proj, running_sig_post_proj):
            plt.plot([pre[0], post[0]], [pre[1], post[1]], 'k-')
        for pre, post in zip(running_sig_pre_proj[1:], running_sig_post_proj[:-1]):
            plt.plot([pre[0], post[0]], [pre[1], post[1]], 'w-')

    plt.show()

    sig = minima

    return sig.detach().numpy()


def exact_bert_pytorch(cov, dm, dx, dy, verbose=False, ret_t_sigt=False):
    # NOTE: Not using lin_tf_params_bert
    #ret = lin_tf_params_from_cov(cov, dm, dx, dy)
    #hx, hy, sigx_y, sigm, sigx, sigy, hxy, sigxy = ret
    ret = whiten(cov, dm, dx, dy, ret_channel_params=True)
    sig_mxy, hx, hy, hxy, sigxy = ret

    imx = 0.5 * npla.slogdet(np.eye(dm) + hx.T @ hx)[1] / np.log(2)
    imy = 0.5 * npla.slogdet(np.eye(dm) + hy.T @ hy)[1] / np.log(2)
    imxy = 0.5 * npla.slogdet(np.eye(dm) + hxy.T @ la.solve(sigxy, hxy))[1] / np.log(2)

    sig = exact_bert_union_info_minimizer(hx, hy)
    covxy__m = np.block([[np.eye(dx), sig], [sig.T, np.eye(dy)]])
    #covxy = covxy__m + np.vstack((hx, hy)) @ np.vstack((hx, hy)).T

    union_info = 0.5 * npla.slogdet(np.eye(dm) + hxy.T @ la.solve(covxy__m + 1e-5 * np.eye(*covxy__m.shape), hxy))[1] / np.log(2)

    uix = union_info - imy
    uiy = union_info - imx
    ri = imx + imy - union_info
    si = imxy - union_info

    # Return None in place of deficiency values to keep return signature consistent
    ret = (imx, imy, imxy, None, None, uix, uiy, ri, si)
    if ret_t_sigt:
        ret = (*ret, None, None, None, sig)

    return ret
