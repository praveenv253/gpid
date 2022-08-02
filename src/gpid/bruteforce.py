#!/usr/bin/env python3

from __future__ import print_function, division

import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.linalg as npla

import torch
import torch.linalg

from .utils import lin_tf_params_from_cov, lin_tf_params_bert
from .estimate import exp_mvar_kl


def exact_deficiency_pytorch(hy_, hx_):
    """
    Computes deficiency of X with respect to Y, i.e. unique information
    corresponding to Y \\ X.

    Assumes hx and hy contain full SNR information, including sigm and sigx/sigy.
    """

    dx, dm = hx_.shape
    dy, dm_ = hy_.shape
    if dm != dm_:
        raise ValueError('Incompatible shapes for Hx and Hy')

    t = torch.eye(dy, dx, requires_grad=True)
    sigt = torch.eye(dy, dy, requires_grad=True)

    hx = torch.from_numpy(hx_.astype(np.float32))
    hy = torch.from_numpy(hy_.astype(np.float32))

    # Initialize t and sigt
    with torch.no_grad():
        t[:, :] = torch.from_numpy((hy_ @ la.pinv(hx_)).astype(np.float32))
        sigt[:, :] = torch.eye(dy, dy) - t @ t.T + (hy - t @ hx) @ (hy - t @ hx).T

        # Project sigt back onto the PSD cone
        lamda, V = torch.linalg.eig(sigt)
        lamda = lamda.real  # Real symmetric matrix should have real eigenvalues
        V = V.real
        lamda[lamda < 0] = 0
        sigt[:, :] = V @ torch.diag(lamda) @ V.T

    eta_t = 1e-3  # Learning rate for t
    eta_sigt = 1e-3  # Learning rate for sigt
    # Adam parameters
    gamma = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    beta3 = 0.99
    eps = 1e-8
    m_t = torch.zeros((dy, dx))
    v_t = torch.zeros((dy, dx))
    m_sigt = torch.zeros((dy, dy))
    v_sigt = torch.zeros((dy, dy))

    noise_std = 0
    stop_threshold = 1e-4
    max_iterations = 10000

    minima = None
    running_obj = []
    i = 1
    extra = 0
    while True:
        # Evaluate the objective
        term1 = torch.eye(dy) + (hy - t @ hx) @ (hy - t @ hx).T
        obj = (torch.trace(torch.linalg.solve(sigt + t @ t.T, term1))
               - dy + torch.logdet(sigt + t @ t.T)) * 0.5 / np.log(2)
        obj.backward()

        with torch.no_grad():
            if minima is None or obj.item() < min(running_obj):
                minima = (copy.deepcopy(t), copy.deepcopy(sigt))

            g_t = t.grad + beta2**i * torch.abs(t.grad).mean() * noise_std * torch.randn(dy, dx)
            m_t = beta1 * m_t + (1 - beta1) * g_t
            v_t = beta2 * v_t + (1 - beta2) * g_t**2
            t -= gamma * (m_t / (1 - beta1**i)) / (torch.sqrt(v_t / (1 - beta2**i)) + eps)
            #t -= beta3**i * eta_t * g_t

            g_sigt = sigt.grad + torch.abs(sigt.grad).mean() * noise_std * torch.randn(dy, dy)
            m_sigt = beta1 * m_sigt + (1 - beta1) * g_sigt
            v_sigt = beta2 * v_sigt + (1 - beta2) * g_sigt**2
            sigt -= gamma * (m_sigt / (1 - beta1**i)) / (torch.sqrt(v_sigt / (1 - beta2**i)) + eps)
            #sigt -= beta3**i * eta_sigt * g_sigt

            # Project sigt back onto the PSD cone
            lamda, V = torch.linalg.eig(sigt)
            lamda = lamda.real  # Real symmetric matrix should have real eigenvalues
            V = V.real
            lamda[lamda < 0] = 0
            sigt[:, :] = V @ torch.diag(lamda) @ V.T

            if len(running_obj) > 0:
                if extra == 0:
                    if (np.abs(np.array(running_obj[-20:]) - obj.item()) < stop_threshold).all() or i >= max_iterations:
                        extra += 1
                elif extra > 500:
                    break
                else:
                    extra += 1

            running_obj.append(copy.deepcopy(obj.item()))
            i += 1

    plt.semilogy(running_obj)
    plt.show()

    t, sigt = minima

    return t.detach().numpy(), sigt.detach().numpy()


def exact_pid_pytorch(cov, dm, dx, dy, verbose=False, ret_t_sigt=False):
    ret = lin_tf_params_from_cov(cov, dm, dx, dy)
    hx, hy, hxy, sigx, sigy, sigxy, covxy, sigm = ret

    # compute mutual informations
    imx = exp_mvar_kl(hx, sigx, np.zeros_like(hx), hx @ sigm @ hx.T + sigx, sigm)
    imy = exp_mvar_kl(hy, sigy, np.zeros_like(hy), hy @ sigm @ hy.T + sigy, sigm)
    imxy = exp_mvar_kl(hxy, sigxy, np.zeros_like(hxy), covxy, sigm)

    # Fold sigm into hx and hy
    sqrt_sigm = la.sqrtm(sigm)
    hx = hx @ sqrt_sigm
    hy = hy @ sqrt_sigm

    t, sigt = exact_deficiency_pytorch(hx, hy)
    def_x_minus_y = exp_mvar_kl(hx, np.eye(dx), t @ hy, t @ t.T + sigt, sigm)
    tx, sigtx = t.copy(), sigt.copy()

    t, sigt = exact_deficiency_pytorch(hy, hx)
    def_y_minus_x = exp_mvar_kl(hy, np.eye(dy), t @ hx, t @ t.T + sigt, sigm)
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


#def exact_ip_dfncy_pytorch(cov, dm, dx, dy, verbose=False, ret_t_sigt=False):


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

        #import pdb
        #pdb.set_trace()

        covxy = np.block([[np.eye(dx), sig_temp],
                          [sig_temp.T, np.eye(dy)]])

        # Project covxy back onto the PSD cone
        lamda, V = la.eigh(covxy)
        lamda = lamda.real  # Real symmetric matrix should have real eigenvalues
        V = V.real
        lamda[lamda < 0] = 0
        covxy = V @ np.diag(lamda) @ V.T

        covx = covxy[:dx, :dx]
        covy = covxy[dx:, dx:]

        # Pull sig out of covxy and re-standardize
        sig_temp_proj = la.solve(la.sqrtm(covx), la.solve(la.sqrtm(covy), covxy[:dx, dx:].T).T)
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

    minima = None
    running_obj = []
    running_sig_pre_proj = [sig_temp,]
    running_sig_post_proj = [sig_temp_proj,]
    i = 1
    extra = 0
    while True:
        # Evaluate the objective
        obj = - 0.5 * torch.logdet(torch.eye(dm) - hy.T @ hy - (hx - sig @ hy).T @ torch.linalg.solve((1 + 1e-5) * torch.eye(dx) - sig @ sig.T, (hx - sig @ hy))) / np.log(2)
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
            covxy = np.block([[np.eye(dx), sig_temp],
                              [sig_temp.T, np.eye(dy)]])

            # Project covxy back onto the PSD cone
            lamda, V = la.eigh(covxy)
            lamda = lamda.real  # Real symmetric matrix should have real eigenvalues
            V = V.real
            lamda[lamda < 0] = 0
            covxy = V @ np.diag(lamda) @ V.T

            covx = covxy[:dx, :dx]
            covy = covxy[dx:, dx:]

            # Pull sig out of covxy and re-standardize
            sig_temp_proj = la.solve(la.sqrtm(covx), la.solve(la.sqrtm(covy), covxy[:dx, dx:].T).T)
            running_sig_post_proj.append(sig_temp_proj.copy())
            sig[:, :] = torch.from_numpy(sig_temp_proj.astype(np.float32))

            if len(running_obj) > 0:
                if extra == 0:
                    if (np.abs(np.array(running_obj[-20:]) - obj.item()) < stop_threshold).all() or i >= max_iterations:
                        extra += 1
                elif extra > 500:
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
            obj = - 0.5 * npla.slogdet(np.eye(dm) - hy_.T @ hy_ - (hx_ - sig @ hy_).T @ la.solve((1 + 1e-5) * np.eye(dx) - sig @ sig.T, (hx_ - sig @ hy_)))[1] / np.log(2)
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
    ret = lin_tf_params_bert(cov, dm, dx, dy)
    hx, hy, sigx_y, sigm, sigx, sigy, hxy, sigxy = ret
    # This sigx is different from the other sigx'es, because it refers to the
    # standardized autocovariance of X (similarly for sigy, sigx_y, etc.).

    imx = - 0.5 * npla.slogdet(np.eye(dm) - hx.T @ hx)[1] / np.log(2)
    imy = - 0.5 * npla.slogdet(np.eye(dm) - hy.T @ hy)[1] / np.log(2)
    imxy = - 0.5 * npla.slogdet(np.eye(dm) - hxy.T @ la.solve(sigxy, hxy))[1] / np.log(2)

    sig = exact_bert_union_info_minimizer(hx, hy)
    covxy = np.block([[sigx, sig], [sig.T, sigy]])

    union_info = - 0.5 * npla.slogdet(np.eye(dm) - hxy.T @ la.solve(covxy + 1e-5 * np.eye(*covxy.shape), hxy))[1] / np.log(2)

    uix = union_info - imy
    uiy = union_info - imx
    ri = imx + imy - union_info
    si = imxy - union_info

    # Return None in place of deficiency values to keep return signature consistent
    ret = (imx, imy, imxy, None, None, uix, uiy, ri, si)
    if ret_t_sigt:
        ret = (*ret, None, None, None, sig)

    return ret
