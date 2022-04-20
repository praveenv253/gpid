#!/usr/bin/env python3

from __future__ import print_function, division

import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

import torch
import torch.linalg

from .utils import lin_tf_params_from_cov
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

    #eta_t = 0.01  # Learning rate for t
    #eta_sigt = 0.01  # Learning rate for sigt
    # Adam parameters
    gamma = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m_t = torch.zeros((dy, dx))
    v_t = torch.zeros((dy, dx))
    m_sigt = torch.zeros((dy, dy))
    v_sigt = torch.zeros((dy, dy))

    noise_std = 1
    stop_threshold = 1e-4
    max_iterations = 5000

    prev_obj = None
    running_error = []
    i = 1
    while True:
        # Evaluate the objective
        term1 = torch.eye(dy) + (hy - t @ hx) @ (hy - t @ hx).T
        obj = (torch.trace(torch.linalg.solve(sigt + t @ t.T, term1))
               - dy + torch.logdet(sigt + t @ t.T))
        obj.backward()

        with torch.no_grad():
            g_t = t.grad + beta2**i * torch.abs(t.grad).mean() * noise_std * torch.randn(dy, dx)
            m_t = beta1 * m_t + (1 - beta1) * g_t
            v_t = beta2 * v_t + (1 - beta2) * g_t**2
            t -= gamma * (m_t / (1 - beta1**i)) / (torch.sqrt(v_t / (1 - beta2**i)) + eps)
            #t -= eta_t * g_t

            g_sigt = sigt.grad + torch.abs(sigt.grad).mean() * noise_std * torch.randn(dy, dy)
            m_sigt = beta1 * m_sigt + (1 - beta1) * g_sigt
            v_sigt = beta2 * v_sigt + (1 - beta2) * g_sigt**2
            sigt -= gamma * (m_sigt / (1 - beta1**i)) / (torch.sqrt(v_sigt / (1 - beta2**i)) + eps)
            #sigt -= eta_sigt * g_sigt

            # Project sigt back onto the PSD cone
            lamda, V = torch.linalg.eig(sigt)
            lamda = lamda.real  # Real symmetric matrix should have real eigenvalues
            V = V.real
            lamda[lamda < 0] = 0
            sigt[:, :] = V @ torch.diag(lamda) @ V.T

        if prev_obj is not None:
            error = abs(obj.item() - prev_obj)
            running_error.append(obj.item())

            if error < stop_threshold or i >= max_iterations:
                break

        prev_obj = copy.deepcopy(obj.item())
        i += 1

    plt.semilogy(running_error)
    plt.show()

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
