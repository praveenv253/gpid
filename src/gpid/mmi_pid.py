#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npla

from .utils import whiten, solve


def mmi_pid(cov, dm, dx, dy):
    ret = whiten(cov, dm, dx, dy, ret_channel_params=True)
    sig_mxy, hx, hy, hxy, sigxy = ret

    imx = 0.5 * npla.slogdet(np.eye(dm) + hx.T @ hx)[1] / np.log(2)
    imy = 0.5 * npla.slogdet(np.eye(dm) + hy.T @ hy)[1] / np.log(2)
    imxy = 0.5 * npla.slogdet(np.eye(dm) + hxy.T @ solve(sigxy + 1e-7 * np.eye(*sigxy.shape), hxy))[1] / np.log(2)

    ri = min(imx, imy)
    uix = imx - ri
    uiy = imy - ri
    si = imxy - (uix + uiy + ri)

    return imx, imy, imxy, None, None, uix, uiy, ri, si
