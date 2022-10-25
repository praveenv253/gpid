#!/usr/bin/env python3

"""
This example consists of three scripts:
mult_poisson_utils.py, mult_poisson_gt.py and mult_poisson_example.py.

mult_poisson_utils.py contains utility functions used to define and sample from
the multivariate Poisson distribution.

mult_poisson_gt.py contains code to compute the ground truth ~-PID using the
admUI package of Banerjee et al. (2018). This package needs to be installed
for the script to be run.

mult_poisson_example.py computes the ~_G-PID and the delta_G-PID for the
estimated covariance matrix of the multivariate Poisson distribution. It is
designed to be run without installing admUI.

plot_mult_poisson.py expects both mult_poisson_gt.py and mult_poisson_example.py
to have been run, so that their outputs can be compared.
"""

from __future__ import print_function, division

import numpy as np
import pandas as pd

from gpid.tilde_pid import exact_gauss_tilde_pid
from gpid.estimate import approx_pid_from_cov
from gpid.mmi_pid import mmi_pid
from mult_poisson_utils import get_params, sample_mult_poisson


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)

    params = get_params()
    lamda_m, lamda_x, lamda_y, w_x1_vals, w_x2, w_y1, w_y2 = params

    dm = lamda_m.size  # Number of dimensions of M
    dx = lamda_x.size
    dy = lamda_y.size

    n = 1000000        # Sample size

    pid_table = pd.DataFrame()

    pid_defn_names = ['tilde', 'delta', 'mmi']
    pid_defns = [exact_gauss_tilde_pid, approx_pid_from_cov, mmi_pid]

    config_cols = ['desc', 'id', 'dm', 'dx', 'dy', 'w_x1']
    pid_cols = ['imxy', 'uix', 'uiy', 'ri', 'si']

    for i, w_x1 in enumerate(w_x1_vals):
        print(i, end=' ', flush=True)

        cols = [(col, '') for col in config_cols]
        vals = ['mult_poisson', i, dm, dx, dy, w_x1]

        w_x = np.array([[w_x1, w_x2]])  # Binomial thinning weights: shape (dx, dm)
        w_y = np.array([[w_y1, w_y2]])

        m, x, y = sample_mult_poisson(n, lamda_m, w_x, w_y, lamda_x, lamda_y)

        # Try a variance-stabilizing square-root transform
        #m = np.sqrt(m)
        #x = np.sqrt(x)
        #y = np.sqrt(y)

        mxy = np.vstack((m, x, y))
        cov = np.corrcoef(mxy)
        #print(cov)

        for pid_name, pid_defn in zip(pid_defn_names, pid_defns):
            ret = pid_defn(cov, dm, dx, dy)
            imxy, uix, uiy, ri, si = ret[2], *ret[-4:]

            cols.extend([(pid_name, col) for col in pid_cols])
            vals.extend([imxy, uix, uiy, ri, si])

        row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
        pid_table = pd.concat((pid_table, row), ignore_index=True)

    print()

    pid_table.to_pickle('../results/mult_poisson_example.pkl.gz')
