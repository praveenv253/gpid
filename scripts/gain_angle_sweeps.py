#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import pandas as pd

from gpid.tilde_pid import exact_gauss_tilde_pid
from gpid.estimate import approx_pid_from_cov
from gpid.mmi_pid import mmi_pid
from gpid.generate import generate_cov_from_config


if __name__ == '__main__':
    dm, dx, dy = 2, 2, 2

    pid_table = pd.DataFrame()

    pid_defn_names = ['tilde', 'delta', 'mmi']
    pid_defns = [exact_gauss_tilde_pid, approx_pid_from_cov, mmi_pid]

    config_cols = ['desc', 'id', 'dm', 'dx', 'dy', 'gain_x', 'gain_y', 'theta']
    pid_cols = ['imxy', 'uix', 'uiy', 'ri', 'si']

    theta = 0
    gain_y = np.sqrt(2)
    gains = np.linspace(0, 3, 10)
    gains[3] = 0.99  # A gain of 1 is unstable

    for i, gain_x in enumerate(gains):
        print(i, end=' ', flush=True)

        cols = [(col, '') for col in config_cols]
        vals = ['gain', i, dm, dx, dy, gain_x, gain_y, theta]

        cov, *dims = generate_cov_from_config(d=(dm, dx, dy), gain=(gain_x, gain_y),
                                              theta=theta)

        for pid_name, pid_defn in zip(pid_defn_names, pid_defns):
            ret = pid_defn(cov, dm, dx, dy)
            imxy, uix, uiy, ri, si = ret[2], *ret[-4:]

            cols.extend([(pid_name, col) for col in pid_cols])
            vals.extend([imxy, uix, uiy, ri, si])

        # Ground truth value: equal to the MMI-PID because theta = 0 and gain_y = 1
        if theta == 0 and gain_y == 1:
            gt_exists = True
            # NOTE: Takes all PID values from above, since MMI runs second in the loop
        else:
            gt_exists = True
            ret = mmi_pid(cov[np.ix_([0, 2, 4], [0, 2, 4])], 1, 1, 1)
            pid_vals = np.r_[ret[2], ret[-4:]]
            ret = mmi_pid(cov[np.ix_([1, 3, 5], [1, 3, 5])], 1, 1, 1)
            pid_vals += np.r_[ret[2], ret[-4:]]
            imxy, uix, uiy, ri, si = pid_vals
        #else:
        #    gt_exists = False
        #    uix, uiy, ri, si = [np.nan,] * 4  # Reset values to nan

        pid_name = 'gt'
        cols.extend([(pid_name, col) for col in pid_cols])
        vals.extend([imxy, uix, uiy, ri, si])
        cols.append(('gt_exists', ''))
        vals.append(gt_exists)

        row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
        pid_table = pd.concat((pid_table, row), ignore_index=True)

    print()

    gain_x, gain_y = np.sqrt(2), np.sqrt(2)
    thetas = np.r_[0, np.logspace(-1, np.log10(np.pi/2), 9)]
    for i, theta in enumerate(thetas):
        print(i, end=' ', flush=True)

        cols = [(col, '') for col in config_cols]
        vals = ['angle', i, dm, dx, dy, gain_x, gain_y, theta]

        cov, *dims = generate_cov_from_config(d=(dm, dx, dy), gain=(gain_x, gain_y),
                                              theta=theta)

        for pid_name, pid_defn in zip(pid_defn_names, pid_defns):
            ret = pid_defn(cov, dm, dx, dy)
            imxy, uix, uiy, ri, si = ret[2], *ret[-4:]

            cols.extend([(pid_name, col) for col in pid_cols])
            vals.extend([imxy, uix, uiy, ri, si])

        # Ground truth value: exists only if theta = 0 or theta = pi/2
        mi = lambda snr: 0.5 * np.log2(1 + snr)
        if theta == 0:
            gt_exists = True
            uix = max(0, mi(gain_x**2) - mi(1))
            uiy = max(0, mi(gain_y**2) - mi(1))
            ri = min(mi(gain_x**2), mi(1)) + min(mi(gain_y**2), mi(1))
            si = imxy - uix - uiy - ri  # NOTE: Taking imxy from for loop!
        elif np.isclose(theta, np.pi/2):
            gt_exists = True
            uix = max(0, mi(gain_x**2) - mi(gain_y**2))
            uiy = max(0, mi(gain_y**2) - mi(gain_x**2))
            ri = min(mi(gain_x**2), mi(gain_y**2)) + mi(1)
            si = imxy - uix - uiy - ri  # NOTE: Taking imxy from for loop!
        else:
            gt_exists = False
            uix, uiy, ri, si = [np.nan,] * 4

        pid_name = 'gt'
        cols.extend([(pid_name, col) for col in pid_cols])
        vals.extend([imxy, uix, uiy, ri, si])
        cols.append(('gt_exists', ''))
        vals.append(gt_exists)

        row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
        pid_table = pd.concat((pid_table, row), ignore_index=True)

    print()

    pid_table.to_pickle('../results/gain_angle_exs_099.pkl.gz')

    print(pid_table)
    assert np.allclose((pid_table['tilde'] - pid_table['gt']).dropna().to_numpy(), 0, atol=1e-5)
