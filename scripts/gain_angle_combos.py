#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import pandas as pd

from gpid.tilde_pid import exact_gauss_tilde_pid
from gpid.estimate import approx_pid_from_cov
from gpid.mmi_pid import mmi_pid
from gpid.generate import generate_cov_from_config


def gain_combos():
    pid_table = pd.read_pickle('../results/pid_table.pkl.gz')

    rng = np.random.default_rng(10)

    pid_defn_names = ['tilde', 'delta', 'mmi']
    pid_defns = [exact_gauss_tilde_pid, approx_pid_from_cov, mmi_pid]

    config_cols = ['dm', 'dx', 'dy', 'id1', 'id2', 'random_rotn']
    pid_cols = ['imxy', 'uix', 'uiy', 'ri', 'si']

    cols = []
    vals = []

    indices = list(pid_table.iloc[:10].index)

    # Create new gain combinations
    for i in indices:
        print(i, end=': ', flush=True)

        for j in indices:
            print(j, end=' ', flush=True)

            randint = None # False # rng.integers(2**16)
            cov, *d = generate_cov_from_config(pid_table=pid_table, id1=i, id2=j,
                                               random_rotn=randint)

            cols = [(col, '') for col in config_cols]
            vals = [*d, i, j, randint]

            for pid_name, pid_defn in zip(pid_defn_names, pid_defns):
                ret = pid_defn(cov, *d)
                imxy, uix, uiy, ri, si = ret[2], *ret[-4:]

                cols.extend([(pid_name, col) for col in pid_cols])
                vals.extend([imxy, uix, uiy, ri, si])

            # Ground truth values
            # No need to check as NaNs will be propagated automatically
            gt_vals = list(pid_table.loc[i, 'gt'] + pid_table.loc[j, 'gt'])
            gt_exists = pid_table.loc[[i, j], 'gt_exists'].all()

            pid_name = 'gt'
            cols.extend([(pid_name, col) for col in pid_cols])
            vals.extend(gt_vals)
            cols.append(('gt_exists', ''))
            vals.append(gt_exists)

            row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
            pid_table = pd.concat((pid_table, row), ignore_index=True)

        print()

    print(pid_table)
    pid_table.to_pickle('../results/pid_table_gain_combos.pkl.gz')

    assert np.allclose((pid_table['tilde'] - pid_table['gt']).dropna().to_numpy(),
                       0, atol=1e-5)


def angle_combos():
    # Has to be run after running gain_combos and generating its savefile
    pid_table = pd.read_pickle('../results/pid_table_gain_combos.pkl.gz')

    rng = np.random.default_rng(20)

    pid_defn_names = ['tilde', 'delta', 'mmi']
    pid_defns = [exact_gauss_tilde_pid, approx_pid_from_cov, mmi_pid]

    config_cols = ['dm', 'dx', 'dy', 'id1', 'id2', 'random_rotn']
    pid_cols = ['imxy', 'uix', 'uiy', 'ri', 'si']

    cols = []
    vals = []

    indices = list(pid_table.iloc[10:20].index)

    # Create new angle combinations
    for i in indices:
        print(i, end=': ', flush=True)

        for j in indices:
            print(j, end=' ', flush=True)

            randint = None # False # rng.integers(2**16)
            cov, *d = generate_cov_from_config(pid_table=pid_table, id1=i, id2=j,
                                               random_rotn=randint)

            cols = [(col, '') for col in config_cols]
            vals = [*d, i, j, randint]

            for pid_name, pid_defn in zip(pid_defn_names, pid_defns):
                ret = pid_defn(cov, *d)
                imxy, uix, uiy, ri, si = ret[2], *ret[-4:]

                cols.extend([(pid_name, col) for col in pid_cols])
                vals.extend([imxy, uix, uiy, ri, si])

            # Ground truth values
            # No need to check as NaNs will be propagated automatically
            gt_vals = list(pid_table.loc[i, 'gt'] + pid_table.loc[j, 'gt'])
            gt_exists = pid_table.loc[[i, j], 'gt_exists'].all()
            cols.extend([('gt', col) for col in pid_cols])
            vals.extend(gt_vals)
            cols.append(('gt_exists', ''))
            vals.append(gt_exists)

            # Expected tilde PID values
            tilde_expected_vals = list(pid_table.loc[i, 'tilde']
                                       + pid_table.loc[j, 'tilde'])
            cols.extend([('tilde_exp', col) for col in pid_cols])
            vals.extend(tilde_expected_vals)

            row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
            pid_table = pd.concat((pid_table, row), ignore_index=True)

        print()

    print(pid_table)
    pid_table.to_pickle('../results/pid_table_combos.pkl.gz')

    assert np.allclose((pid_table['tilde'] - pid_table['gt']).dropna().to_numpy(),
                       0, atol=1e-5)


if __name__ == '__main__':
    #gain_combos()
    angle_combos()
