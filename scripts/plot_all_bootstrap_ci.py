#!/usr/bin/env python3

from __future__ import print_function, division

import itertools as it

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import pandas as pd
import seaborn as sns


def jaccard_index(x):
    gt_hi, gt_lo = x[[('upper_ci', False), ('lower_ci', False)]]
    bs_hi, bs_lo = x[[('upper_ci', True), ('lower_ci', True)]]

    numer = min(gt_hi, bs_hi) - max(gt_lo, bs_lo)
    denom = max(gt_hi, bs_hi) - min(gt_lo, bs_lo) - max(-numer, 0)
    numer = max(numer, 0)

    try:
        jaccard_index = numer / denom
    except ZeroDivisionError:
        jaccard_index = np.nan

    return jaccard_index


if __name__ == '__main__':
    pid_table = pd.read_pickle('../results/bootstrap_ci_v3.pkl.gz')

    gt = pid_table.loc[pid_table['sample_size'].isna()].set_index(['M', 'mode'])['tilde']
    gt.columns.set_names('pi_comp', inplace=True)

    index_cols = ['mode', 'M', 'sample_size', 'trial_id', 'bootstrap']

    pid_df = pid_table.dropna().set_index(index_cols)['tilde']

    # Normalizing
    #pid_df['tilde'] = pid_df['tilde'].div(pid_df[('tilde', 'imxy')], axis=0)

    pid_df.columns.set_names(['pi_comp'], inplace=True)
    pid_df = pid_df.stack().reset_index().rename(columns={0: 'pi_value'})

    #pid_df = pid_df[pid_df['pi_comp'] != 'imxy']
    #gt = gt.drop(columns=['imxy'])

    pi_comps = ['imxy', 'uix', 'uiy', 'ri', 'si']
    hue_order = sum(([(pic, False), (pic, True)] for pic in pi_comps), [])

    cols = ['pi_comp', 'bootstrap']
    pid_df['pi_comp_bootstrap'] = pid_df[cols].agg(tuple, axis=1)

    sns.set_context('talk')

    mvals = pid_df['M'].unique()
    modes = pid_df['mode'].unique()

    for m, mode in it.product(mvals, modes):
        print((m, mode))

        gdf = pid_df.query('M == @m and mode == @mode')
        subgt = gt.loc[(m, mode)]

        plt.figure(figsize=(8, 6))

        sns.barplot(data=gdf,
                    y='pi_value',
                    x='sample_size',
                    hue='pi_comp_bootstrap', hue_order=hue_order,
                    palette='tab20',
                    ci=None, zorder=-1)
        sns.boxplot(data=gdf,
                    y='pi_value',
                    x='sample_size',
                    hue='pi_comp_bootstrap', hue_order=hue_order,
                    palette='tab20',
                    linewidth=1,
                    zorder=5)

        for i, (key, val) in enumerate(subgt.items()):
            plt.axhline(val, color='C%d'%i, zorder=-2)

        h, l = plt.gca().get_legend_handles_labels()
        ax = plt.gca()
        legend1 = ax.legend(h[:10:2], [r'$I(M\;\!; (X, Y\;\!\;\!))$', '$UI_X$', '$UI_Y$', '$RI$', '$SI$'],
                            loc='upper right', frameon=False)
        ax.add_artist(legend1)

        legend_elements = [Patch(facecolor='k', edgecolor='k', label='True CI'),
                           Patch(facecolor='grey', edgecolor='grey', label='Bootstrap CI'),
                           Line2D([0], [0], color='k', lw=2, label='Ground truth')]
        ax.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.45, 0.99),
                  frameon=False)

        if m == 10 and mode != 'fully_redundant':
            y0, y1 = plt.ylim()
            plt.ylim((y0, 1.3 * y1))

        plt.xlabel('Sample size')
        plt.ylabel('PID value (bits) with CIs')
        plt.title('Bootstrap CIs in the "%s" setup ($d = %d$)' % (mode.replace('_', '-'), m))
        plt.tight_layout()

        plt.savefig('../figures/bootstrap-ci--%s--%d.pdf' % (mode, m))
        plt.close()

    #plt.show()
