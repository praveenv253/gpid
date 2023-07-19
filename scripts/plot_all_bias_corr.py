#!/usr/bin/env python3

from __future__ import print_function, division

import itertools as it

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    pid_table = pd.read_pickle('../results/sample_convergence_unbiased_v3.pkl.gz')

    gt = pid_table.loc[pid_table['sample_size'].isna()].set_index(['M', 'mode'])['tilde']
    gt.columns.set_names('pi_comp', inplace=True)

    index_cols = ['mode', 'M', 'sample_size', 'trial_id']

    pid_df = pid_table.dropna().set_index(index_cols)[['tilde', 'unbiased_tilde']]

    # Normalizing
    #pid_df['tilde'] = pid_df['tilde'].div(pid_df[('tilde', 'imxy')], axis=0)

    pid_df.columns.set_names(['biased', 'pi_comp'], inplace=True)
    pid_df = pid_df.rename({'tilde': True, 'unbiased_tilde': False}, axis=1, level='biased')
    pid_df = pid_df.stack(level=(0, 1)) #- gt.stack()
    pid_df = pid_df.reset_index().rename(columns={0: 'pi_value'})

    pi_comps = ['imxy', 'uix', 'uiy', 'ri', 'si']
    hue_order = sum(([(pic, True), (pic, False)] for pic in pi_comps), [])

    cols = ['pi_comp', 'biased']
    pid_df['pi_comp_biased'] = pid_df[cols].agg(tuple, axis=1)

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
                    hue='pi_comp_biased', hue_order=hue_order,
                    palette='tab20',
                    ci=None, zorder=-1)
        sns.boxplot(data=gdf,
                    y='pi_value',
                    x='sample_size',
                    hue='pi_comp_biased', hue_order=hue_order,
                    palette='tab20',
                    linewidth=1,
                    zorder=5)

        for i, (key, val) in enumerate(subgt.items()):
            plt.axhline(val, color='C%d'%i, zorder=-2)

        h, l = plt.gca().get_legend_handles_labels()
        ax = plt.gca()
        legend1 = ax.legend(h[:10:2], ['$I(M\;\!; (X, Y\;\!\;\!))$', '$UI_X$', '$UI_Y$', '$RI$', '$SI$'],
                            loc='upper right', frameon=False)
        ax.add_artist(legend1)

        legend_elements = [Patch(facecolor='k', edgecolor='k', label='Biased'),
                           Patch(facecolor='grey', edgecolor='grey', label='Bias-corrected'),
                           Line2D([0], [0], color='k', lw=2, label='Ground truth')]
        ax.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.45, 0.99),
                  frameon=False)

        plt.xlabel('Sample size')
        plt.ylabel('Partial information value (bits)')
        plt.title('Bias correction in the "%s" set up ($d = %d$)' % (mode.replace('_', '-'), m))
        plt.tight_layout()

        plt.savefig('../figures/bias-corr--%s--%d.pdf' % (mode, m))
        plt.close()

    #plt.show()
