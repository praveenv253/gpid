#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from mutual_info_bias import debias


if __name__ == '__main__':
    pid_table = pd.read_pickle('../results/mutual_info_bias.pkl.gz')
    value_cols = ['imxy_1', 'imxy_2', 'bias', 'gt_1', 'gt_2']
    M_vals = pid_table['M'].unique()
    modes = pid_table['mode'].unique()

    # Print statistics
    pid_table = pid_table.dropna()
    pid_table = pid_table.set_index(['mode', 'M', 'sample_size', 'trial_id'])[value_cols]

    # Is imxy_1 close to imxy_2 for all trials?
    print('max(abs(imxy_1 - imxy_2)) = ', end='')
    print(abs((pid_table['imxy_1'] - pid_table['imxy_2']).values).max())

    # Are the ground truth values the same?
    print('max(abs(gt_1 - gt_2)) = ', end='')
    print(abs((pid_table['gt_1'] - pid_table['gt_2']).values).max())

    # Does bias correction work?
    d = pid_table.reset_index()
    d['unbiased_imxy_1'] = debias(d['imxy_1'], d['bias'])
    d['unbiased_imxy_2'] = debias(d['imxy_2'], d['bias'])
    gt = d.groupby(['mode', 'M']).agg({'gt_1': 'mean', 'gt_2': 'mean'})

    e = pd.melt(d, id_vars=['mode', 'M', 'sample_size', 'trial_id'],
                value_vars=['imxy_1', 'unbiased_imxy_1'], var_name='unbiased',
                value_name='imxy_1_new').rename(columns={'imxy_1_new': 'imxy_1'})
    e = e.replace({'unbiased': ['imxy_1', 'unbiased_imxy_1']},
                  {'unbiased': [False, True]}).reset_index().sort_values(['M', 'unbiased'])
    hue = e[['M', 'unbiased']].apply(tuple, axis=1)
    e['hue'] = hue

    sns.set_context('talk')
    fig, axs = plt.subplots(nrows=1, ncols=len(modes))
    for i, mode in enumerate(modes):
        axs[i].set_title(mode)
        sns.barplot(ax=axs[i], data=e[e['mode'] == mode], y='imxy_1', x='sample_size', hue='hue', palette='tab20', ci='sd') #errorbar=('pi', 90))
        #g = sns.boxplot(data=e[e['mode'] == mode], y='imxy_1', x='sample_size', hue='hue', palette=(['k']*6), dodge=True, zorder=2)
        #h,l = g.get_legend_handles_labels()
        #plt.legend(h[6:], l[6:], title='(M, debiased)')
        axs[i].legend(title='(M, debiased)')
        for j, M in enumerate(M_vals):
            axs[i].axhline(gt['gt_1'].xs((mode, M), level=('mode', 'M')).item(), color=('C%d' % j))

    plt.show()
