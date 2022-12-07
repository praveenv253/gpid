#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    pid_table = pd.read_pickle('../results/bootstrap_ci_unbiased_corrected.pkl.gz')

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

    ncols = pid_df['M'].nunique()
    nrows = pid_df['mode'].nunique()
    pi_comps = ['imxy', 'uix', 'uiy', 'ri', 'si']
    hue_order = sum(([(pic, False), (pic, True)] for pic in pi_comps), [])

    cols = ['pi_comp', 'bootstrap']
    pid_df['pi_comp_bootstrap'] = pid_df[cols].agg(tuple, axis=1)
    pid_df = pid_df.drop(columns=cols)

    #sns.set_context('notebook')
    #for gid, gdf in pid_df.groupby('mode'):
    axs = sns.catplot(kind='bar', data=pid_df, x='sample_size', y='pi_value',
                      hue='pi_comp_bootstrap', row='M', col='mode', #col='M',
                      hue_order=hue_order, palette='tab20', sharey=False,
                      ci=None, zorder=-1)
    axs.map_dataframe(sns.boxplot, x='sample_size', y='pi_value',
                      hue='pi_comp_bootstrap', palette='tab20',
                      hue_order=hue_order, linewidth=1, fliersize=2,
                      boxprops={'alpha': 0.3}, zorder=5)
        #plt.suptitle('mode = %s' % gid)
        #plt.tight_layout(rect=(0, 0, 0.85, 1))

    plt.show()
