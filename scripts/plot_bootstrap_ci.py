#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

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

    ncols = pid_df['M'].nunique()
    nrows = pid_df['mode'].nunique()
    pi_comps = ['imxy', 'uix', 'uiy', 'ri', 'si']
    hue_order = sum(([(pic, False), (pic, True)] for pic in pi_comps), [])

    cols = ['pi_comp', 'bootstrap']
    pid_df['pi_comp_bootstrap'] = pid_df[cols].agg(tuple, axis=1)

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

    # Compute some goodness metrics for the bootstrap CIs
    metrics_df = pid_df.set_index(index_cols + ['pi_comp'])['pi_value']
    metrics_df = metrics_df.unstack('trial_id')
    metrics = pd.DataFrame()
    metrics['upper_ci'] = metrics_df.quantile(q=0.9, axis=1, interpolation='lower')
    metrics['lower_ci'] = metrics_df.quantile(q=0.1, axis=1, interpolation='higher')
    metrics['median'] = metrics_df.median(axis=1)

    metrics = metrics.unstack('bootstrap')
    metrics['jaccard'] = metrics.agg(jaccard_index, axis=1)
    metrics['median_diff'] = (metrics[('median', False)] - metrics[('median', True)]).abs()
    metrics_to_plot = metrics[['jaccard', 'median_diff']].droplevel('bootstrap', axis=1)
    metrics_to_plot.columns.set_names(['metric'], inplace=True)
    metrics_to_plot = metrics_to_plot.stack()
    metrics_to_plot = metrics_to_plot.reset_index().rename(columns={0: 'metric_value'})
    metrics_to_plot['pi_comp_metric'] = metrics_to_plot[['pi_comp', 'metric']].agg(tuple, axis=1)

    print(metrics_to_plot)

    hue_order = sum(([(pic, 'median_diff'), (pic, 'jaccard')] for pic in pi_comps), [])
    sns.catplot(kind='bar', data=metrics_to_plot, x='sample_size', y='metric_value',
                hue='pi_comp_metric', row='M', col='mode', #col='M',
                hue_order=hue_order, palette='tab20', sharey=False)

    plt.show()
