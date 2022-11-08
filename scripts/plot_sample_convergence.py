#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    pid_table = pd.read_pickle('../results/sample_convergence.pkl.gz')

    pid_table.loc[(pid_table['M'] == 20)
                  & (pid_table['sample_size'] == 50), 'tilde'] = np.nan

    cols = ['M', 'sample_size', 'mode', 'tilde']
    melted_df = pid_table[cols].melt(id_vars=cols[:3],
                                     var_name=['pid_defn', 'pi_comp'],
                                     value_name='pi_value')

    ncols = melted_df['M'].nunique()
    nrows = melted_df['mode'].nunique()

    # Actual PID values
    fig, axs = plt.subplots(figsize=(14, 8), nrows=nrows, ncols=ncols)
    axs = axs.flatten()
    for ax, (group_id, group_df) in zip(axs, melted_df.groupby(['mode', 'M'])):
        for i, pi_val in enumerate(group_df[group_df.sample_size.isna()].pi_value):
            ax.axhline(pi_val, c=('C%d' % i))
        sns.boxplot(data=group_df, x='sample_size', y='pi_value', hue='pi_comp',
                    boxprops={'linewidth': 0}, ax=ax)
        ax.set_title(group_id)
    plt.tight_layout()

    # De-biased PID values
    fig, axs = plt.subplots(figsize=(14, 8), nrows=nrows, ncols=ncols)
    axs = axs.flatten()

    pid_df = pid_table.dropna().set_index(['M', 'mode', 'sample_size', 'trial_id'])[['tilde']]
    gt = pid_table.loc[pid_table['sample_size'].isna()].set_index(['M', 'mode'])[['tilde']]

    bias = pid_df.copy()
    bias['bias'] = 2 * pid_df.index.get_level_values('M')**2 / pid_df.index.get_level_values('sample_size')
    debias_factor = 1 - bias['bias'] / pid_df[('tilde', 'imxy')]

    #debias_factor = gt[('tilde', 'imxy')] / pid_df[('tilde', 'imxy')].groupby(level=['M', 'mode', 'sample_size']).agg('mean')

    pid_table_norm = pid_table.set_index(['M', 'mode', 'sample_size', 'trial_id']).copy()
    pid_df = pid_df.mul(debias_factor, axis=0)
    pid_table_norm.update(pid_df)
    pid_table_norm = pid_table_norm.reset_index()

    #pid_table_norm.loc[non_gt_rows, 'tilde'] = pid_table.loc[non_gt_rows, 'tilde'].to_numpy() * (1 - bias / pid_table.loc[non_gt_rows, ('tilde', 'imxy')]).to_numpy()[:, None]
    melted_df = pid_table_norm[cols].melt(id_vars=cols[:3],
                                          var_name=['pi_defn', 'pi_comp'],
                                          value_name='pi_value')
    for ax, (group_id, group_df) in zip(axs, melted_df.groupby(['mode', 'M'])):
        for i, pi_val in enumerate(group_df[group_df.sample_size.isna()].pi_value):
            ax.axhline(pi_val, c=('C%d' % i))
        sns.boxplot(data=group_df, x='sample_size', y='pi_value', hue='pi_comp',
                    boxprops={'linewidth': 0}, ax=ax)
        ax.set_title(group_id)
    plt.tight_layout()

    #bias = {}
    #for group_id, group_df in pid_table.groupby(['sample_size', 'M', 'mode']):
    #    gt = pid_table[(pid_table['M'] == group_id[1])
    #                   & (pid_table['mode'] == group_id[2])
    #                   & pid_table['sample_size'].isna()][('tilde', 'imxy')].iloc[0]
    #    bias[group_id] = group_df[('tilde', 'imxy')].mean() - gt

    #bias_df = pd.DataFrame([(*k, v) for k, v in bias.items()],
    #                       columns=['sample_size', 'M', 'mode', 'bias'])

    #bias_df['bias*N'] = bias_df['bias'] * bias_df['sample_size'] / (bias_df['M']**2)
    #sns.lineplot(data=bias_df, x='sample_size', y='bias*N', hue='M')

    #print(bias_df)

    plt.show()
