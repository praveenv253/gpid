#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    pid_table = pd.read_pickle('../results/sample_convergence_unbiased_v3.pkl.gz')

    #pid_table.loc[(pid_table['M'] == 20)
    #              & (pid_table['sample_size'] == 50), 'tilde'] = np.nan

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

    #pid_df = pid_df[pid_df['pi_comp'] != 'imxy']
    #gt = gt.drop(columns=['imxy'])

    #cols = ['M', 'sample_size', 'mode', 'tilde']
    #melted_df = pid_table[cols].melt(id_vars=cols[:3],
    #                                 var_name=['pid_defn', 'pi_comp'],
    #                                 value_name='pi_value')
    #melted_df = pid_df[pid_df['biased'] == True]
    melted_df = pid_df

    ncols = melted_df['M'].nunique()
    nrows = melted_df['mode'].nunique()
    pi_comps = ['imxy', 'uix', 'uiy', 'ri', 'si']
    hue_order = sum(([(pic, True), (pic, False)] for pic in pi_comps), [])

    cols = ['pi_comp', 'biased']
    pid_df['pi_comp_biased'] = pid_df[cols].agg(tuple, axis=1)
    #pid_df = pid_df.drop(columns=cols)

    tab20 = mpl.cm.get_cmap('tab20')
    colors = tab20(np.arange(10))
    #palette = dict(zip(hue_order, tab20(np.arange(10))))

    sns.set_context('notebook')
    for gid, gdf in pid_df.groupby('mode'):
        #fig, axs = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
        #axs = axs.flatten()
        #for ax, (subgid, subgdf) in zip(axs, gdf.groupby(['M', 'sample_size'])):
        #    sns.barplot(ax=ax,
        #                data=subgdf,
        #                x='pi_comp',
        #                y='pi_value',
        #                hue='biased',
        #                hue_order=(True, False),
        #                ci=None, zorder=-1)
        #    sns.boxplot(ax=ax,
        #                data=subgdf,
        #                x='pi_comp',
        #                y='pi_value',
        #                hue='biased',
        #                hue_order=(True, False),
        #                zorder=5)
        axs = sns.catplot(kind='bar', data=gdf, y='pi_value', x='pi_comp', #x='sample_size',
                          #hue='pi_comp_biased', row='M', #col='mode', #col='M',
                          row='M', col='sample_size', #hue='pi_comp_biased',
                          hue='biased',
                          hue_order=(True, False),
                          #hue_order=hue_order,
                          sharey='row',
                          ci=None, zorder=-1)
        axs.map_dataframe(sns.boxplot, y='pi_value', x='pi_comp', #x='sample_size',
                          palette='tab20', #hue='pi_comp_biased', hue_order=hue_order,
                          hue='biased', hue_order=(True, False),
                          linewidth=1, #fliersize=2,
                          zorder=5)
                          #boxprops={'alpha': 0.3}, zorder=5)

        for ax in axs.axes.flatten():
            #ax.get_legend().remove()
            boxes = ax.findobj(mpl.patches.Rectangle)
            for color, box in zip(list(colors)[::2] + list(colors)[1::2], boxes):
                box.set_facecolor(color)
            boxes = ax.findobj(mpl.patches.PathPatch)
            for color, box in zip(np.hstack((colors[:, :3], 0.3 * colors[:, [3]])), boxes):
                box.set_facecolor(color)

        plt.suptitle('mode = %s' % gid)
        #plt.tight_layout(rect=(0, 0, 0.85, 1))

    plt.show()

    ## Actual PID values
    #fig, axs = plt.subplots(figsize=(14, 8), nrows=nrows, ncols=ncols, sharex=True)
    #axs = axs.flatten()
    #for ax, (group_id, group_df) in zip(axs, melted_df.groupby(['mode', 'M'])):
    #    #for i, pi_val in enumerate(group_df[group_df.sample_size.isna()].pi_value):
    #    gt_mode_m = gt.xs(group_id, level=('mode', 'M')).iloc[0]
    #    # Normalizing ground truth
    #    #gt_mode_m = gt_mode_m.values / gt_mode_m['imxy']
    #    for i, pi_val in enumerate(gt_mode_m, start=1):
    #        ax.axhline(pi_val, c=('C%d' % i))
    #    sns.boxplot(data=group_df, x='sample_size', y='pi_value', hue='pi_comp',
    #                hue_order=hue_order, boxprops={'linewidth': 0}, ax=ax)
    #    ax.get_legend().remove()
    #    ax.set_title(group_id)
    #plt.tight_layout()
    #plt.suptitle('Biased PID values')



    #melted_df = pid_df[pid_df['biased'] == False]
    #fig, axs = plt.subplots(figsize=(14, 8), nrows=nrows, ncols=ncols, sharex=True)
    #axs = axs.flatten()
    #for ax, (group_id, group_df) in zip(axs, melted_df.groupby(['mode', 'M'])):
    #    for i, pi_val in enumerate(gt.xs(group_id, level=('mode', 'M')).iloc[0], start=1):
    #        ax.axhline(pi_val, c=('C%d' % i))
    #    sns.boxplot(data=group_df, x='sample_size', y='pi_value', hue='pi_comp',
    #                hue_order=hue_order, boxprops={'linewidth': 0}, ax=ax)
    #    ax.get_legend().remove()
    #    ax.set_title(group_id)
    #plt.tight_layout()
    #plt.suptitle('Unbiased PID values')




    ## De-biased PID values
    #fig, axs = plt.subplots(figsize=(14, 8), nrows=nrows, ncols=ncols)
    #axs = axs.flatten()

    ##pid_df = pid_table.dropna().set_index(['M', 'mode', 'sample_size', 'trial_id'])[['tilde', 'cv_mi']]
    #pid_df = pid_table.dropna().set_index(['M', 'mode', 'sample_size', 'trial_id'])[['tilde']]
    #gt = pid_table.loc[pid_table['sample_size'].isna()].set_index(['M', 'mode'])[['tilde']]

    ## Debias based on a rough estimate of bias = 2 * M^2 / N, where N = sample_size
    #bias = pid_df.copy()
    #bias['bias'] = 2 * pid_df.index.get_level_values('M')**2 / pid_df.index.get_level_values('sample_size')
    ##bias = bias.reset_index()
    ##entropy_bias = lambda n, p: sum(np.log2(1 - k / n) for k in range(1, p + 1)) / 2
    ##mi_bias = lambda n, p: entropy_bias(n, p) + entropy_bias(n, 2*p) - entropy_bias(n, 3*p)
    ##bias['bias'] = bias.apply(lambda x: mi_bias(x['M'].item(), x['sample_size'].item()), axis=1)
    ##bias = bias.set_index(['M', 'mode', 'sample_size', 'trial_id'])
    #debias_factor = 1 - bias['bias'] / pid_df[('tilde', 'imxy')]

    ## Debias based on the ground truth mutual information
    ##debias_factor = gt[('tilde', 'imxy')] / pid_df[('tilde', 'imxy')].groupby(level=['M', 'mode', 'sample_size']).agg('mean')

    ## Debias based on cross-validated mutual information
    ##debias_factor = pid_df['cv_mi'] / pid_df[('tilde', 'imxy')]

    #pid_table_norm = pid_table.set_index(['M', 'mode', 'sample_size', 'trial_id']).copy()
    #pid_df = pid_df.mul(debias_factor, axis=0)
    #pid_table_norm.update(pid_df)
    #pid_table_norm = pid_table_norm.reset_index()

    ##pid_table_norm.loc[non_gt_rows, 'tilde'] = pid_table.loc[non_gt_rows, 'tilde'].to_numpy() * (1 - bias / pid_table.loc[non_gt_rows, ('tilde', 'imxy')]).to_numpy()[:, None]
    #melted_df = pid_table_norm[cols].melt(id_vars=cols[:3],
    #                                      var_name=['pi_defn', 'pi_comp'],
    #                                      value_name='pi_value')
    #for ax, (group_id, group_df) in zip(axs, melted_df.groupby(['mode', 'M'])):
    #    for i, pi_val in enumerate(group_df[group_df.sample_size.isna()].pi_value):
    #        ax.axhline(pi_val, c=('C%d' % i))
    #    sns.boxplot(data=group_df, x='sample_size', y='pi_value', hue='pi_comp',
    #                boxprops={'linewidth': 0}, ax=ax)
    #    ax.set_title(group_id)
    #plt.tight_layout()

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

    #plt.show()
