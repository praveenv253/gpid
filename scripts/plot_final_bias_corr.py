#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
    #melted_df = pid_df.query('M == 10 and (mode == "bit_of_all" or mode == "fully_redundant")')

    #ncols = melted_df['M'].nunique()
    #nrows = melted_df['mode'].nunique()
    pi_comps = ['imxy', 'uix', 'uiy', 'ri', 'si']
    hue_order = sum(([(pic, True), (pic, False)] for pic in pi_comps), [])

    cols = ['pi_comp', 'biased']
    pid_df['pi_comp_biased'] = pid_df[cols].agg(tuple, axis=1)
    #pid_df = pid_df.drop(columns=cols)

    tab20 = mpl.cm.get_cmap('tab20')
    colors = tab20(np.arange(10))
    #palette = dict(zip(hue_order, tab20(np.arange(10))))

    sns.set_context('talk')

    #for gid, gdf in pid_df.groupby('mode'):
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

    #gdf = pid_df.query('M == 10 and mode == "bit_of_all"')
    #subgt = gt.loc[(10, 'bit_of_all')] #, level=['M', 'mode'])
    gdf = pid_df.query('M == 10 and mode == "fully_redundant"')
    subgt = gt.loc[(10, 'fully_redundant')] #, level=['M', 'mode'])

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
    legend1 = ax.legend(h[:10:2], [r'$I(M\;\!; (X, Y\;\!\;\!))$', '$UI_X$', '$UI_Y$', '$RI$', '$SI$'],
                        loc='upper right', frameon=False)
    ax.add_artist(legend1)


    legend_elements = [Patch(facecolor='k', edgecolor='k', label='Biased'),
                       Patch(facecolor='grey', edgecolor='grey', label='Bias-corrected'),
                       Line2D([0], [0], color='k', lw=2, label='Ground truth')]
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.45, 0.99),
              frameon=False)

    #plt.ylim((0, 2))
    plt.xlabel('Sample size')
    plt.ylabel('Partial information value (bits)')
    #plt.title('Bias correction in the "Bit-of-all" set up')
    plt.title('Bias correction in the "Fully-redundant" set up')
    plt.tight_layout()

    plt.show()
