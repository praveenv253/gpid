#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

if __name__ == '__main__':
    pid_table = pd.read_pickle('../results/canonical_exs.pkl.gz')

    cols = ['desc', 'index', 'tilde', 'delta', 'mmi', 'gt']
    melted_df = pid_table[cols].melt(id_vars=['desc', 'index'],
                                     var_name=['pid_defn', 'pi_comp'],
                                     value_name='pi_value')

    melted_df.replace(inplace=True, to_replace={
        'pid_defn': {
            'tilde': '$\sim$-PID',
            'delta': '$\delta$-PID',
            'mmi': 'MMI-PID',
            'gt': 'Ground truth',
        },
        'pi_comp': {
            'imxy': '$I(M;(X,Y))$',
            'uix': '$UI_X$',
            'uiy': '$UI_Y$',
            'ri': '$RI$',
            'si': '$SI$',
        }
    })
    melted_df.rename(inplace=True, columns={
        'pid_defn': 'PID Definition',
        'pi_comp': 'Partial info component',
        'pi_value': 'Partial information (bits)',
    })

    facets = sns.relplot(kind='line', data=melted_df, col='desc', x='index',
                         y='Partial information (bits)',
                         hue='Partial info component', style='PID Definition',
                         facet_kws=dict(sharex=False))

    titlesize = 18
    labelsize = 16

    # UI_X + RI
    ax = facets.axes[0, 0]
    ax.set_title('Unique and Redundant information', fontsize=titlesize)
    ax.set_xlabel(r'$\sigma_{Y|X}$', fontsize=labelsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=labelsize)
    rows = pid_table[pid_table.desc == 'uix+ri']
    ax.set_xticks(rows['index'])
    ax.set_xticklabels(['%.2f' % val for val in rows.sigma_y__x],
                       rotation=45, ha='right')

    # UI_X + SI
    ax = facets.axes[0, 1]
    ax.set_title('Unique and Synergistic information', fontsize=titlesize)
    ax.set_xlabel(r'$\rho$', fontsize=labelsize)
    rows = pid_table[pid_table.desc == 'uix+si']
    ax.set_xticks(rows['index'])
    ax.set_xticklabels(['%.2f' % val for val in rows.rho],
                       rotation=45, ha='right')

    # RI + SI
    ax = facets.axes[0, 2]
    ax.set_title('Redundant and Synergistic information', fontsize=titlesize)
    ax.set_xlabel(r'$\rho$', fontsize=labelsize)
    rows = pid_table[pid_table.desc == 'ri+si']
    ax.set_xticks(rows['index'])
    ax.set_xticklabels(['%.2f' % val for val in rows.rho],
                       rotation=45, ha='right')

    plt.gcf().subplots_adjust(bottom=0.18)
    plt.show()
