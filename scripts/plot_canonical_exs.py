#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

if __name__ == '__main__':
    pid_table = pd.read_pickle('../results/canonical_exs.pkl.gz')

    pid_defns = ['tilde', 'delta', 'mmi', 'gt']
    linestyles = ['-', '--', ':', '']
    markers = ['', '', '', 'o']

    pid_atoms = ['imxy', 'uix', 'uiy', 'ri', 'si']
    colors = ['k', 'C0', 'C1', 'C2', 'C3']

    suptitlesize = 20
    titlesize = 18
    labelsize = 16
    legendsize = 14
    ticksize = 12

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 5), sharey=True)
    axs = axs.flatten()

    # UI + RI
    ax = axs[0]
    rows = pid_table[pid_table.desc == 'uix+ri']
    for i, pid_defn in enumerate(pid_defns):
        for j, pid_atom in enumerate(pid_atoms):
            ax.plot(rows['id'], rows[(pid_defn, pid_atom)],
                    color=colors[j], linestyle=linestyles[i], marker=markers[i])

    ax.set_title(r'Unique and Redundant', fontsize=titlesize)
    ax.set_xlabel(r'Noise in $Y$  given $X$, $\sigma_{Y|X}$', fontsize=labelsize)
    ax.set_ylabel('Partial information (bits)', fontsize=labelsize)
    ax.set_xticks(rows['id'])
    ax.set_xticklabels(['%.2f' % val for val in rows.sigma_y__x],
                       rotation=45, rotation_mode='anchor', ha='right')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)

    # UI + SI
    ax = axs[1]
    rows = pid_table[pid_table.desc == 'uix+si']
    for i, pid_defn in enumerate(pid_defns):
        for j, pid_atom in enumerate(pid_atoms):
            ax.plot(rows['id'], rows[(pid_defn, pid_atom)], color=colors[j],
                    linestyle=linestyles[i], marker=markers[i])

    ax.set_title(r'Unique and Synergistic', fontsize=titlesize)
    ax.set_xlabel(r'Noise correlation, $\rho$', fontsize=labelsize)
    ax.set_xticks(rows['id'])
    ax.set_xticklabels(['%.2f' % val for val in rows.rho],
                       rotation=45, rotation_mode='anchor', ha='right')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)

    # RI + SI
    ax = axs[2]
    lines = {}  # Dictionary to hold all line handles for legend
    rows = pid_table[pid_table.desc == 'ri+si']
    for i, pid_defn in enumerate(pid_defns):
        for j, pid_atom in enumerate(pid_atoms):
            line = ax.plot(rows['id'], rows[(pid_defn, pid_atom)],
                           color=colors[j], linestyle=linestyles[i],
                           marker=markers[i])[0]
            lines[(colors[j], linestyles[i], markers[i])] = line

    ax.set_title(r'Redundant and Synergistic', fontsize=titlesize)
    ax.set_xlabel(r'Noise correlation, $\rho$', fontsize=labelsize)
    ax.set_xticks(rows['id'])
    ax.set_xticklabels(['%.2f' % val for val in rows.rho],
                       rotation=45, rotation_mode='anchor', ha='right')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)

    fig.suptitle('PIDs for Canonical Gaussian Examples', fontsize=suptitlesize)

    # Legend (tied to last axis)
    handles = [lines[(c, '-', '')] for c in colors]
    texts = [r'$I(M\;\!; (X, Y\;\!\;\!))$', '$UI_X$', '$UI_Y$', '$RI$', '$SI$']
    color_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                             bbox_to_anchor=(1, 0.75), fontsize=legendsize,
                             title='PID component', title_fontsize=labelsize)
    # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#multiple-legends-on-the-same-axes
    ax.add_artist(color_legend)

    handles = [lines[('k', ls, m)] for (ls, m) in zip(linestyles, markers)]
    texts = ['$\sim_G$-PID', '$\delta_G$-PID', 'MMI-PID', 'Ground truth']
    defn_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                            bbox_to_anchor=(1, 0.1), fontsize=legendsize,
                            title='PID definition', title_fontsize=labelsize)

    plt.tight_layout()
    plt.savefig('../figures/canonical-exs.pdf')
    #plt.show()



    #cols = ['desc', 'index', 'tilde', 'delta', 'mmi', 'gt']
    #melted_df = pid_table[cols].melt(id_vars=['desc', 'index'],
    #                                 var_name=['pid_defn', 'pi_comp'],
    #                                 value_name='pi_value')

    #melted_df.replace(inplace=True, to_replace={
    #    'pid_defn': {
    #        'tilde': '$\sim$-PID',
    #        'delta': '$\delta$-PID',
    #        'mmi': 'MMI-PID',
    #        'gt': 'Ground truth',
    #    },
    #    'pi_comp': {
    #        'imxy': '$I(M;(X,Y))$',
    #        'uix': '$UI_X$',
    #        'uiy': '$UI_Y$',
    #        'ri': '$RI$',
    #        'si': '$SI$',
    #    }
    #})
    #melted_df.rename(inplace=True, columns={
    #    'pid_defn': 'PID Definition',
    #    'pi_comp': 'Partial info component',
    #    'pi_value': 'Partial information (bits)',
    #})

    #facets = sns.relplot(kind='line', data=melted_df, col='desc', x='index',
    #                     y='Partial information (bits)',
    #                     hue='Partial info component', style='PID Definition',
    #                     facet_kws=dict(sharex=False))

    #titlesize = 18
    #labelsize = 16

    ## UI_X + RI
    #ax = facets.axes[0, 0]
    #ax.set_title('Unique and Redundant information', fontsize=titlesize)
    #ax.set_xlabel(r'$\sigma_{Y|X}$', fontsize=labelsize)
    #ax.set_ylabel(ax.get_ylabel(), fontsize=labelsize)
    #rows = pid_table[pid_table.desc == 'uix+ri']
    #ax.set_xticks(rows['index'])
    #ax.set_xticklabels(['%.2f' % val for val in rows.sigma_y__x],
    #                   rotation=45, ha='right')

    ## UI_X + SI
    #ax = facets.axes[0, 1]
    #ax.set_title('Unique and Synergistic information', fontsize=titlesize)
    #ax.set_xlabel(r'$\rho$', fontsize=labelsize)
    #rows = pid_table[pid_table.desc == 'uix+si']
    #ax.set_xticks(rows['index'])
    #ax.set_xticklabels(['%.2f' % val for val in rows.rho],
    #                   rotation=45, ha='right')

    ## RI + SI
    #ax = facets.axes[0, 2]
    #ax.set_title('Redundant and Synergistic information', fontsize=titlesize)
    #ax.set_xlabel(r'$\rho$', fontsize=labelsize)
    #rows = pid_table[pid_table.desc == 'ri+si']
    #ax.set_xticks(rows['index'])
    #ax.set_xticklabels(['%.2f' % val for val in rows.rho],
    #                   rotation=45, ha='right')

    #plt.gcf().subplots_adjust(bottom=0.18)
    #plt.show()
