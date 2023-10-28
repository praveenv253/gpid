#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns


if __name__ == '__main__':
    pid_table = pd.read_pickle('../results/gain_angle_exs.pkl.gz')

    pid_defns = ['tilde', 'delta', 'mmi', 'gt']
    linestyles = ['-', '--', ':', '']
    markers = ['', '', '', 'o']

    pid_atoms = ['imxy', 'uix', 'uiy', 'ri', 'si']
    colors = ['k', 'C0', 'C1', 'C2', 'C3']

    titlesize = 18
    labelsize = 16
    legendsize = 14
    ticksize = 12

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs = axs.flatten()

    # Gain
    ax = axs[0]
    gain_rows = pid_table[pid_table.desc == 'gain']
    for i, pid_defn in enumerate(pid_defns):
        for j, pid_atom in enumerate(pid_atoms):
            ax.plot(gain_rows['id'], gain_rows[(pid_defn, pid_atom)],
                    color=colors[j], linestyle=linestyles[i], marker=markers[i])

    ax.set_title(r'Increasing Gain in $X_1$', fontsize=titlesize)
    ax.set_xlabel(r'Gain in $X_1$, $\alpha$', fontsize=labelsize)
    ax.set_ylabel('Partial information (bits)', fontsize=labelsize)
    ax.set_xticks(gain_rows['id'])
    ax.set_xticklabels(['%.2f' % val for val in gain_rows.gain_x],
                       rotation=45, rotation_mode='anchor', ha='right')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)


    # Gain error
    ax = axs[1]
    gain_rows = pid_table[pid_table.desc == 'gain']
    for i, pid_defn in enumerate(pid_defns[:-1]):
        for j, pid_atom in enumerate(pid_atoms):
            ax.semilogy(gain_rows['id'],
                        abs(gain_rows[(pid_defn, pid_atom)] - gain_rows[('gt', pid_atom)]),
                        color=colors[j], linestyle=linestyles[i], marker=markers[i])

    ax.set_title(r'Error in gain example', fontsize=titlesize)
    ax.set_xlabel(r'Gain in $X_1$, $\alpha$', fontsize=labelsize)
    ax.set_ylabel('Error (bits)', fontsize=labelsize)
    ax.set_xticks(gain_rows['id'])
    ax.set_xticklabels(['%.2f' % val for val in gain_rows.gain_x],
                       rotation=45, rotation_mode='anchor', ha='right')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)


    # Angle
    ax = axs[2]
    lines = {}  # Dictionary to hold all line handles for legend
    angle_rows = pid_table[pid_table.desc == 'angle']
    for i, pid_defn in enumerate(pid_defns):
        for j, pid_atom in enumerate(pid_atoms):
            line = ax.plot(angle_rows['id'], angle_rows[(pid_defn, pid_atom)],
                           color=colors[j], linestyle=linestyles[i],
                           marker=markers[i])[0]
            lines[(colors[j], linestyles[i], markers[i])] = line

    ax.set_title(r'Rotation of $X$ w.r.t. $Y$', fontsize=titlesize)
    ax.set_xlabel(r'Angle, $\theta$', fontsize=labelsize)
    ax.set_ylabel('Partial information (bits)', fontsize=labelsize)
    angle_rows = pid_table[pid_table.desc == 'angle']
    ax.set_xticks(angle_rows['id'])
    ax.set_xticklabels(['%.2f' % val for val in angle_rows.theta],
                       rotation=45, rotation_mode='anchor', ha='right')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)

    # Legend (tied to second axis)
    handles = [lines[(c, '-', '')] for c in colors]
    texts = [r'$I(M\;\!; (X, Y\;\!\;\!))$', '$UI_X$', '$UI_Y$', '$RI$', '$SI$']
    color_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                             bbox_to_anchor=(1, 0.77), fontsize=legendsize,
                             title='PID component', title_fontsize=labelsize)
    # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#multiple-legends-on-the-same-axes
    ax.add_artist(color_legend)

    handles = [lines[('k', ls, m)] for (ls, m) in zip(linestyles, markers)]
    texts = ['$\sim_G$-PID', '$\delta_G$-PID', 'MMI-PID', 'Ground truth']
    defn_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                            bbox_to_anchor=(1, 0.2), fontsize=legendsize,
                            title='PID definition', title_fontsize=labelsize)

    plt.tight_layout()
    plt.savefig('../figures/gain-angle-sweeps.pdf')
    #plt.show()


    #cols = ['desc', 'id', 'tilde', 'delta', 'mmi', 'gt']
    #melted_df = pid_table[cols].melt(id_vars=['desc', 'id'],
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

    #facets = sns.relplot(kind='line', data=melted_df, col='desc', x='id',
    #                     y='Partial information (bits)',
    #                     hue='Partial info component', style='PID Definition',
    #                     dashes=[(1, 0), (3, 2), (1, 1), (1, 0)],
    #                     markers=['', '', '', 'o'],
    #                     facet_kws=dict(sharex=False))

    #titlesize = 18
    #labelsize = 16
    #legendsize = 14
    #ticksize = 12

    ## Gain
    #ax = facets.axes[0, 0]
    #ax.set_title(r'Increasing Gain in $X_1$', fontsize=titlesize)
    #ax.set_xlabel(r'Gain in $X_1$', fontsize=labelsize)
    #ax.set_ylabel(ax.get_ylabel(), fontsize=labelsize)
    #rows = pid_table[pid_table.desc == 'gain']
    #ax.set_xticks(rows['id'])
    #ax.set_xticklabels(['%.2f' % val for val in rows.gain_x],
    #                   rotation=45, ha='right')
    #ax.tick_params(axis='both', which='major', labelsize=ticksize)

    ## Angle
    #ax = facets.axes[0, 1]
    #ax.set_title(r'Rotation of $X$ w.r.t. $Y$', fontsize=titlesize)
    #ax.set_xlabel(r'Angle, $\theta$', fontsize=labelsize)
    #rows = pid_table[pid_table.desc == 'angle']
    #ax.set_xticks(rows['id'])
    #ax.set_xticklabels(['%.2f' % val for val in rows.theta],
    #                   rotation=45, ha='right')
    #ax.tick_params(axis='both', which='major', labelsize=ticksize)

    #plt.setp(facets.legend.get_texts(), fontsize=legendsize)

    #plt.gcf().subplots_adjust(bottom=0.18, right=0.78)
    #plt.show()
