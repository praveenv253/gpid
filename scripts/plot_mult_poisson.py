#!/usr/bin/env python3

from __future__ import print_function, division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


if __name__ == '__main__':
    gt_pid_table = pd.read_pickle('../results/mult_poisson_gt.pkl.gz')
    other_pid_table = pd.read_pickle('../results/mult_poisson_example.pkl.gz')

    config_cols = ['desc', 'id', 'dm', 'dx', 'dy', 'w_x1']
    pid_table = pd.merge(gt_pid_table, other_pid_table, on=config_cols)
    #pid_table = other_pid_table

    pid_defns = ['tilde', 'delta', 'mmi', 'gt']#[:3]
    linestyles = ['-', '--', ':', '']#[:3]
    markers = ['', '', '', 'o']#[:3]

    pid_atoms = ['imxy', 'uix', 'uiy', 'ri', 'si']
    colors = ['k', 'C0', 'C1', 'C2', 'C3']

    titlesize = 18
    labelsize = 16
    legendsize = 14
    ticksize = 12


    # Un-normalized PID values
    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()
    lines = {}  # Dictionary to hold all line handles for legend
    rows = pid_table[pid_table.desc == 'mult_poisson']
    for i, pid_defn in enumerate(pid_defns):
        for j, pid_atom in enumerate(pid_atoms):
            line = ax.plot(rows['id'], rows[(pid_defn, pid_atom)],
                           color=colors[j], linestyle=linestyles[i],
                           marker=markers[i])[0]
            lines[(colors[j], linestyles[i], markers[i])] = line

    ax.set_title(r'PIDs for multivariate Poisson example', fontsize=titlesize)
    ax.set_xlabel(r'Weight from $M_1$ to $X$', fontsize=labelsize)
    ax.set_ylabel('Partial information (bits)', fontsize=labelsize)
    ax.set_xticks(rows['id'])
    ax.set_xticklabels(['%.g' % val for val in rows.w_x1])#, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)

    # Legend
    handles = [lines[(c, '-', '')] for c in colors]
    texts = ['$I(M;(X,Y))$', '$UI_X$', '$UI_Y$', '$RI$', '$SI$']
    color_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                             bbox_to_anchor=(1, 0.7), fontsize=legendsize,
                             title='PID component', title_fontsize=labelsize)
    # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#multiple-legends-on-the-same-axes
    ax.add_artist(color_legend)

    handles = [lines[('k', ls, m)] for (ls, m) in zip(linestyles, markers)]
    texts = ['$\sim$-PID', '$\delta$-PID', 'MMI-PID', 'Ground truth']#[:3]
    defn_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                            bbox_to_anchor=(1, 0.2), fontsize=legendsize,
                            title='PID definition', title_fontsize=labelsize)

    plt.tight_layout()
    plt.savefig('../figures/mult-poisson-ex.pdf')
    #plt.show()


    # Normalized PID values
    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()
    lines = {}  # Dictionary to hold all line handles for legend
    rows = pid_table.copy()

    # Normalize
    for pid_defn in pid_defns:
        rows[pid_defn] = (rows[pid_defn].to_numpy()
                          / rows[(pid_defn, 'imxy')].to_numpy()[:, np.newaxis])

    for i, pid_defn in enumerate(pid_defns):
        for j, pid_atom in enumerate(pid_atoms):
            line = ax.plot(rows['id'], rows[(pid_defn, pid_atom)],
                           color=colors[j], linestyle=linestyles[i],
                           marker=markers[i])[0]
            lines[(colors[j], linestyles[i], markers[i])] = line
            if j == 0:
                line.remove()

    ax.set_ylim(-0.03, 0.63)
    ax.set_title(r'PIDs for multivariate Poisson example', fontsize=titlesize)
    ax.set_xlabel(r'Weight from $M_1$ to $X$', fontsize=labelsize)
    ax.set_ylabel('Normalized partial information', fontsize=labelsize)
    ax.set_xticks(rows['id'])
    ax.set_xticklabels(['%.g' % val for val in rows.w_x1])#, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)

    # Legend
    handles = [lines[(c, '-', '')] for c in colors[1:]]
    texts = ['$I(M;(X,Y))$', '$UI_X$', '$UI_Y$', '$RI$', '$SI$'][1:]
    color_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                             bbox_to_anchor=(1, 0.7), fontsize=legendsize,
                             title='PID component', title_fontsize=labelsize)
    # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#multiple-legends-on-the-same-axes
    ax.add_artist(color_legend)

    handles = [lines[('k', ls, m)] for (ls, m) in zip(linestyles, markers)]
    texts = ['$\sim$-PID', '$\delta$-PID', 'MMI-PID', 'Ground truth']#[:3]
    defn_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                            bbox_to_anchor=(1, 0.2), fontsize=legendsize,
                            title='PID definition', title_fontsize=labelsize)

    plt.tight_layout()
    plt.savefig('../figures/mult-poisson-ex-norm.pdf')
    #plt.show()
