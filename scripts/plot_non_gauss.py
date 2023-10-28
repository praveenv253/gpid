#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


if __name__ == '__main__':
    pid_table = pd.read_pickle('../results/non_gauss_examples.pkl.gz')

    #config_cols = ['desc', 'id', 'dm', 'dx', 'dy', 'w_x1']
    #pid_table = pd.merge(gt_pid_table, other_pid_table, on=config_cols)
    #pid_table = other_pid_table

    pid_defns = ['tilde', 'gt']
    linestyles = ['-', '']
    markers = ['', 'o']

    pid_atoms = ['imxy', 'uix', 'uiy', 'ri', 'si']
    colors = ['k', 'C0', 'C1', 'C2', 'C3']

    system_descs = {'binom': 'multivariate Binomial',
                    'poiss': 'multivariate Poisson',
                    'binom_zi': 'zero-inflated Binomial',
                    'poiss_zi': 'zero-inflated Poisson',}

    titlesize = 18
    labelsize = 16
    legendsize = 14
    ticksize = 12

    for desc, system_desc in system_descs.items():
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
        axs = axs.flatten()

        # Un-normalized PID values

        ax = axs[0]
        lines = {}  # Dictionary to hold all line handles for legend
        rows = pid_table[pid_table.desc == desc]
        for i, pid_defn in enumerate(pid_defns):
            for j, pid_atom in enumerate(pid_atoms):
                line = ax.plot(rows['id'], rows[(pid_defn, pid_atom)],
                               color=colors[j], linestyle=linestyles[i],
                               marker=markers[i])[0]
                lines[(colors[j], linestyles[i], markers[i])] = line

        fig.suptitle(r'PID values (left) and PID normalized by $I(M\;\!; (X, Y\;\!\;\!))$ (right)'
                     + '\nin a %s spike-count simulation' % system_desc, fontsize=titlesize)
        #ax.set_title(r'PIDs for multivariate Poisson example', fontsize=titlesize)
        ax.set_xlabel(r'Weight from $M_1$ to $X$', fontsize=labelsize)
        ax.set_ylabel('Partial information (bits)', fontsize=labelsize)
        ax.set_xticks(rows['id'])
        ax.set_xticklabels(['%.g' % val for val in rows.w_x1])#, rotation=45, ha='right')
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.grid(True)

        # Normalized PID values

        ax = axs[1]
        lines = {}  # Dictionary to hold all line handles for legend
        rows = pid_table.copy()
        rows = rows[rows.desc == desc]

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
        #ax.set_title(r'PIDs for multivariate Poisson example', fontsize=titlesize)
        ax.set_xlabel(r'Weight from $M_1$ to $X$', fontsize=labelsize)
        ax.set_ylabel('Normalized partial information', fontsize=labelsize)
        ax.set_xticks(rows['id'])
        ax.set_xticklabels(['%.g' % val for val in rows.w_x1])#, rotation=45, ha='right')
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.grid(True)

        # Legend
        handles = [lines[(c, '-', '')] for c in colors]
        texts = ['$I(M\;\!;(X, Y\;\!\;\!))$', r'$UI_X$', r'$UI_Y$', r'$RI$', r'$SI$']
        color_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                                 bbox_to_anchor=(1, 0.75), fontsize=legendsize,
                                 title='PID component', title_fontsize=labelsize)
        # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#multiple-legends-on-the-same-axes
        ax.add_artist(color_legend)

        handles = [lines[('k', ls, m)] for (ls, m) in zip(linestyles, markers)]
        texts = ['$\sim_G$-PID', 'Ground truth\n(Banerjee et al.\n$\sim$-PID)']
        defn_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                                bbox_to_anchor=(1, 0.15), fontsize=legendsize,
                                title='PID definition', title_fontsize=labelsize)

        #plt.tight_layout()
        plt.subplots_adjust(left=0.07, right=0.8, bottom=0.12, top=0.85, wspace=0.25)
        plt.savefig('../figures/non-gauss--%s.pdf' % desc.replace('_', '-'))
        plt.close()

    #plt.show()
