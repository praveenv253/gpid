#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


def process(g, normed=False):
    g.set_xlabels('')
    g.set_ylabels('')

    xticklabels = ['$I_{MXY}$', '$UI_X$', '$UI_Y$', '$RI$', '$SI$']
    if normed:
        xticklabels = xticklabels[1:]
    g.set_xticklabels(xticklabels)


if __name__ == '__main__':
    #structures = ('VISp', 'VISl', 'VISal')
    structures = ('VISp', 'VISl', 'VISam')
    #top_pcs = 10
    top_pcs = 20

    filename = ('../results/vbn-pids-time--' + '-'.join(s.lower() for s in structures)
                + '--%d.csv' % top_pcs)

    pid_df = pd.read_csv(filename)
    pid_df = (pid_df.query('experience_level == "Familiar"')
              .drop(columns=['experience_level']))

    pid_df = pid_df.replace({'change': 'Change', 'non_change': 'Non-change'})
    pid_df = pid_df.rename(columns={'cond': 'Condition'})

    pid_df_normed = pid_df.copy()
    cols = ['uix', 'uiy', 'ri', 'si']
    pid_df_normed[cols] = pid_df_normed[cols].div(pid_df_normed[['imxy']].values)

    sns.set_context('talk')

    data = pd.melt(pid_df, id_vars=['mouse_id', 'time', 'Condition'],
                   value_vars=['imxy', 'uix', 'uiy', 'ri', 'si'],
                   var_name='pid_comp', value_name='pid_val')

    g = sns.catplot(kind='strip', data=data, col='time', x='pid_comp', y='pid_val',
                    hue='Condition', dodge=True, height=5, aspect=0.7)
    process(g)
    plt.suptitle(r'PID values in bits; '
                 r'$M = \mathrm{%s}$, $X = \mathrm{%s}$, $Y = \mathrm{%s}$'
                 r' (top %d PCs)' % (structures + (top_pcs,)))
    fig = plt.gcf()
    fig.supylabel('PID value (bits)')
    fig.supxlabel('PID components over time (in ms)')
    g.tight_layout()

    data_normed = pd.melt(pid_df_normed, id_vars=['mouse_id', 'time', 'Condition'],
                          value_vars=['uix', 'uiy', 'ri', 'si'],
                          var_name='pid_comp', value_name='pid_val')

    g = sns.catplot(kind='strip', data=data_normed, col='time',  x='pid_comp',
                    y='pid_val', hue='Condition', dodge=True, height=5, aspect=0.7)
    process(g, normed=True)
    plt.suptitle(r'PID fraction of total mutual information; '
                 r'$M = \mathrm{%s}$, $X = \mathrm{%s}$, $Y = \mathrm{%s}$'
                 r' (top %d PCs)' % (structures + (top_pcs,)))
    fig = plt.gcf()
    fig.supylabel('PID fraction')
    fig.supxlabel('PID components over time (in ms)')
    g.tight_layout()

    plt.show()
