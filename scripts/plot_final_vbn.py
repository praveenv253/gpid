#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from statannotations_new.Annotator import Annotator

from scipy import stats


if __name__ == '__main__':
    #pid_df = pd.read_csv('vbn-pids-time--visp-visl-visal.csv')
    pid_df = pd.read_csv('vbn-pids-time--visp-visl-visam.csv')
    pid_df_normed = pid_df.copy()
    cols = ['uix', 'uiy', 'ri', 'si']
    pid_df_normed[cols] = pid_df_normed[cols].div(pid_df_normed[['imxy']].values)

    data = pd.melt(pid_df, id_vars=['mouse_id', 'experience_level', 'time', 'cond'],
                   value_vars=['imxy', 'uix', 'uiy', 'ri', 'si'],
                   var_name='pid_comp', value_name='pid_val')
    data['exp_cond'] = data['experience_level'] + '_' + data['cond']
    subdata = data.query('pid_comp == "ri" and experience_level == "Familiar"')

    sns.set_context('talk')

    plt.figure(figsize=(7, 7))
    plot_params = dict(data=subdata, x='time', y='pid_val', hue='cond')
    g = sns.stripplot(**plot_params, dodge=True)
    sns.boxplot(**plot_params, boxprops=dict(alpha=0.5))

    pairs = [[(t1, 'change'), (t2, 'change')]
             for t1, t2 in zip([0, 50, 100, 150], [50, 100, 150, 200])]
    pairs.extend([[(t1, 'non_change'), (t2, 'non_change')]
                  for t1, t2 in zip([0, 50, 100, 150], [50, 100, 150, 200])])
    pairs = sum(([i, j] for i, j in zip(pairs[:4], pairs[4:])), [])  # Reorder
    pairs.extend([[(t, 'change'), (t, 'non_change')] for t in [0, 50, 100, 150, 200]])

    annotator = Annotator(g, pairs, **plot_params)
    annotator.configure(test='Mann-Whitney')
    annotator.apply_and_annotate()

    plt.title('Redundancy betw 3 visual cortical areas')
    plt.xlabel('Time after stimulus onset (ms)')
    plt.ylabel('Redundancy, $RI(\mathrm{VISp} : \mathrm{VISl} ; \mathrm{VISam})$ (bits)')

    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h[2:], ['Change', 'Non-change'])

    plt.tight_layout()



    data_normed = pd.melt(pid_df_normed, id_vars=['mouse_id', 'experience_level', 'time', 'cond'],
                          value_vars=['uix', 'uiy', 'ri', 'si'],
                          var_name='pid_comp', value_name='pid_val')
    data_normed['exp_cond'] = data_normed['experience_level'] + '_' + data_normed['cond']
    subdata_normed = data_normed.query('pid_comp == "ri" and experience_level == "Familiar"')

    plt.figure(figsize=(7, 7))
    plot_params = dict(data=subdata_normed, x='time', y='pid_val', hue='cond')
    g = sns.stripplot(**plot_params, dodge=True)
    sns.boxplot(**plot_params, boxprops=dict(alpha=0.5))

    pairs = [[(t1, 'change'), (t2, 'change')]
             for t1, t2 in zip([0, 50, 100, 150], [50, 100, 150, 200])]
    pairs.extend([[(t1, 'non_change'), (t2, 'non_change')]
                  for t1, t2 in zip([0, 50, 100, 150], [50, 100, 150, 200])])
    pairs = sum(([i, j] for i, j in zip(pairs[:4], pairs[4:])), [])  # Reorder
    pairs.extend([[(t, 'change'), (t, 'non_change')] for t in [0, 50, 100, 150, 200]])

    annotator = Annotator(g, pairs, **plot_params)
    annotator.configure(test='Mann-Whitney')
    annotator.apply_and_annotate()

    plt.title('Redundancy fraction betw 3 visual cortical areas')
    plt.xlabel('Time after stimulus onset (ms)')
    plt.ylabel(r'$RI(\mathrm{VISp} : \mathrm{VISl} ; \mathrm{VISam})'
               r'/ I(\mathrm{VISp} ; \mathrm{VISl} , \mathrm{VISam})$')

    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h[2:], ['Change', 'Non-change'])

    plt.tight_layout()



    #x = data_normed.query('exp_cond == "Familiar_change" and time == 50 and pid_comp == "ri"')['pid_val']
    #y = data_normed.query('exp_cond == "Familiar_non_change" and time == 50 and pid_comp == "ri"')['pid_val']
    #_, pval = stats.ranksums(x, y)
    #print('RI Familiar change vs no-change t = 50: %f' % pval)

    #x = data_normed.query('exp_cond == "Familiar_change" and time == 100 and pid_comp == "ri"')['pid_val']
    #y = data_normed.query('exp_cond == "Familiar_non_change" and time == 100 and pid_comp == "ri"')['pid_val']
    #_, pval = stats.ranksums(x, y)
    #print('RI Familiar change vs no-change t = 100: %f' % pval)

    #x = data_normed.query('exp_cond == "Novel_change" and time == 50 and pid_comp == "ri"')['pid_val']
    #y = data_normed.query('exp_cond == "Novel_non_change" and time == 50 and pid_comp == "ri"')['pid_val']
    #_, pval = stats.ranksums(x, y)
    #print('RI Novel change vs no-change t = 50: %f' % pval)

    #x = data_normed.query('exp_cond == "Novel_change" and time == 100 and pid_comp == "ri"')['pid_val']
    #y = data_normed.query('exp_cond == "Novel_non_change" and time == 100 and pid_comp == "ri"')['pid_val']
    #_, pval = stats.ranksums(x, y)
    #print('RI Novel change vs no-change t = 100: %f' % pval)

    plt.show()
