#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from statannotations_new.Annotator import Annotator

from scipy import stats


if __name__ == '__main__':
    structures = ('VISp', 'VISl', 'VISal')
    #structures = ('VISp', 'VISl', 'VISam')
    #top_pcs = 10
    top_pcs = 20
    #top_pcs = 'max'
    eq_samp = False
    #eq_samp = True

    filename = ('../results/vbn-pids-time--'
                + ('eq-samp--' if eq_samp else '')
                + '-'.join(s.lower() for s in structures)
                + ('--%d' % top_pcs if type(top_pcs) == int else '--%s' % top_pcs)
                + '.csv')

    pid_df = pd.read_csv(filename)

    if top_pcs != 'max':
        pid_df = pid_df.set_index(['mouse_id', 'experience_level', 'time', 'cond']).unstack(['experience_level', 'time', 'cond']).dropna()
        pid_df = pid_df.stack(['experience_level', 'time', 'cond']).reset_index()

    pid_df_normed = pid_df.copy()
    cols = ['uix', 'uiy', 'ri', 'si']
    pid_df_normed[cols] = pid_df_normed[cols].div(pid_df_normed[['imxy']].values)

    data = pd.melt(pid_df, id_vars=['mouse_id', 'experience_level', 'time', 'cond'],
                   value_vars=['imxy', 'uix', 'uiy', 'ri', 'si'],
                   var_name='pid_comp', value_name='pid_val')
    data['exp_cond'] = data['experience_level'] + '_' + data['cond']

    if top_pcs == 'max':
        # XXX: Quick-fix - remove really large outliers due to poor convergence
        data.loc[data['pid_val'] > 50, 'pid_val'] = np.nan

    subdata = data.query('pid_comp == "ri" and experience_level == "Familiar"')

    sns.set_context('talk')

    fig_unnorm = plt.figure(figsize=(7, 7))
    plot_params = dict(data=subdata, x='time', y='pid_val', hue='cond')
    g = sns.stripplot(**plot_params, dodge=True)
    sns.boxplot(**plot_params, boxprops=dict(alpha=0.5))

    pairs = []
    #pairs = [[(t1, 'change'), (t2, 'change')]
             #for t1, t2 in zip([0, 50, 100, 150], [50, 100, 150, 200])]
    #pairs.extend([[(t1, 'non_change'), (t2, 'non_change')]
                  #for t1, t2 in zip([0, 50, 100, 150], [50, 100, 150, 200])])
    #pairs = sum(([i, j] for i, j in zip(pairs[:4], pairs[4:])), [])  # Reorder
    pairs.extend([[(t, 'change'), (t, 'non_change')] for t in [0, 50, 100, 150, 200]])

    annotator = Annotator(g, pairs, **plot_params)
    annotator.configure(test='Mann-Whitney')
    annotator.apply_and_annotate()

    plt.title('Redundancy betw 3 visual cortical areas\n'
              + ('(top %d principal components)' % top_pcs
                 if type(top_pcs) == int else '(%s principal components)' % top_pcs))
    plt.xlabel('Time after stimulus onset (ms)')
    plt.ylabel('Redundancy, $RI(\mathrm{%s} : \mathrm{%s} ; \mathrm{%s})$ (bits)'
               % structures)

    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h[2:], ['Change', 'Non-change'])

    plt.tight_layout()


    ## Normalized plots

    data_normed = pd.melt(pid_df_normed, id_vars=['mouse_id', 'experience_level', 'time', 'cond'],
                          value_vars=['uix', 'uiy', 'ri', 'si'],
                          var_name='pid_comp', value_name='pid_val')
    data_normed['exp_cond'] = data_normed['experience_level'] + '_' + data_normed['cond']
    subdata_normed = data_normed.query('pid_comp == "ri" and experience_level == "Familiar"')

    fig_norm = plt.figure(figsize=(7, 7))
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

    plt.title('Redundancy fraction betw 3 visual cortical areas\n'
              + ('(top %d principal components)' % top_pcs
                 if type(top_pcs) == int else '(%s principal components)' % top_pcs))
    plt.xlabel('Time after stimulus onset (ms)')
    plt.ylabel(r'$RI(\mathrm{%s} : \mathrm{%s} ; \mathrm{%s})' % structures
               + r'/ I(\mathrm{%s} ; (\mathrm{%s} , \mathrm{%s}))$' % structures)

    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h[2:], ['Change', 'Non-change'])

    plt.tight_layout()

    print()
    print('Number of mice: ', pid_df['mouse_id'].nunique())
    print(pid_df
          .set_index(['mouse_id', 'experience_level', 'time', 'cond'])
          .unstack(['experience_level', 'time', 'cond']).dropna())

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

    #plt.figure(fig_unnorm)
    #plt.savefig(filename.replace('results', 'figures')
    #            .replace('vbn-pids-time', 'vbn-ri')
    #            .replace('csv', 'pdf'))
    #plt.figure(fig_norm)
    #plt.savefig(filename.replace('results', 'figures')
    #            .replace('vbn-pids-time', 'vbn-ri-norm')
    #            .replace('csv', 'pdf'))
    plt.show()
