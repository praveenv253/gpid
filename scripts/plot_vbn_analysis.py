#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


if __name__ == '__main__':
    structures = ('VISp', 'VISl', 'VISal')
    #structures = ('VISp', 'VISl', 'VISam')
    #top_pcs = 10
    top_pcs = 20

    filename = ('../results/vbn-pids-time--' + '-'.join(s.lower() for s in structures)
                + '--%d.csv' % top_pcs)

    pid_df = pd.read_csv(filename)
    pid_df_normed = pid_df.copy()
    cols = ['uix', 'uiy', 'ri', 'si']
    pid_df_normed[cols] = pid_df_normed[cols].div(pid_df_normed[['imxy']].values)

    data = pd.melt(pid_df, id_vars=['mouse_id', 'experience_level', 'time', 'cond'],
                   value_vars=['imxy', 'uix', 'uiy', 'ri', 'si'],
                   var_name='pid_comp', value_name='pid_val')
    data['exp_cond'] = data['experience_level'] + '_' + data['cond']

    #plt.figure()
    sns.catplot(kind='strip', data=data, col='time', x='pid_comp', y='pid_val',
                hue='exp_cond', dodge=True)#, legend=False, alpha=0.5)
    #sns.boxplot(data=data, x='pid_comp', y='pid_val', hue='exp_cond', boxprops=dict(alpha=0.5))
    plt.suptitle('PID values in bits (VISp -> VISl, VISal)')
    #plt.tight_layout()

    data_normed = pd.melt(pid_df_normed, id_vars=['mouse_id', 'experience_level', 'time', 'cond'],
                          value_vars=['uix', 'uiy', 'ri', 'si'],
                          var_name='pid_comp', value_name='pid_val')
    data_normed['exp_cond'] = data_normed['experience_level'] + '_' + data_normed['cond']

    #plt.figure()
    sns.catplot(kind='strip', data=data_normed, col='time',  x='pid_comp',
                y='pid_val', hue='exp_cond', dodge=True)#, legend=False, alpha=0.5)
    #sns.boxplot(data=data_normed, x='pid_comp', y='pid_val', hue='exp_cond', boxprops=dict(alpha=0.5))
    plt.suptitle('PID fraction of total mutual info (VISp -> VISl, VISal)')
    #plt.tight_layout()

    #df = pid_df_normed.reset_index()
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
