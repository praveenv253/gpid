#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    pid_df = pd.read_csv('vbn-pids.csv')
    pid_df_normed = pid_df.copy()
    cols = ['uix', 'uiy', 'ri', 'si']
    pid_df_normed[cols] = pid_df_normed[cols].div(pid_df_normed[['imxy']].values)

    data = pd.melt(pid_df, id_vars=['mouse_id', 'experience_level', 'cond'],
                   value_vars=['imxy', 'uix', 'uiy', 'ri', 'si'],
                   var_name='pid_comp', value_name='pid_val')
    data['exp_cond'] = data['experience_level'] + '_' + data['cond']

    plt.figure()
    sns.stripplot(data=data, x='pid_comp', y='pid_val', hue='exp_cond', alpha=0.5, dodge=True, legend=False)
    sns.boxplot(data=data, x='pid_comp', y='pid_val', hue='exp_cond', boxprops=dict(alpha=0.5))

    data_normed = pd.melt(pid_df_normed, id_vars=['mouse_id', 'experience_level', 'cond'],
                          value_vars=['uix', 'uiy', 'ri', 'si'],
                          var_name='pid_comp', value_name='pid_val')
    data_normed['exp_cond'] = data_normed['experience_level'] + '_' + data_normed['cond']

    plt.figure()
    sns.stripplot(data=data_normed, x='pid_comp', y='pid_val', hue='exp_cond', alpha=0.5, dodge=True, legend=False)
    sns.boxplot(data=data_normed, x='pid_comp', y='pid_val', hue='exp_cond', boxprops=dict(alpha=0.5))

    plt.show()
