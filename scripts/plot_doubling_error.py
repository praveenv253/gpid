#!/usr/bin/env python3

from __future__ import print_function, division

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(pid_values, expansion_factor, gains, suptitle, ylabel):
    cmaps = ['Greys', 'Blues', 'Oranges', 'Greens', 'Reds']
    titles = [r'$I(M\;\!; (X, Y\;\!\;\!))$', '$UI_X$', '$UI_Y$', '$RI$', '$SI$']
    fnames = ['imxy', 'uix', 'uiy', 'ri', 'si']
    suptitlesize = 18
    labelsize = 16
    titlesize = 16
    ticksize = 12
    fig, axs = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True,
                            figsize=(11, 4.5))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        colors = plt.get_cmap(cmaps[i])(np.linspace(0.1, 1, gains.size))
        for j in range(gains.size):
            ax.loglog(expansion_factor, pid_values[j, :, i], color=colors[j])
        ax.grid(True)
        ax.set_title(titles[i], fontsize=titlesize)
        ax.set_xticks(expansion_factor)
        ax.set_xticklabels(expansion_factor, rotation=90)
        #ax.set_yticks(2.0**np.arange(-2, 8))
        #ax.set_yticklabels(['%g' % i for i in 2.0**np.arange(-2, 8)])
        ax.tick_params(axis='both', labelsize=ticksize)
        #ax.set_aspect('equal')
        ax.minorticks_off()

    fig.suptitle(suptitle, fontsize=suptitlesize)
    fig.supxlabel('Dimension of each of $M$, $X$ and $Y$', fontsize=labelsize)
    fig.supylabel(ylabel, fontsize=labelsize)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    data = joblib.load('../results/doubling_example.pkl')

    # pid_values has shape (num_gains, num_doubles, 5)
    pid_values = data['pid_vals']
    #pid_values = np.ma.masked_array(pid_values, mask=(pid_values < 1e-3))

    gains = data['gains']
    num_doubles = data['num_doubles']

    pid_values = np.array(pid_values).reshape((-1, num_doubles, 5))

    # Get ground truth data
    pid_table = pd.read_pickle('../results/gain_angle_exs_099.pkl.gz')
    gt = pid_table[pid_table['desc'] == 'gain']['gt'].values  # shape (10, 5)

    expansion_factor = 2**(np.arange(num_doubles) + 1)  # Dim of M, X, Y
    gt_doubles = (gt[:, np.newaxis, :]
                  * expansion_factor[np.newaxis, :, np.newaxis] / 2)

    # Absolute error
    abs_error = np.abs(pid_values - gt_doubles)
    fig_abs = plot(abs_error, expansion_factor, gains,
                   suptitle='Absolute error in the $\sim_G$-PID over increasing dimensionality',
                   ylabel='Abs. error in PID value (bits)')
    plt.savefig('../figures/doubling-abs-error.pdf')
    plt.close()

    # Relative error
    rel_error = abs_error / gt_doubles
    fig_rel = plot(rel_error, expansion_factor, gains,
                   suptitle='Relative error in the $\sim_G$-PID over increasing dimensionality',
                   ylabel='Rel. error in PID value')
    plt.savefig('../figures/doubling-rel-error.pdf')
    plt.close()

    abs_error_df = pd.DataFrame(abs_error.max(axis=2),
                                index=pd.Index(gains, name='gain'),
                                columns=expansion_factor).reset_index()
    rel_error_df = pd.DataFrame(rel_error.max(axis=2),
                                index=pd.Index(gains, name='gain'),
                                columns=expansion_factor).reset_index()

    pid_names = ['I(M; (X,Y))', 'UI_X', 'UI_Y', 'RI', 'SI']
    pid_df = pd.DataFrame(pid_values[-1, :, :], index=expansion_factor, columns=pid_names)
    df = pd.concat({'Absolute error (bits)': abs_error_df.iloc[9, 1:],
                    'Relative error': rel_error_df.iloc[9, 1:],
                    'Compute time (s)': pd.Series(data['time_taken'][-1], index=expansion_factor)}, axis=1)
    pid_df = pid_df.join(df)
    pid_df.index.rename('Dimension', inplace=True)

    pd.set_option('display.precision', 2)
    print(pid_df)

    #plt.show()
