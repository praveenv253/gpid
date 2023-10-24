#!/usr/bin/env python3

from __future__ import print_function, division

import joblib
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = joblib.load('../results/doubling_comparison.pkl')

    # pid_values has shape (num_gains, num_doubles, 5)
    pid_table = data['pid_table']
    gains = data['gains']
    num_doubles = data['num_doubles']

    tilde_pid_values = pid_table['tilde'].values.reshape((gains.size, -1, 5))
    tilde_pid_values = np.ma.masked_array(tilde_pid_values, mask=(tilde_pid_values < 1e-3))

    delta_pid_values = pid_table['delta'].values.reshape((gains.size, -1, 5))
    delta_pid_values = np.ma.masked_array(delta_pid_values, mask=(delta_pid_values < 1e-3))

    # Don't plot values close to zero
    expansion_factor = 2**(np.arange(num_doubles) + 1)

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
            ax.loglog(expansion_factor, tilde_pid_values[j, :, i], color=colors[j])
            ax.loglog(expansion_factor, delta_pid_values[j, :, i], color=colors[j], linestyle='--')

        ax.grid(True)
        ax.set_title(titles[i], fontsize=titlesize)
        ax.set_xticks(expansion_factor)
        ax.set_xticklabels(expansion_factor, rotation=45, ha='right',
                           rotation_mode='anchor')
        ax.set_yticks(2.0**np.arange(-2, 10))
        ax.set_yticklabels(['%g' % i for i in 2.0**np.arange(-2, 10)])
        ax.tick_params(axis='both', labelsize=ticksize)
        ax.set_aspect('equal')
        ax.minorticks_off()

    fig.suptitle('Stability of the $\sim_G$-PID over increasing dimensionality',
                 fontsize=suptitlesize)
    fig.supxlabel('Dimension of each of $M$, $X$ and $Y$', fontsize=labelsize)
    fig.supylabel('Partial Information (bits)', fontsize=labelsize)

    plt.tight_layout()
    #plt.savefig('../figures/doubling-example.pdf')
    plt.show()
