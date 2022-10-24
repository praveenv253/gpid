#!/usr/bin/env python3

from __future__ import print_function, division

import joblib
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = joblib.load('../results/doubling_example.pkl')

    # pid_values has shape (num_gains, num_doubles, 5)
    pid_values = data['pid_vals']
    pid_values = np.ma.masked_array(pid_values, mask=(pid_values < 1e-3))

    #gains = np.linspace(0, 3, 10)
    #num_doubles = 7
    gains = data['gains']
    num_doubles = data['num_doubles']

    expansion_factor = 2**(np.arange(num_doubles) + 1)

    cmaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples']
    titles = ['$I(M ; (X, Y))$', '$UI_X$', '$UI_Y$', '$RI$', '$SI$']
    fnames = ['imxy', 'uix', 'uiy', 'ri', 'si']
    suptitlesize = 18
    labelsize = 16
    titlesize = 16
    ticksize = 12
    fig, axs = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True,
                            figsize=(10, 4))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        colors = plt.get_cmap(cmaps[i])(np.linspace(0.1, 1, gains.size))
        for j in range(gains.size):
            ax.loglog(expansion_factor, pid_values[j, :, i], color=colors[j])
        ax.grid(True)
        ax.set_title(titles[i], fontsize=titlesize)
        ax.set_xticks(expansion_factor)
        ax.set_xticklabels(expansion_factor, rotation=90, ha='center')
        ax.set_yticks(2.0**np.arange(-2, 8))
        ax.set_yticklabels(['%g' % i for i in 2.0**np.arange(-2, 8)])
        ax.tick_params(axis='both', labelsize=ticksize)
        ax.set_aspect('equal')
        ax.minorticks_off()

    fig.supxlabel('Dim of $M$, $X$ or $Y$', fontsize=labelsize)
    fig.supylabel('Partial Info (bits)', fontsize=labelsize)

    plt.tight_layout()
    plt.savefig('../figures/doubling-example.pdf')
    #plt.show()
