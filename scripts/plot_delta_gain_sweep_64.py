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
    delta_pid_values = pid_table['delta'].values.reshape((gains.size, -1, 5))
    #delta_pid_values = np.ma.masked_array(delta_pid_values, mask=(delta_pid_values < 1e-3))

    colors = ['k', 'C0', 'C1', 'C2', 'C3']

    #plt.figure(figsize=(4, 4))
    plt.figure()

    for i in range(5):
        if i == 0:
            l1, = plt.plot(gains, tilde_pid_values[:, -1, i], color=colors[i])
            l2, = plt.plot(gains, delta_pid_values[:, -1, i], linestyle='--', color=colors[i])
        plt.plot(gains, tilde_pid_values[:, -1, i], color=colors[i])
        plt.plot(gains, delta_pid_values[:, -1, i], linestyle='--', color=colors[i])

    plt.grid()

    ax = plt.gca()
    ax.set_xticks(gains)
    ax.set_xticklabels(['%.2f' % val for val in gains],
                       rotation=45, rotation_mode='anchor', ha='right')
    ax.tick_params(axis='both', which='major', labelsize=14)

    titlesize = 18
    labelsize = 18
    legendsize = 16
    ax.set_title('$\sim$- vs. $\delta$-PID at $d_M = d_X = d_Y = 64$', fontsize=titlesize)
    ax.set_xlabel(r'Gain in $X_1$, $\alpha$', fontsize=labelsize)
    ax.set_ylabel('Partial information (bits)', fontsize=labelsize)

    plt.legend((l1, l2), ('$\sim_G$-PID', '$\delta_G$-PID'), fontsize=legendsize)

    plt.tight_layout()
    plt.savefig('../figures/gain-sweep-64.pdf')
    #plt.show()
