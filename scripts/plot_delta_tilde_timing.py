#!/usr/bin/env python3

from __future__ import print_function, division

import joblib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ## Comparison between delta and tilde up to d=64

    data = joblib.load('../results/doubling_comparison.pkl')

    gains = data['gains']
    num_doubles = data['num_doubles']
    tt = np.array(data['time_taken']).reshape((gains.size, num_doubles, 2))
    mean_tt = tt.mean(axis=0)

    labelsize = 16
    titlesize = 18
    legendsize = 16
    ticksize = 12
    linestyles = ['-', '--']

    plt.figure(figsize=(2.5, 4))
    x = np.arange(num_doubles)
    for i in range(2):
        plt.semilogy(x, mean_tt[:, i], color='k', ls=linestyles[i])

    #plt.title('Timing analysis of $\sim_G$- and $\delta_G$\n'
    #          'PIDs for the doubling example', fontsize=titlesize)
    plt.title('Timing analysis         ', fontsize=titlesize, pad=10)
    plt.xlabel('$d$', fontsize=titlesize, labelpad=8)
    plt.ylabel('Time taken (seconds)', fontsize=labelsize)

    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(['%d' % i for i in 2**(x+1)], rotation=90)#, ha='right',
                       #rotation_mode='anchor')
    ax.tick_params(axis='both', labelsize=ticksize)

    plt.legend(('$\sim_G$-PID', '$\delta_G$-PID'), fontsize=legendsize,
               handlelength=1, handletextpad=0.3, frameon=False)
    plt.grid(alpha=0.5)
    plt.tight_layout()

    plt.savefig('../figures/timing-comparison.pdf')
    plt.close()


    ## Performance of tilde up to d=1024

    data = joblib.load('../results/doubling_example.pkl')

    gains = data['gains']
    num_doubles = data['num_doubles']
    tt = np.array(data['time_taken']).reshape((gains.size, num_doubles))
    mean_tt = tt.mean(axis=0)

    labelsize = 16
    titlesize = 18
    legendsize = 16
    ticksize = 12

    plt.figure()
    x = np.arange(num_doubles)
    plt.semilogy(x, mean_tt, color='k')

    plt.title('Timing analysis of the $\sim_G$-PID\n'
              'for the doubling example', fontsize=titlesize)
    #plt.title('Timing analysis', fontsize=titlesize, pad=10)
    plt.xlabel('$d$, Dimension of $M$, $X$ and $Y$', fontsize=labelsize)
    plt.ylabel('Time taken (seconds)', fontsize=labelsize)

    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(['%d' % i for i in 2**(x+1)])#, rotation=90)#, ha='right',
                       #rotation_mode='anchor')
    ax.tick_params(axis='both', labelsize=ticksize)

    plt.grid(alpha=0.5)
    plt.tight_layout()

    plt.savefig('../figures/timing-analysis--tilde.pdf')
    plt.close()

    #plt.show()
