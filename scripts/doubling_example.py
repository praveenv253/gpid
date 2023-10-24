#!/usr/bin/env python3

from __future__ import print_function, division

import joblib
import numpy as np
import time

from gpid.tilde_pid import exact_gauss_tilde_pid
from gpid.generate import random_rotation_mxy, merge_covs


if __name__ == '__main__':
    gains = np.linspace(0, 3, 10)
    gains[3] = 0.99  # A gain of 1 is unstable
    #gains = np.array([3.0,])
    num_doubles = 10
    random_rotn = True

    pid_vals = []
    sizes = []
    covs = []
    time_taken = []

    # import joblib
    # covs_saved = joblib.load('bge_out2.pkl')['covs']

    try:
        for i, gain in enumerate(gains):
            print(i, end=': ', flush=True)

            covs.append([])
            time_taken.append([])
            for j in range(num_doubles):
                #print(j, end=' ', flush=True)
                start_time = time.perf_counter()

                if j == 0:
                    dm, dx, dy = 2, 2, 2
                    hx = np.array([[gain, 0], [0, 1]])
                    hy = np.array([[1, 0], [0, np.sqrt(2)]])
                    sigm = np.eye(dm)
                    sigx_m = np.eye(dx)
                    sigy_m = np.eye(dy)
                    sigw = np.zeros((dx, dy))
                    cov = np.block([[sigm, sigm @ hx.T, sigm @ hy.T],
                                    [hx @ sigm, hx @ sigm @ hx.T + sigx_m, hx @ sigm @ hy.T + sigw],
                                    [hy @ sigm, hy @ sigm @ hx.T + sigw.T, hy @ sigm @ hy.T + sigy_m]])
                    cov_old = cov.copy()
                else:
                    cov, dm, dx, dy = merge_covs(cov_old, cov_old.copy(), dm, dx, dy)
                    cov_old = cov.copy()

                if random_rotn:
                    cov = random_rotation_mxy(cov, dm, dx, dy)

                #cov = covs_saved[i][j]
                dm = dx = dy = 2**(j+1)

                #covs[i].append(cov.copy())

                ret1 = exact_gauss_tilde_pid(cov, dm, dx, dy, ret_t_sigt=False)
                #cov, dm, dx, dy = switch_x_and_y(cov, dm, dx, dy)
                #ret2 = exact_gauss_tilde_pid(cov, dm, dx, dy, ret_t_sigt=False)

                #if ret1[3] < ret2[3]:
                #    imxy, uix, uiy, ri, si = ret1[2], *ret1[-4:]
                #else:
                #    # Note the change in order of uiy and uix wrt to above
                #    imxy, uiy, uix, ri, si = ret2[2], *ret2[-4:]

                imxy, uix, uiy, ri, si = ret1[2], *ret1[-4:]
                pid_vals.append([imxy, uix, uiy, ri, si])
                sizes.append(cov.shape[0])

                end_time = time.perf_counter()
                time_taken[i].append(end_time - start_time)
                print('%.1f' % time_taken[i][-1], end=' ', flush=True)

            print()
    except:
        raise
    finally:
        #pid_vals = np.array(pid_vals).reshape((-1, num_doubles, 5))
        joblib.dump({'covs': covs, 'pid_vals': pid_vals, 'gains': gains,
                     'num_doubles': num_doubles, 'time_taken': time_taken},
                    '../results/doubling_example.pkl')
