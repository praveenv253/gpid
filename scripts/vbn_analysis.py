#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

from gpid.tilde_pid import exact_gauss_tilde_pid
from vbn_data_utils import load_all_data, get_ecephys_sessions_area_unit_counts
from vbn_utils import get_stim_table_and_unit_tensor_for_session


def get_change_non_change_flash_masks(session_stim_table):
    flashes_for_no_change = np.r_[4:11]  # Flashes from 4-10 for no change

    is_non_change_flash = (
        session_stim_table.trial_flash.isin(flashes_for_no_change)
        & ~session_stim_table['omitted']
        & ~session_stim_table['previous_omitted']
        & ~session_stim_table['lick_for_flash']
        & (session_stim_table['image_name']
           == session_stim_table['initial_image_name'])
        & (session_stim_table['reward_rate'] >= 2)
    )

    is_change_flash = (
        session_stim_table['is_change']
        & (session_stim_table['reward_rate'] >= 2)
    )

    return is_change_flash, is_non_change_flash


def get_pids_for_session(session_id, structures, data):
    ret = get_stim_table_and_unit_tensor_for_session(session_id, structures, data)
    session_stim_table, unit_tensor, unit_indices_by_area = ret

    # Get change and non-change flash indices for familiar and novel session
    # Let the brain activity be the full 50-250ms to start with.
    # Compare the PID between VISp, VISl and one other area (say VISal)
    # See if this is different for change/non-change and familiar/novel
    is_change_flash, is_non_change_flash = get_change_non_change_flash_masks(session_stim_table)

    #time_of_interest = np.r_[50:250]  # Milliseconds
    times_of_interest = np.r_[0:250].reshape((-1, 50))  # Milliseconds
    cond_names = ['change', 'non_change']

    pids = []
    for time_of_interest in times_of_interest:
        t = time_of_interest[0]
        print(t, end='', flush=True)

        # Sum spike counts in time of interest; shape (num_neurons, num_flashes)
        activity_of_interest = unit_tensor[:, :, time_of_interest].sum(axis=2)

        for name, flash_mask in zip(cond_names,
                                    [is_change_flash, is_non_change_flash]):
            print(name[0], end='', flush=True)

            # Compute covariance matrix
            dm, dx, dy = (unit_indices_by_area[area].size for area in structures)
            activity_subset = activity_of_interest[:, flash_mask].T

            try:
                # Use the top-10 principal components in each area to keep things manageable
                n_comps = 10
                area_activity_pcs = []
                expld_vars = []
                for area in structures:
                    pca = PCA(n_components=n_comps)
                    area_activity_pc = pca.fit_transform(activity_subset[:, unit_indices_by_area[area]])
                    area_activity_pcs.append(area_activity_pc)
                    expld_vars.append(pca.explained_variance_ratio_)
                activity_pcs = np.hstack(area_activity_pcs)
                dm, dx, dy = [n_comps,] * 3

                cov = np.cov(activity_pcs.T)
                ret = exact_gauss_tilde_pid(cov, dm, dx, dy, unbiased=True,
                                            sample_size=flash_mask.sum())
                #(imx, imy, imxy_debiased, union_info, obj, uix, uiy, ri, si) = ret
                imxy_debiased, uix, uiy, ri, si = (ret[2], *ret[-4:])
            except:
                imxy_debiased, uix, uiy, ri, si = [np.nan,] * 5

            pids.append({'imxy': imxy_debiased, 'uix': uix, 'uiy': uiy, 'ri': ri, 'si': si})

        print(' ', end='')

    index = pd.MultiIndex.from_product([times_of_interest[:, 0], cond_names])
    pid_df = pd.DataFrame.from_records(pids, index=index)

    return pid_df


if __name__ == '__main__':
    data = load_all_data()
    area_unit_counts = get_ecephys_sessions_area_unit_counts(data)

    # Area-wise noise alignment: each of these areas is examined separately
    #structures = ['VISp', 'VISl', 'VISal', 'VISam', 'VISpm']
    structures = ['VISp', 'VISl', 'VISam']

    min_unit_count_thresh = 20
    session_indices = area_unit_counts.index[
        (area_unit_counts[structures] > min_unit_count_thresh).all(axis=1)
    ]

    # Subselect sessions from mice with both familiar and novel sessions
    mice_with_both_fam_and_nov = (
        data.ecephys_sessions_table
        .set_index('ecephys_session_id')
        .loc[session_indices]
        .reset_index()
        .set_index(['mouse_id', 'experience_level'])
        .sort_index()
        ['ecephys_session_id']
        .unstack()
        .dropna()
        .astype('int')
        .stack()
    )

    ret = {}
    for i, session_id in enumerate(mice_with_both_fam_and_nov):
        #if i >= 4:
        #    break
        print('%d, %d' % (i, session_id), end=': ', flush=True)
        ret[session_id] = get_pids_for_session(session_id, structures, data)
        print()

    ret = pd.concat(ret)
    ret.index.set_names(['session_id', 'time', 'cond'], inplace=True)

    ret = ret.unstack().unstack()

    # Make this table map from session_id to mouse_id and experience_level
    mice_with_both_fam_and_nov = (
        mice_with_both_fam_and_nov
        .rename('session_id')
        .reset_index()
        .set_index('session_id')
    )
    # Add a column level to mouse_id and experience_level in preparation for
    # merging with the results for each session in ret.
    # The dummy column level is to avoid a future deprecation warning
    mice_with_both_fam_and_nov.columns = pd.MultiIndex.from_product([
        ['dummy1'], ['dummy2'], mice_with_both_fam_and_nov.columns
    ])

    df = pd.merge(mice_with_both_fam_and_nov, ret, how='inner',
                  left_on='session_id', right_index=True)
    # Set the index back to mouse_id and experience_level
    df = df.set_index([('dummy1', 'dummy2', 'mouse_id'),
                       ('dummy1', 'dummy2', 'experience_level')])
    df.index.rename(['mouse_id', 'experience_level'], inplace=True)
    df.columns.rename(['pid_comp', 'cond', 'time'], inplace=True)
    # Pivot image_name from columns into the index
    df = df.stack().stack()

    # Separate the shared images in the novel context
    #shared_images = ['im083_r', 'im111_r']
    #df['image_type'] = df['experience_level']
    #df.loc[(df['experience_level'] == 'Novel')
    #        & df['image_name'].isin(shared_images), 'image_type'] = 'Shared'
    #df = df.groupby(['mouse_id', 'image_type']).mean()

    # Remove rows in the results table with NaNs.
    bad_mouse_ids = df[df.isna().any(axis=1)].index.get_level_values('mouse_id').values
    cleaned_df = df.drop(bad_mouse_ids, level='mouse_id')

    print(cleaned_df)
    cleaned_df.to_csv('vbn-pids.csv')
