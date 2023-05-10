#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import itertools as it


def convert_ndarray_to_df(arr, dims, names, value_name):
    cols = it.product(*dims)
    df = pd.DataFrame(cols, columns=names)
    df[value_name] = arr.flatten()
    return df


def get_tensor_unit_table(unit_table, tensor_unit_ids):
    """
    Returns unit table with units ordered as they are in the tensor

    INPUTS:
        unit_table: unit dataframe with unit metadata
        tensor_unit_ids: the unit ids stored in the session tensor (ie session_tensor['unitIds'][()])

    OUTPUTS:
        tensor_unit_table: unit table filtered for the units in the tensor and reordered for convenient indexing
    """

    units = unit_table.set_index('unit_id').loc[tensor_unit_ids].reset_index()

    return units


def get_unit_indices_by_area(unit_table, tensor_unit_ids, areaname, method='equal'):
    """
    Get the indices for the unit dimension of the tensor for only those units in a given area

    INPUTS:
        unit_table: unit dataframe for session
        tensor_unit_ids: the unit ids stored in the session tensor (ie session_tensor['unitIds'][()])
        areaname: the area of interest for which you would like to filter units
        method:
            if 'equal' only grab the units with an exact match to the areaname
            if 'contains' grab all units that contain the areaname in the string.
                This can be useful to, for example, grab all of the units in visual cortex regardless
                of area (areaname would be 'VIS')

    OUTPUT
        the indices of the tensor for the units of interest
    """

    units = get_tensor_unit_table(unit_table, tensor_unit_ids)
    if method == 'equal':
        unit_indices = units[units['structure_acronym']==areaname].index.values

    elif method == 'contains':
        unit_indices = units[units['structure_acronym'].str.contains(areaname)].index.values

    return unit_indices


def get_tensor_for_unit_selection(unit_indices, spikes):
    """
    Subselect a portion of the tensor for a particular set of units. You might try to do this
    with something like spikes[unit_indices] but this ends up being very slow. When the H5 file is saved,
    the data is chunked by units, so reading it out one unit at a time if much faster

    INPUTS:
        unit_indices: the indices of the array along the unit dimension that you'd like to extract
        spikes: the spikes tensor (ie tensor['spikes'] from the h5 file)

    OUTPUT:
        the subselected spikes tensor for only the units of interest
    """

    s = np.zeros((len(unit_indices), spikes.shape[1], spikes.shape[2]), dtype=bool)
    for i, u in enumerate(unit_indices):
        s[i] = spikes[u]

    return s


def get_stim_table_and_unit_tensor_for_session(session_of_interest, structures, data,
                                               omission_in_trial=False):

    session_stim_table = data.stim_table[data.stim_table['session_id'] == session_of_interest].reset_index()
    session_tensor = data.tensor[str(session_of_interest)]

    session_units = data.unit_table[data.unit_table['ecephys_session_id'] == session_of_interest]
    session_units = session_units.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
    session_units = get_tensor_unit_table(session_units, session_tensor['unitIds'][()])

    area_unit_indices = []     # Concatenated list of unit indices for all areas in `structures`
    unit_indices_by_area = {}  # Dict mapping areas to unit indices in `area_unit_indices`
    num_units = 0
    for area_of_interest in structures:
        unit_indices = get_unit_indices_by_area(session_units, session_tensor['unitIds'][()], area_of_interest)
        area_unit_indices.append(unit_indices)
        unit_indices_by_area[area_of_interest] = np.arange(num_units, num_units + len(unit_indices))
        num_units += len(unit_indices)

    unit_indices = np.concatenate(area_unit_indices)
    unit_tensor = get_tensor_for_unit_selection(unit_indices, session_tensor['spikes'])

    # Add a new column to indicate if there was any omitted flash in each trial
    if omission_in_trial:
        session_stim_table['omission_in_trial'] = (
            session_stim_table.groupby('behavior_trial_id')
            ['omitted']
            .transform(lambda x: x.any())
        )
        session_stim_table = session_stim_table.dropna(subset=['omission_in_trial'])

    return session_stim_table, unit_tensor, unit_indices_by_area


def get_binned_spike_counts():
    # XXX Write properly

    # Count spikes in bins of size `bin_size`
    bin_size = 10  # milliseconds
    s = unit_tensor.shape
    num_bins = s[2] // bin_size
    unit_tensor_condensed = unit_tensor.reshape((*s[:2], num_bins, bin_size)).sum(axis=-1)
    # Shape of unit_tensor_condensed is now (num_units, num_flashes, num_bins)

    # Number of time bins to pass as features to the classifier
    window_size = 50 // bin_size
    # Number of time bins to move forward before recomputing the classifier
    step_size = 10 // bin_size

    time_axis_bins = np.arange(0, num_bins - window_size + 1, step_size)  # in units of bins
    time_axis_ms = time_axis_bins * bin_size                              # in units of milliseconds


def get_interpolated_running_speed():
    # XXX Write properly

    # Interpolate running speeds
    running_speed = running_speeds[session_of_interest]
    time_points = session_stim_table['start_time'].values
    time_points = time_points[:, np.newaxis] + time_axis_bins / 1000 + bin_size / 1000 / 2
    interp_run_speeds = np.interp(time_points, running_speed['timestamps'].values, running_speed['speed'].values)
    interp_run_speeds = np.abs(interp_run_speeds)  # Consider only magnitude of speed

    return interp_run_speeds
