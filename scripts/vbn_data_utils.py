#!/usr/bin/env python3

from __future__ import print_function, division

import os
import h5py as h5
import joblib
import pandas as pd

from types import SimpleNamespace


def load_all_data():
    # Location for data files
    data_root = os.path.join(os.environ['HOME'], 'data', 'vbn_supplemental_tables')

    # Container for all data objects
    data = SimpleNamespace()
    data.data_root = data_root

    # Ecephys sessions table
    sessions_table_file = os.path.join(data_root, 'ecephys_sessions_table.csv')
    data.ecephys_sessions_table = pd.read_csv(sessions_table_file)

    # Stimulus presentations table
    stim_table_file = os.path.join(data_root, 'master_stim_table.csv')
    stim_table = pd.read_csv(stim_table_file)
    data.stim_table = stim_table.drop(columns='Unnamed: 0') #drop redundant column

    # Unit table with unit metadata
    unit_table_file = os.path.join(data_root, 'units_with_cortical_layers.csv')
    data.unit_table = pd.read_csv(unit_table_file)
    data.unit_table.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

    # Unit tensor: (neurons, trials, time) array of spike data
    tensor_file = os.path.join(data_root, 'vbnAllUnitSpikeTensor.hdf5')
    data.tensor = h5.File(tensor_file)

    # Running speed data
    running_speeds_file = os.path.join(data_root, 'running_speed-all.pkl')
    data.running_speeds = joblib.load(running_speeds_file)

    return data


def get_ecephys_sessions_area_unit_counts(data):
    """
    Get the number of units in each area, for all sessions.
    """

    session_ids = data.ecephys_sessions_table['ecephys_session_id'].values

    # Count the number of units in each area
    num_units_df = (
        data.unit_table
        .groupby(['ecephys_session_id', 'structure_acronym'])
        ['structure_acronym']
        .count()
    )
    # Convert into a wide data frame with columns as areas; replace NaN with 0
    num_units_df = (
        num_units_df
        .unstack()
        .fillna(0)
        .astype(int)
    )

    # Only keep the session_id's that are in the ecephys sessions table
    num_units_df = num_units_df.loc[session_ids]

    return num_units_df
