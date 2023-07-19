# gpid

A package that implements Partial Information Decompositions for Gaussian random variables, including some basic bias correction.

## Requirements

Setup the environment using:

```setup
conda env create -f environment.yml
conda activate gpid
```

We include two environment files, `environment.yml` and `minimal_environment.yml`. The latter is a condensed version that has not been tested, but could help resolve package inconsistencies by specifying a minimal list of packages.

Install the `gpid` package by running the following within the folder containing the `setup.py` file.

```
pip install -e .
```

To run the `mult_poisson_gt.py` script, please install [`computeUI`](https://github.com/infodeco/computeUI) from the paper of Banerjee et al. (ISIT, 2018) ([arxiv version](https://arxiv.org/abs/1709.07487)), which provides the `admUI` module.

To run the scripts for analyzing the Allen Institute's [Visual Behavior Neuropixels data](https://allensdk.readthedocs.io/en/latest/visual_behavior_neuropixels.html), please install the [`allensdk`](https://allensdk.readthedocs.io/en/latest/install.html).

We include a local copy of the [statannotations](https://github.com/trevismd/statannotations) package due to some installation incompatibilities that we faced.

All code was tested on a Linux machine running Ubuntu 20.04. If some saved pickle files do not open as expected, please try updating the respective package (e.g., joblib or pandas).

## The `gpid` package

The main source code used for computing the `~_G`-PID can be found in the `src/` folder.

1. The `generate.py` script is used to generate various covariance matrix configurations.
2. The `tilde_pid.py` script is used to estimate the tilde-PID.
3. The `estimate.py` script is used to estimate the delta-PID.
4. The `mmi_pid.py` script is used to estimate the MMI-PID.
5. The `utils.py` script contains some common utility functions that are used by one or more of the above estimators.

## Description of scripts

The `scripts/` folder contains the scripts used for computing the PIDs for the various examples in this paper.
All of these scripts need to be run from within this folder for them to execute properly.
Scripts that do not start with `plot_` will typically configure an example, compute PID values and store results.
Scripts that start with `plot_` should be run after, and will plot the corresponding results.

1. `canonical_examples.py` and  `plot_cacnonical_exs.py`: used to compute PIDs in Figure 1.
2. `gain_angle_sweeps.py` and `plot_gain_angle_sweeps.py`: used to compute examples in Figure 2.
3. `doubling_example.py`, `plot_doubling_example.py` and `plot_doubling_error.py`: used to compute PID values for Example 10, shown in Figure 3 of the paper. The `_error` script is used to plot the absolute and relative errors shown in Figure 8 in the supplementary material.
4. `sample_convergence.py`, `plot_final_bias_corr.py` and `plot_all_bias_corr.py`: used for bias-correction examples shown in Figure 4 (plotted using the `final` script). The `all` script is used for plotting all examples, shown in the supplementary material.
5. `mult_poisson_example.py`, `mult_poisson_gt.py`, `mult_poisson_utils.py`, `plot_mult_poisson.py`: used for the multivariate Poisson example shown in Figure 5. Described in detail in `mult_poisson_example.py`. Requires the `computeUI` package to be installed.
6. `vbn_analysis.py`, `vbn_data_utils.py`, `vbn_utils.py`, `plot_final_vbn.py`, `plot_final_vbn_allpicomps.py`: used for analyzing the Visual Behavior Neuropixels data from the Allen Institute, results of which are shown in Figure 6. The `_allpicomps` plotting script is used for plotting Figures 15 and 16 shown in the supplementary material. Also see the notes below on accessing and preprocessing the data.
7. `bootstrap_ci.py` and `plot_all_bootstrap_ci.py`: used for preliminary analysis of confidence intervals, presented in Figures 11 and 12, in the supplementary material.

Some miscellaneous scripts not presented in the paper may also exist in the repository.

## Accessing the Visual Behavior Neuropixels data

Running `vbn_analysis.py` requires accessing and preprocessing the Visual Behavior Neuropixels data.

The `make_stim_trial_tables.py`, `VBN_concatenate_stim_trials_tables.py` and `vbn_preprocess.py` scripts were used to fetch and preprocess the data, however, by default they would download about 2TB of data to a local disk (paths in these scripts will need to be changed). As a result, these have not been tested in a brand new environment, but we are committed to making a more reproducible version of these scripts if our paper gets accepted.

Similarly, paths in `vbn_data_utils.py` will need to be changed.

One of the data files is available under the `data/` folder.

## Saved results

All saved results are located in the `results/` folder.
Re-running the code will overwrite these results.

## Figures

All figures that are generated using the `plot_*.py` scripts are stored in the `figures/` folder.
Running these scripts will overwrite these figures.
