# Code used for the decoding analysis and data loading/analysis/plotting in the Ribeiro et al., Critical Scaling of Novelty in the Cortex, Nature Communications, 2025.

Below is a brief summary of the Python and MATLAB scripts posted in this repository.

### pstimutils.py
This is the main utility library for loading, handling, and analyzing perturbation stimulus data . It does so by providing
classes and functions for data loading, manipulation, and analysis of neural recordings from perturbation experiments (stored
in .mat or .npy.gz files! See the Zenodo repository for this manuscript).

### collect_model_performance.py
This script evaluates the performance of Random Forest and XGBoost classifiers on perturbation stimulus datasets. It calculates accuracy and F1 scores, as well as trial retention metrics after filtering, and saves the results.

### dropping_accuracy_filter_trials.py
This script analyzes the impact of feature removal on classifier accuracy for perturbation stimulus data. It iteratively removes the most important features based on SHAP values and records the drop in accuracy.

### perturbation1.py
This script provides functions for processing and analyzing perturbation stimulus datasets. It includes utilities for data loading, feature extraction, cross-validation, and SHAP-based feature pruning.

### plot_gauss2d_with_errorbars.m
This MATLAB script visualizes 2D Gaussian data distributions with error bars. It plots data points, a confidence ellipse, and the major/minor axes of the distribution.

### plot_multi_pkl_files.py
This script reads multiple pickle files containing model performance data and generates plots. It averages results from different simulation runs and visualizes the mean accuracy with optional error bars.

### read_filt_pkl_files.py
This script reads and processes pickle files created with the dropping_accuracy_filter_trials.py analysis, listed above. It aggregates data from multiple simulation runs and plots the mean accuracy (and optionally SEM) as a function of feature removal.

### run_bl_analysis.py
This script evaluatesthe performance of a classifier on background spiking activity. It compares the decoding accuracy of a model trained on spontaneous activity to one trained on stimulus-evoked activity.

### run_class_radius_analysis.py
This script analyzes the impact of excluding neurons within a certain radius of a target cell on classification performance. It iteratively increases the exclusion radius and records the change in F1 score for a specific class.

### run_overall_radius_analysis.py
This script investigates how classifier accuracy is affected by excluding neurons within a certain radius of all target cells. It systematically increases the exclusion radius and evaluates the performance of Random Forest and XGBoost models.

### run_removal_analysis.py
This script performs a feature removal analysis using SHAP values to determine feature importance. It iteratively prunes the most important features and records the impact on the F1 score for a specific class.

### run_removal_analysis_accuracy.py
This script evaluates the effect of feature removal on overall classification accuracy. It uses SHAP values to rank features by importance and then measures the drop in accuracy as features are progressively removed.

### run_sim_bl_analysis.py
This script runs a classification analysis on simulated data to compare the performance of a model trained on non-target cell activity versus a model trained on all cells. It uses parallel processing to run multiple iterations and saves the results.

### run_vstime_pertstim.py
This script analyzes how classifier accuracy changes over time relative to a perturbation stimulus. It runs a decoding analysis in a sliding window around the stimulus onset and saves the time-resolved performance in the results folder (vstresults).

### plot_vstime_pertstim.py
This script plots the accuracy versus the time relative to a perturbation stimulus onset. It reads analysis results obtained with run_vstime_pertstim.py code.
