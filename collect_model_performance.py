import os
import re
import io
from contextlib import redirect_stdout
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from pstimutils import PertStimDataSimple
from perturbation1 import get_ml_data, cv_metric

def compute_decoder_metrics(psd, dataset_id, valid_targets):
    """Compute accuracy and F1 scores for RF and XGBoost classifiers."""
    X, y, classes_to_keep = get_ml_data(psd)[:3]
    vt_label = [i for i, cls in enumerate(classes_to_keep) 
                if cls in valid_targets[dataset_id]]
    
    # Random Forest metrics
    rf_accuracy = cv_metric(X, y, metric='accuracy', folds=10, clf=RandomForestClassifier(n_estimators=1000))
    rf_f1 = cv_metric(X, y, metric='f1', folds=10, clf=RandomForestClassifier(n_estimators=1000))
    
    # XGBoost metrics
    xgb_accuracy = cv_metric(X, y, metric='accuracy', folds=10, clf=XGBClassifier(n_estimators=500))
    xgb_f1 = cv_metric(X, y, metric='f1', folds=10, clf=XGBClassifier(n_estimators=500))
    
    return {
        'vt_label': vt_label,
        'rf_accuracy': rf_accuracy,
        'rf_f1': rf_f1,
        'xgb_accuracy': xgb_accuracy,
        'xgb_f1': xgb_f1
    }

def compute_trial_retention_metrics(psd):
    """Compute percentage of remaining trials and F1 scores after filtering."""
    X = psd.Xdata
    y = psd.ydata
    ec = psd.ensemble_cells
    
    # Create mask for non-zero spike trials
    mask = np.ones(len(y), dtype=bool)
    for i, label in enumerate(y):
        col_idx = ec[label]
        if X[i, col_idx] == 0:
            mask[i] = False
    
    # Calculate percentage of remaining samples per class
    unique_classes, original_counts = np.unique(y, return_counts=True)
    original_counts_dict = dict(zip(unique_classes, original_counts))
    
    filtered_y = y[mask]
    remaining_counts = {cls: np.sum(filtered_y == cls) for cls in unique_classes}
    
    percentage_remaining = {cls: remaining_counts[cls] / original_counts_dict[cls] for cls in unique_classes}
    
    # Compute F1 scores
    rf_f1 = cv_metric(X, y, metric='f1', folds=10, clf=RandomForestClassifier(n_estimators=1000))
    xgb_f1 = cv_metric(X, y, metric='f1', folds=10, clf=XGBClassifier(n_estimators=500))
    
    return {
        'percentage_remaining': percentage_remaining,
        'rf_f1': rf_f1,
        'xgb_f1': xgb_f1
    }

if __name__ == "__main__":
    # Define valid targets for each dataset
    valid_targets = {
        '02': [1, 2, 4, 5, 6],
        '06': [1, 3, 5],
        '07': [0, 1, 2, 3, 4, 5],
        '32': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        '34': [1, 2, 3, 6],
        '35': [0, 2, 3, 6, 7, 8, 9],
        '36': [0, 1, 2, 3, 4, 6, 7, 8]
    }

    # Get datasets
    datasets = []
    for filename in os.listdir(f'{os.getcwd()}/data'):
        match = re.match(r'BigDataSetA_(\d+)\.npy\.gz', filename)
        if match:
            datasets.append(match.group(1))
    datasets.sort(key=int)

    # Run analysis
    results = {}
    for dataset_id in datasets:
        print(f"Processing dataset {dataset_id}...")
        results[dataset_id] = {}
        
        # Load data with spike activity type
        with redirect_stdout(io.StringIO()):
            psd = PertStimDataSimple(dataspec=f'data/BigDataSetA_{dataset_id}.npy.gz')
        
        # Compute decoder metrics
        results[dataset_id]['decoder_metrics'] = compute_decoder_metrics(psd, dataset_id, valid_targets)
        
        # Compute trial retention metrics
        results[dataset_id]['trial_retention'] = compute_trial_retention_metrics(psd)

    # Save results
    import pickle
    with open('model_results.pkl', 'wb') as f:
        pickle.dump(results, f)