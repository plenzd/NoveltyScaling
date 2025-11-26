import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import pickle

from pstimutils import PertStimDataSimple
from perturbation1 import *

def downsample_to_match(source_labels, reference_labels, tolerance=0.2, random_state=None):
    """
    Downsample source labels to match reference label distribution within tolerance.
    
    Parameters
    ----------
    source_labels : array-like
        Labels to downsample from
    reference_labels : array-like
        Target label distribution
    tolerance : float, default=0.2
        Minimum samples = reference_count * (1 - tolerance)
    random_state : int or None
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Indices of selected samples from source
    """
    rng = np.random.default_rng(random_state)
    
    classes, reference_counts = np.unique(reference_labels, return_counts=True)
    source_counts = np.unique(source_labels, return_counts=True)[1]
    
    selected_idxs = []
    for i, cls in enumerate(classes):
        min_count = int(reference_counts[i] * (1 - tolerance))
        max_count = reference_counts[i]
        source_count = source_counts[i]
        cls_idxs = np.where(source_labels == cls)[0]
        
        if source_count < min_count:
            raise ValueError(
                f"Class {cls}: only {source_count} in source (min needed: {min_count})"
            )
        
        sample_count = min(source_count, max_count)
        selected_idxs.append(rng.choice(cls_idxs, size=sample_count, replace=False))
    
    return np.concatenate(selected_idxs)


def evaluate_background_spikes(psd, dataset, n_iterations=100, tolerance=0.5):
    """
    Evaluate XGBoost classifier performance on background spikes data.
    
    Parameters
    ----------
    psd : PertStimDataSimple
        Loaded dataset object
    dataset : str
        Dataset identifier
    n_iterations : int, default=100
        Number of train/test iterations
    tolerance : float, default=0.5
        Tolerance for downsampling to match experimental distribution
    
    Returns
    -------
    dict or None
        Dictionary with 'accuracy' and 'f1' arrays, or None if validation fails
    """
    # Get experimental labels and allowed classes
    _, exp_y, allowed_classes = get_ml_data(psd, encode_labels=False)[:3]
    min_count = int(min(np.unique(exp_y, return_counts=True)[1]) * 0.8)
    
    # Get background spikes
    X, y = get_bg_spks(psd, min_count=min_count, encode_labels=False)[:2]
    
    # Filter to allowed classes
    mask = np.isin(y, allowed_classes)
    X = X[mask]
    y = y[mask]
    
    # Validate class coverage
    if set(np.unique(y)) != set(np.unique(exp_y)):
        print(f"Missing classes in background spikes data for {dataset}")
        return None
    
    # Run multiple train/test iterations
    accuracy_scores = []
    f1_scores = []
    
    for i in range(n_iterations):
        try:
            # Downsample to match experimental distribution
            selected = downsample_to_match(source_labels=y, reference_labels=exp_y, tolerance=tolerance)
            
            # Prepare data
            X_selected = X[selected]
            y_selected = LabelEncoder().fit_transform(y[selected])
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, stratify=y_selected)
            
            # Train and evaluate
            model = XGBClassifier(n_estimators=500)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average=None))
            
        except ValueError as e:
            print(f"Skipping dataset {dataset} on iteration {i}: {e}")
            break
    
    if not accuracy_scores:
        return None
    
    return {
        'accuracy': np.array(accuracy_scores),
        'f1': np.array(f1_scores)
    }


def run_background_analysis(datasets, psds, n_iterations=100, tolerance=0.5):
    """
    Run background spikes classification analysis on multiple datasets.
    
    Parameters
    ----------
    datasets : list
        List of dataset identifiers
    psds : dict
        Dictionary mapping datasets to loaded PSD objects
    n_iterations : int, default=100
        Number of train/test iterations per dataset
    tolerance : float, default=0.5
        Tolerance for downsampling
    
    Returns
    -------
    dict
        Results dictionary with accuracy and F1 scores for each dataset
    """
    results = {}
    
    for dataset in datasets:
        print(f"Processing background analysis for dataset {dataset}...")
        psd = psds[dataset]
        
        result = evaluate_background_spikes(
            psd, dataset, n_iterations, tolerance
        )
        
        if result is not None:
            results[dataset] = result
    
    return results


if __name__ == "__main__":
    # Load datasets
    datasets = get_dataset_names()
    psds = {dataset: get_psd(dataset) for dataset in datasets}
    
    # Run background spikes analysis
    background_results = run_background_analysis(datasets=datasets,
        psds=psds,
        n_iterations=100,
        tolerance=0.5
    )
    
    # Save results
    with open('background_analysis_results.pkl', 'wb') as f:
        pickle.dump(background_results, f)