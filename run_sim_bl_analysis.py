import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from tqdm import tqdm
import os
import pickle

from pstimutils import PertStimDataSimple

def get_non_tc_spks(psd, min_count=5):
    """Extract non-TC spike data and prepare training data."""
    baseline = psd.trials_raster[100:, :, :]
    baseline = np.swapaxes(baseline, 1, 2)
    baseline = baseline.reshape(50, baseline.shape[0] // 50, baseline.shape[1], baseline.shape[2])
    baseline = np.concatenate(baseline.mean(axis=0), axis=0)

    raster = np.copy(baseline)
    ec_spks = raster[:, psd.ensemble_cells.flatten()]
    multilabel_arr = (ec_spks > 0).astype(int)

    unique_rows, counts = np.unique(multilabel_arr, axis=0, return_counts=True)
    valid_patterns = unique_rows[(np.sum(unique_rows, axis=1) == 1) & (counts > min_count)]

    indices = []
    for pattern in valid_patterns:
        matches = np.all(multilabel_arr == pattern, axis=1)
        indices.extend(np.where(matches)[0])
    indices = np.array(indices)

    selected_rows = multilabel_arr[indices]
    labels = np.argmax(selected_rows, axis=1)

    exclude_cols = psd.ensemble_cells[np.unique(labels)]
    X_ntc = np.delete(raster[indices], exclude_cols, axis=1)
    y_ntc = LabelEncoder().fit_transform(labels)

    psd.get_ml_data_tr(arange=50)
    X = psd.Xuse
    y = psd.yuse

    return X_ntc, y_ntc, X, y


def run_iteration(i, X_ntc_full, y_ntc_full, X_full, y_full, max_per_class=300, col_frac=0.2):
    """Run a single iteration of training and evaluation."""
    X_ntc = X_ntc_full.copy()
    y_ntc = y_ntc_full.copy()
    X = X_full.copy()
    y = y_full.copy()
    
    # Subsample per class
    selected_indices = []
    for cls in np.unique(y_ntc):
        cls_indices = np.where(y_ntc == cls)[0]
        chosen = np.random.choice(cls_indices, min(len(cls_indices), max_per_class), replace=False)
        selected_indices.extend(chosen)
    selected_indices = np.array(selected_indices)
    
    X_ntc = X_ntc[selected_indices]
    y_ntc = y_ntc[selected_indices]

    # Random column selection
    num_cols = int(col_frac * X.shape[1])
    selected_idxs = np.random.choice(X.shape[1], num_cols, replace=False)
    X_ntc = X_ntc[:, selected_idxs]
    X = X[:, selected_idxs]

    # Train/test split and model for NTC
    X_train, X_test, y_train, y_test = train_test_split(X_ntc, y_ntc, test_size=0.2, random_state=i)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    f1_ntc = f1_score(y_test, model.predict(X_test), average='macro')

    # Train/test split and model for full X
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    f1 = f1_score(y_test, model.predict(X_test), average='macro')
    
    return i, f1_ntc, f1


def main(sim, n_iters=100):
    datasets = [f for f in os.listdir('..') if f.startswith(sim)]
    psd_list = [PertStimDataSimple(dataspec=f'../{dataset}', verbose=0) for dataset in datasets]

    all_results = {}

    for psd_idx, psd in enumerate(psd_list):
        print(f"Running PSD {psd_idx}")
        X_ntc, y_ntc, X, y = get_non_tc_spks(psd)

        results = Parallel(n_jobs=-1)(
            delayed(run_iteration)(i, X_ntc, y_ntc, X, y) 
            for i in tqdm(range(n_iters), desc=f"PSD {psd_idx} Iterations")
        )

        results.sort(key=lambda x: x[0])
        f1_ntc, f1 = zip(*[(r[1], r[2]) for r in results])
        all_results[psd_idx] = {'f1_ntc': list(f1_ntc), 'f1': list(f1)}

    with open(f'{sim}_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run classification analysis on simulation data')
    parser.add_argument('sim', type=str, help='Simulation name prefix (e.g., SimData300ctxB101P00045)')
    parser.add_argument('--n_iters', type=int, default=100, help='Number of iterations (default: 100)')
    
    args = parser.parse_args()
    main(args.sim, n_iters=args.n_iters)