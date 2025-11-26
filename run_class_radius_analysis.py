import argparse
import numpy as np
from contextlib import redirect_stdout
import io
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pathlib import Path
import pickle

from pstimutils import PertStimDataSimple
from perturbation1 import get_ml_data, cv_metric

# Set up argument parser
parser = argparse.ArgumentParser(description='Per class radius analysis for perturbation decoding model')
parser.add_argument('--dataset', type=str, required=True, help='Data file')
parser.add_argument('--cls', type=int, required=True, help='Class of interest')
parser.add_argument('--model', type=str, required=True, help='ML model')
args = parser.parse_args()

# Load data
with redirect_stdout(io.StringIO()):  # Suppress output
    psd = PertStimDataSimple(dataspec=f'data/BigDataSetA_{args.dataset}.npy.gz', grpid=0, keep_xdata=True)

# Choose model
if args.model == 'rf':
    clf = RandomForestClassifier(n_estimators=1000)
elif args.model == 'xgb':
    clf = XGBClassifier(n_estimators=500)
else:
    print('Invalid model')
    exit(1)  # Exit if invalid model

# Get dataset properties
X,y = get_ml_data(psd)[:2]
n_neurons = X.shape[1]

# Initialize results arrays
n_drops = []
f1 = []

# Iteratively increase zone of exclusion around specific target cell
for radius in range(51):
    print(radius)
    # Get data and filter by radius
    Xuse, yuse, _, filtered_ec = get_(psd, threshold=0.5, keep_targets=True)
    tc_idx = filtered_ec[args.cls]
    cellcoords = psd.cellcoords.T
    dist_matrix = cdist(cellcoords, cellcoords, metric='euclidean')
    within_thresh = np.where(dist_matrix[tc_idx] < radius)[1]
    rmv_idxs = np.concatenate((within_thresh, filtered_ec.flatten()))
    rmv_idxs = np.unique(rmv_idxs)
    Xuse = np.delete(Xuse, rmv_idxs, axis=1)

    # Evaluate classifiers
    f1.append(cv_metric(Xuse, yuse, metric='f1', folds=10, clf=RandomForestClassifier(n_estimators=1000))[:,args.cls].mean())
    n_drops.append(n_neurons - Xuse.shape[1])

# Process results
f1 = np.array(f1)
n_drops = np.array(n_drops)
n_drops = np.diff(n_drops)
n_drops = np.insert(n_drops, 0, 0)

dd = {}
dd[args.model] = {}
dd[args.model][args.dataset] = {}
dd[args.model][args.dataset][args.cls] = {
    'f1': f1,
    'n_drops':n_drops
}

# Save dictionary to file
subdir = Path("per_class_radius_analysis")
subdir.mkdir(exist_ok=True)
output_filename = f"per_class_radius_analysis/{args.model}_{args.dataset}_cls{args.cls}.pkl"
with open(output_filename, 'wb') as f:
    pickle.dump(dd, f)

print(f"Results saved to {output_filename}")