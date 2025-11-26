import argparse
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from pathlib import Path
import pickle
from sklearn.metrics import accuracy_score

from pstimutils import PertStimDataSimple
from perturbation1 import get_ml_data

# Set up argument parser
parser = argparse.ArgumentParser(description='ML model evaluation with SHAP-based feature removal')
parser.add_argument('--dataset', type=str, required=True, help='Data file')
parser.add_argument('--model', type=str, required=True, help='ML model')
parser.add_argument('--iter', type=str, required=True, help='Iteration')
args = parser.parse_args()

# Choose model
if args.model == 'rf':
    model = RandomForestClassifier(n_estimators=1000, max_depth=6)
elif args.model == 'xgb':
    model = XGBClassifier(n_estimators=500)
else:
    print('Invalid model')
    exit(1)

# Load data
psd = PertStimDataSimple(dataspec=args.dataset, verbose=0)

# Get ordering
Xuse, yuse = get_ml_data(psd)[0:2]
X_train, X_test, y_train, y_test = train_test_split(Xuse, yuse, test_size=0.2, stratify=yuse)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
explainer = shap.TreeExplainer(model)
shap_values = explainer(Xuse).values
global_importance = np.abs(shap_values).mean(axis=(0, 2))
order = np.argsort(global_importance)[::-1]

# Loop and collect accuracy
n_neurons = len(order)
accuracy = []
for i in range(n_neurons):
    print(i)
    keep_neurons = order[i:]
    X = np.copy(Xuse)
    X = X[:, keep_neurons]
    X_train, X_test, y_train, y_test = train_test_split(X, yuse, test_size=0.2, stratify=yuse)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))
accuracy = np.array(accuracy)

# Create nested dictionary structure
dd = {}
dd[args.model] = {}
dd[args.model][args.dataset] = {}
dd[args.model][args.dataset][args.iter] = accuracy

# Save dictionary to file
subdir = Path("shap_pruning_accuracy")
subdir.mkdir(exist_ok=True)
output_filename = subdir / f"{args.model}_{args.dataset}_iter{args.iter}.pkl"
with open(output_filename, 'wb') as f:
    pickle.dump(dd, f)

print(f"Results saved to {output_filename}")