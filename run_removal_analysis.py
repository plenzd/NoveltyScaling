import argparse
from contextlib import redirect_stdout
import io
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pathlib import Path
import pickle

from pstimutils import PertStimDataSimple
from perturbation1 import *

#Set up argument parser
parser = argparse.ArgumentParser(description='ML model evaluation with SHAP-based feature removal')
parser.add_argument('--dataset', type=str, required=True, help='Data file')
parser.add_argument('--cls', type=int, required=True, help='Class of interest')
parser.add_argument('--model', type=str, required=True, help='ML model')
parser.add_argument('--iter', type=str, required=True, help='Iteration')
args = parser.parse_args()

#Load data
psd = PertStimDataSimple(dataspec=args.dataset,verbose=0)

#Choose model
if args.model == 'rf':
    clf = RandomForestClassifier(n_estimators=1000,max_depth=6)
elif args.model == 'xgb':
    clf = XGBClassifier(n_estimators=500)
else:
    print('Invalid model')
    exit(1)

#Do pruning
f1, best_predictors = shap_pruning(psd, args.cls, clf=clf,keep_targets=True)

# Create nested dictionary structure
dd = {}
dd[args.model] = {}
dd[args.model][args.dataset] = {}
dd[args.model][args.dataset][args.cls] = {}
dd[args.model][args.dataset][args.cls][args.iter] = {
    'f1': f1,
    'best_predictors': best_predictors,
}

#Save dictionary to file
subdir = Path("shap_pruning_keep_targets")
subdir.mkdir(exist_ok=True)
safe_model = args.model.replace('/', '_')
output_filename = f"shap_pruning_keep_targets/{safe_model}_{args.dataset}_cls{args.cls}_iter{args.iter}.pkl"
with open(output_filename, 'wb') as f:
    pickle.dump(dd, f)

print(f"Results saved to {output_filename}")