import os
import re
from contextlib import redirect_stdout
import io
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

from pstimutils import PertStimDataSimple
from perturbation1 import *

datasets = []
for filename in os.listdir(f'{os.getcwd()}/data'):
    match = re.match(r'BigDataSetA_(\d+)\.npy\.gz', filename)
    if match: 
        datasets.append(match.group(1))
datasets.sort(key=int)

dd = {}

for dataset in datasets:
   print(dataset)
   with redirect_stdout(io.StringIO()): #Suppress output
       psd = PertStimDataSimple(dataspec=f'data/BigDataSetA_{dataset}.npy.gz',grpid=0,keep_xdata=True)
   
   #Track accuracy for each radius threshold in current dataset
   rf_accuracy = []
   xgb_accuracy = []
   
   X = get_ml_data(psd)[0]
   n_neurons = X.shape[1]
   drops = []
   
   #Test different radius thresholds
   for radius in range(51):
       print(radius)
       X, y = get_ml_data(psd, threshold=0.5, keep_targets=False, filter_radius=True, radius_threshold=radius)[:2]

       #Evaluate classifiers
       rf_accuracy.append(cv_metric(X,y,metric='accuracy',folds=10,clf=RandomForestClassifier(n_estimators=1000)).mean())
       xgb_accuracy.append(cv_metric(X,y,metric='accuracy',folds=10,clf=XGBClassifier(n_estimators=500)).mean())

       drops.append(n_neurons-X.shape[1])
   
   dd[dataset] = {'rf_accuracy':rf_accuracy,
                  'xgb_accuracy':xgb_accuracy,
                  'drops':drops}
   
with open('overall_radius_analysis.pkl', 'wb') as f:
    pickle.dump(dd, f)

print('Results saved to overall_radius_analysis.pkl')