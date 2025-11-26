import argparse
import time, os
import numpy as np
from pstimutils import PertStimDataSimple
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
import shap
import pickle
import matplotlib.pyplot as plt

nmatching=2109
nmatching=0
if nmatching:
  rezdir='dafm%d_results' % nmatching
else:
  rezdir='daf_results'

os.makedirs(rezdir, exist_ok=True)

shap.initjs()
try:
    from sputils import keyboard, tally
    havesputils=True
except:
    havesputils=False
    
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='Path to input file')
parser.add_argument('--iter', type=int, help='',default=0)
parser.add_argument('--thresh', type=float, help='',default=1.)
args = parser.parse_args()

tthresh=args.thresh

file_name = f'{rezdir}/{args.file}_{tthresh}_{args.iter}.pkl'
print("file_name=", file_name)
print("tthresh=", tthresh)

start_time = time.time()

# Load variables
nbefore=51
psd = PertStimDataSimple(dataspec=f'../{args.file}', troffset=-nbefore)
psd.get_ml_data_tr(arange=50)
baseactsum=psd.Xuse.sum(1)
trmask=baseactsum>=tthresh
psd.get_ml_data_tr(aoffset=nbefore, arange=50)
pstimactsum=psd.Xuse.sum(1)

X = psd.Xuse
print("trmask.sum()=", trmask.sum())
print("Before X.shape=", X.shape)
num_cols = int(0.2 * psd.Xdata.shape[1])
selected_idxs = np.random.choice(X.shape[1], num_cols, replace=False)
X = X[trmask][:,selected_idxs]
y = psd.yuse[trmask]
nfsamps=X.shape[0]
print("After filtering X.shape=", X.shape)
  
if nmatching:
  print("Now matching to supercritical regime!")
  subsampled_idxs = np.random.choice(X.shape[0], nmatching, replace=False)
  X = X[subsampled_idxs,:]
  y = y[subsampled_idxs]

print("Final X.shape=", X.shape)

#if havesputils and False:
if args.iter == 1:
    plt.hist(baseactsum, 100, alpha=0.6, label='baseline')
    plt.hist(pstimactsum, 100, alpha=0.6, label='pstim')
    plt.title('Filtered samp %d; Used %d' % (nfsamps, X.shape[0]))
    plt.legend()
#    plt.show()
    figname = f'tmpfigs/{args.file}_{tthresh}_50dist.pdf'
    plt.savefig(figname)
    ytt=tally(y)
    print("ytt=", ytt)
#    keyboard('check it')

# Keep track of original indices
og_idx = np.arange(X.shape[1])

accuracy = []
f1 = []
best_predictors = []
total_features = X.shape[1]
features_removed = 0

while X.shape[1] > 0:
    # Fit, predict, score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    model = XGBClassifier(n_estimators=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test,y_pred))
    f1.append(f1_score(y_test,y_pred,average=None))
    
    # Calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X).values.mean(axis=2).mean(axis=0)
    
    # Remove best feature
    best_feature = np.argmax(shap_values.mean(axis=0))
    best_predictors.append(og_idx[best_feature])
    X = np.delete(X, best_feature, axis=1)
    og_idx = np.delete(og_idx, best_feature)
    
    features_removed += 1
    
    # Report progress
    percent_complete = (features_removed / total_features) * 100
    elapsed_time = time.time() - start_time
    print(f"Progress: {percent_complete:.1f}% complete | Features processed: {features_removed}/{total_features} | Time elapsed: {elapsed_time:.2f}s")

total_time = time.time() - start_time
print(f"Completed in {total_time:.2f} seconds")

dd = {}
dd[args.file] = {}
dd[args.file][args.iter] = {}
dd[args.file][args.iter] = {'accuracy':accuracy,
                 'best_predictors':best_predictors}


with open(file_name, 'wb') as f:
    pickle.dump(dd, f)
print(f'Saved to {file_name}')

