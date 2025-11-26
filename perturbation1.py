import os
import re
from contextlib import redirect_stdout
import io
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score,accuracy_score
import shap
shap.initjs()

from pstimutils import PertStimDataSimple

def get_dataset_names():
    return sorted(
        (f"data/BigDataSetA_{match.group(1)}.npy.gz"
         for filename in os.listdir(f"{os.getcwd()}/data")
         if (match := re.match(r"BigDataSetA_(\d+)\.npy\.gz", filename))),
        key=lambda x: int(re.search(r"(\d+)", x).group(1))
    )

def get_psd(dataset,adtype='spk'):
    with redirect_stdout(io.StringIO()): #Suppress output
        psd = PertStimDataSimple(dataspec=dataset,grpid=0,keep_xdata=True,adtype=adtype)
    return psd

def get_ml_data(psd,threshold=0.5,keep_targets=False,filter_radius=False,radius_threshold=15,encode_labels=True):
    #Load variables
    X = psd.Xdata
    y = psd.ydata
    ec = psd.ensemble_cells

    #Generate zero spike trial mask
    mask = np.ones(len(y), dtype=bool)
    for i, label in enumerate(y):
        col_idx = ec[label]  
        row_idx = i  
        if X[row_idx, col_idx] == 0:
            mask[i] = False

    #Filter classes by percent remaining samples
    unique_classes, original_counts = np.unique(y, return_counts=True)
    original_counts_dict = dict(zip(unique_classes, original_counts))
    filtered_y = y[mask]
    remaining_counts = {cls: np.sum(filtered_y == cls) for cls in unique_classes}
    percentage_remaining = {
        cls: (remaining_counts[cls] / original_counts_dict[cls])
        for cls in unique_classes
    }
    classes_to_keep = {cls for cls, pct in percentage_remaining.items() if pct >= threshold}
    classes_to_keep = np.array(list(classes_to_keep))
    final_mask = np.array([cls in classes_to_keep for cls in y]) & mask

    ec_idx = np.unique(y[final_mask])
    filtered_ec = ec[ec_idx]
    if encode_labels == True:
        yuse = LabelEncoder().fit_transform(y[final_mask])
    else:
        yuse = y[final_mask]
    filtered_X = X[final_mask]

    #Filter neurons within given radius of target
    if filter_radius == True:
        cellcoords = psd.cellcoords.T
        dist_matrix = cdist(cellcoords, cellcoords, metric='euclidean')
        rmv_idxs = []
        for i in filtered_ec:
            within_thresh = np.where(dist_matrix[i] < radius_threshold)[1]
            rmv_idxs.append(within_thresh)
        rmv_idxs.append(filtered_ec.flatten())
        rmv_idxs = np.unique(np.concatenate(rmv_idxs))
        filtered_X = np.delete(filtered_X,rmv_idxs,axis=1)
        Xuse = filtered_X

    #Default Xuse
    elif filter_radius == False:
        if keep_targets == True:
            Xuse = filtered_X
        else:
            Xuse = np.delete(filtered_X,filtered_ec,axis=1)

    return Xuse,yuse,classes_to_keep,filtered_ec,final_mask

def cv_metric(Xuse,yuse,metric,folds=10,clf=XGBClassifier(n_estimators=500)):
    results = []
    cv = StratifiedKFold(n_splits=folds, shuffle=True)
    for train_idx, test_idx in cv.split(Xuse, yuse):
        X_train, X_test = Xuse[train_idx], Xuse[test_idx]
        y_train, y_test = yuse[train_idx], yuse[test_idx]
        model = clf
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if metric == 'f1':
            results.append(f1_score(y_test, y_pred, average=None))
        if metric =='accuracy':
            results.append(accuracy_score(y_test,y_pred))
    return np.array(results)

def shap_pruning(psd, cls, clf=XGBClassifier(n_estimators=500), show_progress=True, keep_targets=False):
    import time
    start_time = time.time()
    
    # Keep track of original indices
    og_idx = np.arange(get_ml_data(psd, keep_targets=True)[0].shape[1])
    filtered_ec = get_ml_data(psd)[3]
    if keep_targets == False:
        og_idx = np.delete(og_idx, filtered_ec)
    
    # Load variables
    Xuse, yuse = get_ml_data(psd, keep_targets=keep_targets)[0:2]
    
    f1 = []
    best_predictors = []
    shap_values_list = []
    og_idx_list = []
    og_idx_list.append(og_idx.copy())  # Store a copy of the initial indices

    total_features = Xuse.shape[1]
    features_removed = 0
    
    while Xuse.shape[1] > 0:
        # Fit, predict, score
        X_train, X_test, y_train, y_test = train_test_split(Xuse, yuse, test_size=0.2, stratify=yuse)
        model = clf
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1.append(f1_score(y_test, y_pred, average=None)[cls])
        
        # Calculate shap values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(Xuse).values[yuse==cls, :, cls]
        shap_values_list.append(shap_values)
        
        #Remove best feature
        best_feature = np.argmax(shap_values.mean(axis=0))
        best_predictors.append(og_idx[best_feature])
        Xuse = np.delete(Xuse, best_feature, axis=1)
        og_idx = np.delete(og_idx, best_feature)
        og_idx_list.append(og_idx)
        
        features_removed += 1
        
        # Report progress if flag is set
        if show_progress:
            percent_complete = (features_removed / total_features) * 100
            elapsed_time = time.time() - start_time
            print(f"Progress: {percent_complete:.1f}% complete | Features processed: {features_removed}/{total_features} | Time elapsed: {elapsed_time:.2f}s")
    
    total_time = time.time() - start_time
    if show_progress:
        print(f"Completed in {total_time:.2f} seconds")
    
    return np.array(f1), np.array(best_predictors)

def get_bg_spks(psd,ec=None,match_dims=False,min_count=50,encode_labels=True):
    #Generate multilabel_arr
    with redirect_stdout(io.StringIO()):
        psd.get_trials_raster()
    raster = psd.trials_raster[12:-6, :, :]
    cells = raster.shape[1]
    trials = raster.shape[2]
    bins = 6
    
    # Calculate how many complete bin groups we can fit
    time_bins = raster.shape[0]
    complete_bin_groups = time_bins // bins

    # Reshape using the calculated number of complete bin groups
    raster = raster[:complete_bin_groups*bins, :, :]  # Trim to fit complete bins
    raster = raster.reshape(complete_bin_groups, bins, cells, trials).mean(axis=1).transpose(1, 0, 2).reshape(cells, -1).T

    ec_spks = raster[:, psd.ensemble_cells.flatten()]
    multilabel_arr = (ec_spks > 0).astype(int)

    unique_rows, counts = np.unique(multilabel_arr, axis=0, return_counts=True)
    row_sums = np.sum(unique_rows, axis=1)
    single_label_mask = row_sums == 1
    count_mask = counts > min_count
    combined_mask = single_label_mask & count_mask
    selected_patterns = unique_rows[combined_mask]
    
    if selected_patterns.shape[0] == 0:
        return np.array([]), np.array([])
    indices = []
    for pattern in selected_patterns:
        matches = np.all(multilabel_arr == pattern, axis=1)
        pattern_indices = np.where(matches)[0]
        indices.extend(pattern_indices)
    
    #Create the array of selected rows
    selected_rows = multilabel_arr[indices]

    #Create label vector where each element is the position of the "1" in each row
    labels = np.argmax(selected_rows, axis=1)
    spks = raster[indices,psd.ensemble_cells[np.unique(labels)]].T
    if match_dims == True:
        Xuse = np.delete(raster[indices],ec,axis=1)
    else:
        Xuse = np.delete(raster[indices],psd.ensemble_cells[np.unique(labels)],axis=1)

    if encode_labels == True:
        yuse = LabelEncoder().fit_transform(labels)
    elif encode_labels == False:
        yuse = labels
    
    return Xuse,yuse,spks,labels