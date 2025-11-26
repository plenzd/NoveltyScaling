#!/usr/bin/env python

from sputils import *
from pstimutils import *
from runavalutils import evaluate_prediction_noob
from mlutils import Klasifajer, split_train_test
from sklearn.preprocessing import LabelEncoder

nargs=len(sys.argv)

if nargs<2:
   print('Usage: '+sys.argv[0]+' datafile')
   sys.exit(0)

prmsd={'name': 'defaultrun', 'verbose':0, 'data': '', 'exclude': 2, 'adtype': 'spk', 'ttfrac': 0.8, 'noob': 4, 'model': 'all', 'fwinst': 0, 'fwinend': 6, 'fwindow': '0-6', 'naver': 6, 'atro': 1, 'trext': 0, 'maxfwd': 50, 'qtest': 0, 'nshuf': 0, 'grpid': 0, 'skpstr': -1, 'ncmin': 3, 'filt': 0}

def my_get_filtered_masks(psd, threshold=0.5, reverse=False, return_raw=False):
    X = psd.Xdata
    y = psd.ydata
    ec = psd.ensemble_cells

    print("reverse=", reverse)
    if reverse: mask = np.zeros(len(y), dtype=bool)
    else: mask = np.ones(len(y), dtype=bool)
    for i, label in enumerate(y):
        col_idx = ec[label]  
        row_idx = i
        if X[row_idx, col_idx] == 0:
                if reverse: mask[i] = True
                else: mask[i] = False
    
    unique_classes, original_counts = np.unique(y, return_counts=True)
    original_counts_dict = dict(zip(unique_classes, original_counts))
    filtered_y = y[mask]
    remaining_counts = {cls: np.sum(filtered_y == cls) for cls in unique_classes}
    if threshold>1:
       print("Using number minimum")
       number_remaining = {
          cls: remaining_counts[cls]
          for cls in unique_classes
       }
       classes_to_keep = {cls for cls, num in number_remaining.items() if num >= threshold}
    else:
       print("Using fraction minimum")
       fraction_remaining = {
          cls: (remaining_counts[cls] / original_counts_dict[cls])
          for cls in unique_classes
       }
       classes_to_keep = {cls for cls, pct in fraction_remaining.items() if pct >= threshold}
    classes_to_keep = np.array(list(classes_to_keep))
    final_mask = np.array([cls in classes_to_keep for cls in y]) & mask
    if return_raw:
       return final_mask, classes_to_keep, mask
    else:
       return final_mask, classes_to_keep

def get_targ_act(psd, trial_mask):
    X = psd.Xdata
    y = psd.ydata
    ecs = psd.ensemble_cells
    totact=np.zeros(len(ecs))
    counts=np.zeros(len(ecs), 'int')
    activities=[[] for _ in range(len(ecs))]
    for i, label in enumerate(y):
       if trial_mask[i]:
        col_idxs = ecs[label]
        assert len(col_idxs)==1
        col_idx=col_idxs[0]
        row_idx = i
        counts[label] += 1
        value=X[row_idx, col_idx]
        totact[label] += value
        activities[label].append(X[row_idx, col_idx])
    return counts, totact, activities

def apply_trial_filter(X, y, ec, filter_mask):
    ec_idx = np.unique(y[filter_mask])
    nclasses=len(ec_idx)
    filtered_ec = ec[ec_idx]
    yuse = LabelEncoder().fit_transform(y[filter_mask])
    Xuse = X[filter_mask]
    return Xuse, yuse, nclasses, filtered_ec
 
def filter_classes_single_trial(X, y, ec):
    iclasses=np.unique(y)
    filter_mask=np.ones(len(y), 'bool')
    for icl in iclasses:
       tticl= (y==icl)
       if tticl.sum()==1:
          filter_mask[tticl]=False
    return apply_trial_filter(X, y, ec, filter_mask)

def true_raiyyan_get_ml_data(psd,threshold=0.5,keep_targets=False,filter_radius=False,radius_threshold=15):
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
    yuse = LabelEncoder().fit_transform(y[final_mask])
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

 
def filter_ml_data_trials(psd, threshold=0.5,keep_targets=False,filter_radius=False,radius_threshold=15):
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
    fraction_remaining = {
        cls: (remaining_counts[cls] / original_counts_dict[cls])
        for cls in unique_classes
    }
    classes_to_keep = {cls for cls, pct in fraction_remaining.items() if pct >= threshold}
    classes_to_keep = np.array(list(classes_to_keep))
    final_mask = np.array([cls in classes_to_keep for cls in y]) & mask

    ec_idx = np.unique(y[final_mask])
    filtered_ec = ec[ec_idx]
    yuse = LabelEncoder().fit_transform(y[final_mask])
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

oargs=parse_arguments_dict(prmsd, sys.argv)

datafile = prmsd["data"]
exclude = prmsd["exclude"]
adtype = prmsd["adtype"]
ttfrac = prmsd["ttfrac"]
noob = prmsd["noob"]
cmodel = prmsd["model"]
fwinst = prmsd["fwinst"]
fwinend = prmsd["fwinend"]
fwindow = prmsd["fwindow"]
qtest = prmsd['qtest']
nshuf = prmsd['nshuf']
grpid = prmsd['grpid']
skpstr = prmsd['skpstr']
trext = prmsd['trext']
maxfwd = prmsd['maxfwd']
naver = prmsd["naver"]
atro = prmsd["atro"]
ncmin = prmsd['ncmin']
filt = prmsd['filt']

use_trial_filtering=filt

if skpstr<0:
   skpstr=naver

if '-' in fwindow:
   fwinst, fwinend=map(int, fwindow.split('-'))
   fwindow=[fwinst, fwinend]

nargs=len(oargs)    
iarg=1
if nargs>iarg: datafile=oargs[iarg]
iarg+=1

modelprmdict={'rf': {'mxf': None, 'mxdepth': None}, 'tf': {'hiddenls': [100, 50]}, 'xg' : {'mxf': None, 'mxdepth': None}}
ddd="""
if nargs>iarg: exclude=int(sys.argv[iarg])
iarg+=1
if nargs>iarg: noob=int(sys.argv[iarg])
iarg+=1
if nargs>iarg: ttfrac=float(sys.argv[iarg])
iarg+=1
if nargs>iarg: cmodel=sys.argv[iarg]
iarg+=1
"""
random_train=True
#perdat=PertStimDataSimple(datafile, 'all', exclude=exclude, fwindow=fwindow)
#nclasses=perdat.nclasses
#keyboard('after loading')

adtypeslist=['spk', 'prob', 'fluo']
modelslist=['rfp1', 'rfp5', 'rf1', 'rf2', 'rf5'] + ['xgp1', 'xgp5', 'xg1', 'xg2', 'xg5']

modelslist=['rfp5', 'rf1', 'rf2'] + ['xgp5', 'xg1', 'xg2']

if qtest:
   modelslist=['rfp1', 'xgp1']
   noob=2

if cmodel:
   if cmodel=='all':
      smodels=modelslist
   else:
      smodels=[cmodel]
else:
  smodels=modelslist
  
if adtype:
   if adtype=='all':
      adtypes=adtypeslist
   else:
      adtypes=[adtype]
else:
  adtype='spk'

nadts=len(adtypes)
nmodels=len(smodels)

troffset=-naver-atro
if skpstr>0: skpstride=skpstr
else: skpstride=naver

print("atro=", atro)
print("troffset=", troffset)
print("naver=", naver)
print("skpstride=", skpstride)

if trext: rollstr='a%do%de%ds%d' % (naver, troffset, trext, skpstride)
else: rollstr='a%do%ds%d' % (naver, troffset, skpstride)

if True:
  perdat=PertStimDataSimple(datafile, troffset=troffset, trextend=trext, adtype=adtype, exclude=exclude, fwindow=fwindow, keep_xdata=True)
  ndur, ncells, ntrials = perdat.trials_raster.shape
  maxdurindx=min([maxfwd-troffset, ndur])
#  maxoffset=maxdurindx-naver-troffset WRONG troffset already include in ndur
  maxoffset=maxdurindx-naver
  
  print("nduration=", ndur)
  print("ncells=", ncells)
  print("ntrials=", ntrials)
  print("maxdurindx=", maxdurindx)
  print("maxoffset=", maxoffset)
  print("Max offset into trials_raster maxoffset=", maxoffset)
  
  ntargets=perdat.pdd['ntargets']
  print("ntargets=", ntargets)
  if ntargets<1: nclasses=perdat.nclasses
  else: nclasses=ntargets
  chancerr=100./nclasses
  print("nclasses=", nclasses)
  
# define the results file and check if it exists
  datastr=adtype                                                            
  if nmodels==1: cmodelstr=cmodel
  else: cmodelstr='nmods%d' % nmodels

  if use_trial_filtering:
     if use_trial_filtering<0:
        savfileprefix='_revfilt%g_vstime' % ncmin
     else:
        savfileprefix='_filt_vstime'
  else: savfileprefix='_vstime'

  replstr='%s_%s_%s_modl-%s_nclass%d_noob%d_tt%d.npy' % (savfileprefix, rollstr, datastr, cmodelstr, ntargets, noob,int(np.round(100*ttfrac)))     
  if '.mat' in datafile[-6:]: savfile='vstresults/' + os.path.basename(datafile.replace('.mat', replstr))
  elif '.npy.gz' in datafile[-9:]: savfile='vstresults/' + os.path.basename(datafile.replace('.npy.gz', replstr))

  if use_trial_filtering:
     datamarker='BigDataSetA_'
     perdat.get_ml_data_tr(nshuffles=nshuf, adtype=adtype, grpid=grpid, aoffset=-troffset, arange=6, astride=None, ntimef=1, omitstims=[], keep_xdata=True)
     indxmark=datafile.find(datamarker)
     iend=datafile.find('.npy')
     if iend<0 or indxmark<0 or indxmark>iend:
        raise ValueError("File needs to have .npy for filtering to work!!")
     datanumname=datafile[indxmark+len(datamarker):iend]
     if len(datanumname)==1: datanumname='0'+datanumname
#     if len(datanumname)==3: datanumname=datanumname[1:]
     print("datanumname=", datanumname)
     masks_dict=np.load('%strial_filter_mask.npy' % datamarker, allow_pickle=True).tolist()
     filter_mask=masks_dict[datanumname]
     filter_mask_orig=filter_mask.copy()
     myfwdmask, kept_classes_fwd =my_get_filtered_masks(perdat, threshold=0.5, reverse=False, return_raw=False)
     if False:
        XXt1,yyt1,ckeep1,ect1,ryyanfwdmask1 = filter_ml_data_trials(perdat, threshold=0.5, keep_targets=False,filter_radius=False, radius_threshold=15)
        XXt,yyt,ckeep,ect,ryyanfwdmask = true_raiyyan_get_ml_data(perdat, threshold=0.5, keep_targets=False,filter_radius=False, radius_threshold=15)
        cmp1alleq=all(myfwdmask==ryyanfwdmask)
        print("cmp1alleq=", cmp1alleq)
        cmp1alleq=all(ryyanfwdmask1==ryyanfwdmask)
        print("raiyyan masks cmp1alleq=", cmp1alleq)
     myrevmask, kept_classes_rev=my_get_filtered_masks(perdat, threshold=(ncmin-0.1), reverse=True, return_raw=False)
     alleq=all(myfwdmask==filter_mask_orig)
     print("ARE THEY EQUAL?")
     if alleq:
        print("YES, they are!")
     else:
        print("NO! They are NOT!")
        print("get_targ_act(perdat, myfwdmask)")
        print("get_targ_act(perdat, filter_mask_orig)")
        print("****Somehow they are not matching now when skip is 1\n"*10)
#        keyboard('check myfwdmask == filter_mask_orig')

     if False: # Analyze excluded classes
        print("For Data A%s ncmin=%d there are %d target cells kept" % (datanumname, ncmin, len(kept_classes_rev)))
        print("perdat.ensemble_cells=", perdat.ensemble_cells[:,0])
#        Xuse, yuse, nclasses_new, ec_new= apply_trial_filter(perdat.Xuse, perdat.yuse, perdat.ensemble_cells, myrevmask)
        print("kept_classes_fwd=", kept_classes_fwd)
        print("kept_classes_rev=", kept_classes_rev)
        a1,b1,c1=get_targ_act(perdat, myrevmask)
        print("Overall kept=", a1)
        print("Overall activity=", b1)
        print("Cell indices of the kept target cells:", perdat.ensemble_cells[kept_classes_rev,0])
        print("Number of trials kept for target cells]=", a1[kept_classes_rev])
        print("Total activity of the target for kept trials=", b1[kept_classes_rev])
        sys.exit(0)
        a2,b2,c2=get_targ_act(perdat, myfwdmask)
        print("a2=", a2)
        print("b2=", b2)
        print("a2[kept_classes_fwd]=", a2[kept_classes_fwd])
        print("b2[kept_classes_fwd]=", b2[kept_classes_fwd])
        a1b,b1b,c1b=get_targ_act(perdat, np.logical_not(myfwdmask))
        print("a1b=", a1b)
        print("b1b=", b1b)
        print("a1b[kept_classes_rev]=", a1b[kept_classes_rev])
        print("b1b[kept_classes_rev]=", b1b[kept_classes_rev])
        keyboard("a,b,c=get_targ_act(perdat, varmasks)")

     print("Before filter_mask.sum=", filter_mask.sum())
     if use_trial_filtering<0:
#        filter_mask=np.logical_not(filter_mask_orig)
        filter_mask=myrevmask
        print("After filter_mask.sum=", filter_mask.sum())
  
  if os.path.exists(savfile):
     print("Already have the results for this file:", savfile)
     sys.exit(0)
   
  timeoffsets=list(range(0, maxoffset, skpstride))
  ntimes=len(timeoffsets)
  paccuracys=np.zeros((noob, ntimes, nmodels))
  ntoffsets=len(timeoffsets)
  print("Processing timeoffsets=")
  print(timeoffsets)
  
  for itoffs,toffset in enumerate(timeoffsets):
    print("\n\n\n****************************\n************************ Processing time toffset=%d (%d of %d)" % (toffset, itoffs+1, ntoffsets))
    if nshuf:
        print("Using get_shuffled")
        perdat.get_ml_data_tr(nshuffles=nshuf, adtype=adtype, grpid=grpid, aoffset=toffset, arange=naver, astride=None, ntimef=1, omitstims=[])
    else:
        print("Using get_ml_data_tr")
        perdat.get_ml_data_tr(nshuffles=0, adtype=adtype, grpid=grpid, aoffset=toffset, arange=naver, astride=None, ntimef=1, omitstims=[])
#        keyboard('check Xuse with ml_data_tr')
#        perdat.get_ml_data(adtype=adtype, grpid=grpid, fwindow=fwindow, omitstims=[])

    
    if use_trial_filtering:
          print("nclasses=", nclasses)
          Xdata_orig=perdat.Xdata
          ydata_orig=perdat.ydata
          ec_orig=perdat.ensemble_cells
          Xuse, yuse, nclasses_new, ec_new= apply_trial_filter(perdat.Xuse, perdat.yuse, perdat.ensemble_cells, filter_mask)
          if itoffs==0:
            print("ec_new=", ec_new)
            print("Before 1-class filtering:\n tally(yuse)=", tally(yuse))
            print("nclasses_new1=", nclasses_new)
            tkt1=tally(yuse)
            for tk in tkt1: print("%d -> %d" % (tk, tkt1[tk]))
            if False:
                Xuse, yuse, nclasses_new, ec_new= filter_classes_single_trial(Xuse, yuse, ec_new)
                print("ec_new=", ec_new)
                print("After 1-class filtering:\n tally(yuse)=", tally(yuse))
                print("nclasses_new2=", nclasses_new)
                tkt2=tally(yuse)
                for tk in tkt2: print("%d -> %d" % (tk, tkt2[tk]))
          
#       Xuse,yuse,classes_to_keep,filtered_ec,final_mask = filter_ml_data_trials(perdat, threshold=0.5, keep_targets=False,filter_radius=False,radius_threshold=15)
    else:
          Xuse=perdat.Xuse
          yuse=perdat.yuse


#    if True: # for icm, cmodel in enumerate(models):
    for icm, cmodel in enumerate(smodels):
      modelprms=modelprmdict[cmodel[:2]]
      print("********* Processing model %s" % (cmodel))
      for ioob in range(noob):
#        try:
        if 1:
          print("ioob=", ioob+1)
          Xtrain, ytrain, Xtest, ytest = split_train_test(Xuse, yuse, ttfrac=ttfrac, random_train=random_train)
          klsb=Klasifajer(cmodel, modelprms, data=Xuse, labels=yuse, nclasses=nclasses)
          klsb.fit(Xtrain,ytrain)
          if use_trial_filtering: nclasses=nclasses_new
          else: nclasses=perdat.nclasses
          chancerr1=100./nclasses
          if not np.allclose(chancerr, chancerr1):
             print("WARNING: chancerr CHANGED:", chancerr, chancerr1)
             chancerr=chancerr1
          if ioob==0: print("tally(ytrain)=", tally(ytrain))
          paccuracy=klsb.evaluate(Xtest, ytest)
          print("%s ioob=%d paccuracy=%g (chance %g)" % (adtype, ioob, paccuracy, chancerr))
          paccuracys[ioob,itoffs, icm]=paccuracy
#        except:
        else:
          print("FAILED to process %s ioob=%d" % (adtype, ioob))
          keyboard('check why it failed')
          paccuracys[ioob, itoffs, icm]=-1.
  sys.stdout.flush()
  
  print("datafile=", datafile)
  print("Saving results into save file=", savfile)
  np.save(savfile, np.array([paccuracys, [timeoffsets, naver, troffset, chancerr, ttfrac, nclasses], smodels, adtypes], dtype=object))
