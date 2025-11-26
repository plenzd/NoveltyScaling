import os, sys, time, re, gzip
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy.random as npr
# Uncomment this if you need tensorflow
#import tensorflow as tf
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import confusion_matrix as confmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import h5py

__version__="1.0.1"
padd_unfinished=True
use_sparse_tr=False

try:
  from asputils import file2dict, keyboard, tally, myloadmat, mpcolors
  from amlutils import Klasifajer, split_train_test, get_train_test_indices
except:
  from sklearn.model_selection import train_test_split as split_train_test
  print('Unable to load sputils and/or mlutils!!! Maybe split_train_test is not working the same as train_test_split in sklearn!?')
  def tally(alist, nout=1, sortby=0):
   ttuples=list(zip(*np.unique(alist,return_counts=True)))
   if nout==1:
      return dict(ttuples)
   elif nout==2:
      ttuples.sort(key=lambda x: x[sortby], reverse=True)
      return list(zip(*ttuples))

verbose=0
# This is for VisStimData ... 
important_vars_vstim=['raster_data', 'psarray', 'istarr', 'nclasstrials', 'rftimes', 'rftrid', 'rflabels']
raster_datavars_vstim = ['rfdata', 'istim', 'jstim', 'stimlabels',  'frate', 'ps_dtlen'] 
optraster_datavars_vstim = ['rftimes', 'rftrid', 'rflabels', 'ps_dtlen_max', 'ps_dtlen_min']

# This is for class PertStimDataSimple
helpstr="""
pdd[adtype][grp_index] # contains different rasters (nbins, ncells)
# important_vars_pertstim=['trials_raster', 'trial_stim_inds', 'orig_classlabels', 'cellcoords', 'icells_stim', 'icells_nonstim', 'target_cells', 
    trials_raster = (duration, cells, ntrials)
    self.icells_stim=istimcells # list of stimulated cell indices
    self.n_stimcells=len(istimcells)
    self.cells_select=np.ones(ncells).astype(bool)
    self.cells_select[istimcells]=False
    self.icells_nonstim=np.where(self.cells_select)[0]
    self.n_nonstimcells=len(self.icells_nonstim)
"""
# [datafilename    kvalue   kmax   coordinate_file
gavdatainfo= {
      'vis1': ['data1/av_3143_1_z2_k26.mat', 26, 7, ''],
      'vis2': ['data1/av_3143_2_z2_k12.mat', 12, 7, ''],
      'vis3': ['data1/av_6742_z2_k16.mat', 16, 5, 'data1/6742_coords.mat'],
      'vis4': ['data1/av_3143_1_dn_z2_k17.mat', 17, 12, ''],
      'vis5': ['data1/av_3143_2_dn_z2_k9.mat', 9, 4, ''],
      'vis6' : ['data1/av_6742_dn_z2_k4.mat', 4, 7, 'data1/6742_denoised_coords.mat'],
      'visB1' : ['data1/68490_21.11.23_VisOnly25x_synecontrast.mat', 0, 0, ''],
      'visC1' : [None, 1, 28, 'data1/3143_1_coords.mat'],
      'visC2' : [None, 2, 9, 'data1/3143_2_coords.mat'],
      'default' : [None, -1, -1, 'local'],
}

def find_optimal_interstimultus_duration(idurations, messg='', verbose=1):
    idurations[idurations<2]=2**30
    ndurs=len(idurations)
    duration=min(idurations)
    if verbose>1 and messg: print(messg)
    if verbose>1: print("    Shortest inter-stimulation duration>0=%d" % duration)
    cmode=sp.stats.mode(idurations)
    if verbose>1: print("    durations mode=%d" % cmode.mode)
    if verbose>3:
      tdurs=tally(idurations)
      for td in tdurs:
        print("duration %d count %d" % (td, tdurs[td]))
                   
    if cmode.mode != duration:
      neqdurs=(idurations==duration).sum()
      if verbose>0: 
        print("   Shortest duration %d is not the mode=%d!" % (duration, cmode.mode))
        if verbose>1: print("   shortest duration count = %d vs mode count=%d" % (neqdurs, cmode.count))
      uu,cc=np.unique(idurations, return_counts=True)
      duration=cmode.mode
      for u1,c1 in zip(uu,cc):
        if verbose>1: print('    Duration %g count %d' % (u1, c1))
        if ( ((c1>10) or (c1>(ndurs//10))) and (u1<duration) and abs(u1-duration)<3): duration=u1
      if verbose>1: print("    Setting duration to ", duration)
      if cmode.count/neqdurs > 1.1:
        if abs(cmode.mode-duration)>1:
          print("Something is wrong with this stimulation raster!!")
          keyboard('Check what is wrong!')
    return duration

  
def raster_dict_to_sparse(mydict, shape=None, csc=True, dtype=None, switch_ij=False):
  
    data = mydict['data']
    indices = mydict['ir']
    indptr = mydict['jc']
    
    if False:
      nirs=mydict['ir'].size
      njcs=mydict['jc'].size
      if nirs>njcs:
        inds='ir'
        iptrs='jc'
      else:
        inds='jc'
        iptrs='ir'
      if switch_ij:
        tmp=inds
        inds=iptrs
        iptrs=tmp
      indices = mydict[inds]
      indptr = mydict[iptrs]
  
    if dtype is None: dtype=data.dtype
    else: data=data.astype(dtype)

    if shape is None:
        n_rows = int(indices.max()) + 1
        n_cols = len(indptr) - 1
        shape = (n_rows, n_cols)
        
    cscmat=sp.sparse.csc_matrix((data, indices, indptr), shape=shape)

    if csc: return cscmat
    else: return cscmat.tocsr()

def getcongmat2d(params):
  th, ss, tx, ty = params
  c=ss*(tx*np.cos(th) - ty*np.sin(th))
  f=ss*(ty*np.cos(th) + tx*np.sin(th))
  congmat=np.array([[ss*np.cos(th),-ss*np.sin(th), c], [ ss*np.sin(th), ss*np.cos(th), f], [0., 0., 1.]])
  return congmat

def homography_func(pts, params):
    hh = getcongmat2d(params)
    opts=np.c_[pts,np.ones(pts.shape[0])]
    return (hh @ opts.T)[:2,:].T

def residuals_function(params, pts1, pts2):
  return (homography_func(pts1, params) - pts2).ravel()
    
def find_best_congtransf(pts1, pts2, pguess=np.array([0., 1, 0, 0])):
  result = least_squares(residuals_function, pguess, args=(pts1, pts2))
  return result.x

def test_congmat2d(npts=10, nn=50, dth=0.03, txy=0.5, sc=1., mx=20, ptime=0.5):
  pts=rand(npts,2)*mx
  opts=np.c_[pts,np.ones(npts)]
  for ii in range(nn):
    hh=getcongmat2d([dth*ii,sc,0.,0.])
    npts=(hh @ opts.T)[:2,:].T
    hh=getcongmat2d([dth*ii,sc,txy, txy])
    npts2=(hh @ opts.T)[:2,:].T
    plt.plot(pts[:,0],pts[:,1],'ro')
    plt.plot(npts[:,0],npts[:,1],'go')
    plt.plot(npts2[:,0],npts2[:,1],'bo');
    plt.pause(ptime)
    xylm=np.array([-(mx+np.abs(txy)), mx+np.abs(txy)])*1.4
    plt.xlim(xylm)
    plt.ylim(xylm)
    plt.gca().set_aspect('equal', adjustable='box')
  show()

def get_tiago_shuffled_sample(iPAtrialoffs, iPAsindx, iXsrt, nstimclasses=8, ntperclass=20, nshuffles=1, ssimg=None):
  allcinds=[]
  if ssimg is not None: shuffimgs=[]
#  print("nshuffles=", nshuffles)
  
  for ishufs in range(nshuffles):
    cinds=[]
    if ssimg is not None: shufimg=np.zeros_like(ssimg)
    for icls in range(nstimclasses):
      rbtrls=np.random.permutation(ntperclass)
      for itrl in range(ntperclass):
        iimg=ntperclass*icls+itrl
        iPAlocal=np.where(ssimg[iimg,:]==1)[0]                              
        ipas=iPAtrialoffs[icls][itrl]
        if ipas.size:
          itrnd=rbtrls[itrl]
          irimg=iPAsindx[icls, itrnd]
          if ssimg is not None:
            shufimg[irimg,ipas]=4
          try:
            cinds.extend(iXsrt[irimg,ipas])
          except:
            print('cinds.extend(iPAsindx[irimg,ipas])')
            keyboard('failed cinds')
            
    allcinds.append(cinds)
        
    if ssimg is not None: shuffimgs.append(shufimg)

  if ssimg is None: return allcinds
  else: return allcinds, shuffimgs
   
def get_fa_consecutive_sample(iPAtrialoffs, iPAsindx, iXsrt, nstimclasses=8, ntperclass=20, nshuffles=1, ssimg=None):
  allcinds=[]
  if ssimg is not None: shuffimgs=[]
#  print("nshuffles=", nshuffles)
  keyboard('check #1: tiago nshuffles')
  
  for ishufs in range(nshuffles):
    cinds=[]
    if ssimg is not None: shufimg=np.zeros_like(ssimg)
    for icls in range(nstimclasses):
      rbtrls=np.random.permutation(ntperclass)
      for itrl in range(ntperclass):
        iimg=ntperclass*icls+itrl
        iPAlocal=np.where(ssimg[iimg,:]==1)[0]                              
        ipas=iPAtrialoffs[icls][itrl]
        if ipas.size:
          itrnd=rbtrls[itrl]
          irimg=iPAsindx[icls, itrnd]
          if ssimg is not None:
            shufimg[irimg,ipas]=4
          try:
            cinds.extend(iXsrt[irimg,ipas])
          except:
            print('cinds.extend(iPAsindx[irimg,ipas])')
            keyboard('failed cinds')
            
    allcinds.append(cinds)
        
    if ssimg is not None: shuffimgs.append(shufimg)

#  keyboard('check inside tiago at the very end')

  if ssimg is None: return allcinds
  else: return allcinds, shuffimgs

def get_matched_indices(yaval, yparabtally, nstimclasses=8, trids=None, ntrials=160):
    cinds=[]
    cstatus=[]
    if trids is None:
      for icl in range(nstimclasses):
         nneeded=yparabtally[icl]
         icurr=np.where(yaval == icl)[0]
         ncurr=len(icurr)
         print("ncurr=", ncurr)
         print("nneeded=", nneeded)
         if ncurr<nneeded:
           cstatus.append(icl)
           irndcls=np.random.choice(ncurr, size=nneeded, replace=True)
         else:
           irndcls=np.random.choice(ncurr, size=nneeded, replace=False)
         cinds.extend(icurr[irndcls])
    else:
      for icl in range(nstimclasses):
         nneeded=yparabtally[icl]
         ifill=0
         for itrl in range(ntrials):
           itrc=np.where((yaval == icl) & (trids == itrl))[0]
           ntc=len(itrc)
           print("For icl=%d and itrl=%d found ntc=%d cases" % (icl, itrl, ntc))
           print("ifill=", ifill)
           nn1=nneeded-ifill
           print("nn1=", nn1)
           if ntc>=nn1:
             irndcls=np.random.choice(ntc, size=nn1, replace=False)
             cinds.extend(itrc[irndcls])
             print('FILLED THE TICKED FOR icl=', icl)
             if False: time.sleep(1)
             ifill=nneeded
             break
           else:
             cstatus.append(10*itrl+icl)
             print("ifill BEFORE=", ifill)
             print("ntc=", ntc)
             ifill+=ntc
             print("ifill AFTER=", ifill)
             cinds.extend(itrc)
         print("FINAL ifill=", ifill)
         if ifill<nneeded:
           nn1=nneeded-ifill
           icurr=np.where(yaval == icl)[0]
           ncurr=len(icurr)
           irndcls=np.random.choice(ncurr, size=nn1, replace=True)
           cinds.extend(icurr[irndcls])
           print('FAILED TO MATCH ALL SAMPLES')
           keyboard('Insided matching function')
    return cinds, cstatus

  
def relabel_classes(labls):
  uvals=np.unique(labls)
  nlabls=np.zeros_like(labls)
  for iuv, uv in enumerate(uvals):
    nlabls[labls==uv]=iuv
  return nlabls
  
def get_labels_tally(yaval, nstimclass=8):      
  ycurtally=tally(yaval)
  atally=np.zeros(nstimclass)
  for ii in range(nstimclasses):
    if ii in ycurtally: atally[ii] = ycurtally[ii]
    else: atally[ii]=0
  return atally

def data2mode(dataname):
  if dataname in ['vis1', 'vis2', 'vis3', 'vis4', 'vis5', 'vis6']:
    return 1
  elif dataname[:4] in ['visB']:
    return 2
  elif dataname[:4] in ['visC', 'visc']:
    return 3
  else:
    return 0

def get_angle_selectivity(psarray, ulabels=None, istim=None, angles=None):
  
#OLD:    ntbins,ncells,nstimperangle, Nangles=psarray.shape
    nstimperangle, Nangles, ntbins,ncells=psarray.shape
    print("psarray.shape=", psarray.shape)
    if angles is None:
       dangl=360./Nangles
       angles=dangl*np.arange(Nangles) # angles in degrees
    elif isinstance(angles,(int,np.integer)):
       dangl=360./angles
       angles=dangl*np.arange(angles) # angles in degrees
    nangles=len(angles)
    assert nangles==Nangles
# with original datarray: [nt, ncell, nreps, nclass]    Rangle=np.transpose(psarray, [0,2,1,3]).sum(0).sum(0) # Nrois, Nangles
#  needs: nt nreps ncell nclass  (psarray now using [nreps, nclass, nt ncell] )
    Rangle=np.transpose(psarray, [2,0,3,1]).sum(0).sum(0) # Nrois, Nangles
    Nframes_angle=np.zeros(Nangles)
    for i in range(Nangles):
      Nframes_angle[i]=nstimperangle*ntbins
      
    nROIs=ncells
    PD=np.zeros(nROIs)
    DSI=np.zeros(nROIs)
    OSI=np.zeros(nROIs)
    
    for iROI in range(nROIs):
        sumRangle, A, B, A2, B2 = 0., 0., 0., 0., 0.
        for i in range(Nangles):
          firing_rate = Rangle[iROI][i]/Nframes_angle[i]
          rad = (angles[i]*np.pi)/180.
          sumRangle += firing_rate
          A += firing_rate*np.cos(rad)
          B += firing_rate*np.sin(rad)
          A2 += firing_rate*np.cos(2*rad)
          
        if sumRangle>0.:
          if A < 0.: PD[iROI] = 180. + np.arctan(B/A)*180./np.pi
          elif B >= 0.: PD[iROI] = np.arctan(B/A)*180./np.pi
          else: PD[iROI] = 360. + np.arctan(B/A)*180./np.pi
          DSI[iROI] = np.sqrt(A*A + B*B)/sumRangle
          OSI[iROI] = np.sqrt(A2*A2 + B2*B2)/sumRangle
        else:
          PD[iROI] = -1.
          DSI[iROI] = 0.
          OSI[iROI] = 0.
          
    return DSI, OSI, PD

class AvalancheInfo:
  
  def __init__(self, avspec, kvalue=0, pacutoff=None, dd={}, dco=0, avdatainfo=gavdatainfo):

    self.avdatainfo=avdatainfo
    self.pacutoff=pacutoff
    self.avstarts, self.avdurs, self.avsizes, self.navals = None, None, None, None
    self.avshinfo=None
        
    if ( avspec is not None ) and (avspec in self.avdatainfo.keys()):
        self.avinfile = self.avdatainfo[avspec][0]
        if kvalue<=0: self.kvalue = self.avdatainfo[avspec][1]
        else: self.kvalue = kvalue
        if pacutoff is None: self.pacutoff = self.avdatainfo[avspec][2]
        else: self.pacutoff = pacutoff
        
        if self.avinfile is None:
          ikval=self.kvalue-1
          if 'av_start' in dd: self.avstarts=dd['av_start'][ikval,0].ravel()
          if 'durations' in dd: self.avdurs=dd['durations'][ikval,0].ravel()
          if 'sizes' in dd: self.avsizes=dd['sizes'][ikval,0].ravel()
        else:    
          avalinfo=myloadmat(self.avinfile)
          self.avstarts=avalinfo['av_start_prob'].ravel().astype('int')
          self.avdurs=avalinfo[ 'durations_prob'].ravel()
          self.avsizes=avalinfo[ 'sizes_prob'].ravel()
          self.navals=len(self.avdurs)

    if self.avstarts is None: return
    
    self.navals=len(self.avstarts)
    ishifdiff=np.diff(self.avstarts)
    ishifts=np.where(ishifdiff<0)[0]+1
    ishifts=np.insert(ishifts,0,0)
    self.nshifts=len(ishifts)
    assert self.nshifts == self.kvalue
    ishifts=np.append(ishifts,None)
    self.ishifts=ishifts
    self.avshinfo=[{} for _ in range(self.nshifts)]
    print("self.pacutoff=", self.pacutoff)

    for ishf in range(self.nshifts):
      print("Assigning different shuffles information ishf=", ishf)
      self.avshinfo[ishf]['avst']=self.avstarts[self.ishifts[ishf]:self.ishifts[ishf+1]]
      self.avshinfo[ishf]['avdur'] = curavdurs = self.avdurs[self.ishifts[ishf]:self.ishifts[ishf+1]]
      self.avshinfo[ishf]['ipa']=np.where(curavdurs<=(self.pacutoff+dco))[0]
      self.avshinfo[ishf]['ifa']=np.where(curavdurs>(self.pacutoff+dco))[0]
      
  def get_pafa_info(self, vsdclass, nshiftexplore=None):
      vsdclass.get_pafa_info(self, nshiftexplore=nshiftexplore)
      
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# ['spk', 'prob', 'fluo', 'cellcoords', 'frametimes', 'optostimtimes', 'StimOrder', 'target', 'target_ensemble'])
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# ['spk', 'prob', 'fluo', 'cellcoords', 'frametimes', 'optostimtimes', 'StimOrder', 'target', 'target_ensemble'])
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class PertStimDataSimple:
  
  def __init__(self, dataspec, adtype='spk', exclude=2, grpid=None, trwindow=None, troffset=0, trextend=0, fwindow=[], mode=0, nshuffles=0, xdinfo = {}, keep_xdata=False, verbose=1, **psdkwds):

    self.version=__version__
    self.xdinfo= xdinfo
    self.trial_select=None
    self.trials_raster=None
    self.trial_shuffled_raster=None
    self.verbose=verbose

    if isinstance(dataspec,str):
      if os.path.exists(dataspec):
        datafile=dataspec
#      elif 'BigDataSetA_' in dataspec:
      elif 'DataSet' in dataspec or 'SimData' in dataspec:
        if '_' in dataspec:
          items=dataspec.split('_')
          numstr=items[-1]
          fileroot='_'.join(items[:-1])
          print("fileroot=", fileroot)
          print("numstr=", numstr)
          datafile='%s_%s.npy.gz' % (fileroot, numstr)
        else:
          numstr=''
          datafile=dataspec
        if not os.path.exists(datafile):
          print("File specified with %s was not recognized! Diverting to alternate reading of .mat!" % dataspec)
          print("THIS TYPE OF READING A FILE IS NOT USED currently! Please make sure you specified the data file correctly!")
          sys.exit(0)
          convert_bigdata_to_npy_file(dataspec)
    elif isinstance(dataspec,list):
      datafile=dataspec[0]
      adtype=dataspec[1]

    if datafile[-4:]=='.mat':
      pdd=load_mat_ensemble_stim_data(datafile)
      print("CORRECT THIS AND ASSIGN PROPER ntargets in pdd for emx mice, i.e., HCS/HCS_ data")
      pdd['ntargets']=-1
    else:
      if ( 'DataSet' in datafile or 'SimData' in datafile ) and ( '.npy' in datafile):
        if self.verbose>-1: print("Loading datafile=", datafile)
        pdd=load_npy_pst_data_file_A(datafile)

    self.pdd=pdd
    if self.verbose>3: print("pdd['target'].shape=", pdd['target'].shape)
    if grpid is None:
      grp_index=pdd['grp1indx']
      grpid=0
    elif isinstance(grpid,(int,np.integer)):
      grp_index=pdd['grp1indx']+grpid

    self.default_grp_index = grp_index
    if self.verbose>0: print("Using grp_index=", grp_index)
    ngrps=len(pdd['spk'])
    
    if pdd['prob']:
      for pddg in pdd['prob']:
        nnans=np.isnan(pddg).sum()
        if nnans:
          if self.verbose>0: print("Data has nans! n_nans=%d" % nnans)
          np.nan_to_num(pddg, copy=False)
      assert ngrps == len(pdd['prob']) == len(pdd['fluo'])
    
    if grp_index>=ngrps:
      raise ValueError("Cannot use grp_index=%d! Total number of groups is %d" % (grp_index, ngrps))

    self.exclude=exclude # exclude level 2 both 0 and stim-cells are excluded; 1 - only 0s; 0 - nothing excluded
    
    # ['spk', 'prob', 'fluo', 'cellcoords', 'frametimes', 'optostimtimes', 'StimOrder', 'target', 'target_ensemble'])    rasterdatas={}
    if False:
      if adtype=='all':
        adtypes=['spk','prob', 'fluo']
        adt0=adtypes[0]
        rasterdatas[adt0]=pdd[adt0][grp_index]
        ntimes, ncells = rasterdatas[adt0].shape
        for adt in adtypes[1:]:
           rasterdatas[adt]=pdd[adt][grp_index]
           assert rasterdatas[adt].shape == (ntimes, ncells)
      else:
        rasterdatas[adtype]=pdd[adtype][grp_index]

    if isinstance(pdd['spk'][grp_index], dict):
       spmtx = raster_dict_to_sparse(pdd['spk'][grp_index], shape=None, csc=False, switch_ij=False, dtype='uint8').T
#       keyboard("After: spmtx = raster_dict_to_sparse(pdd['spk'][grp_index], shape=None, csc=False, switch_ij=False, dtype='uint8').T")
       if self.verbose>2: print("spmtx.indptr.size=", spmtx.indptr.size)
       pdd['spk'][grp_index] = spmtx
       pass      
    elif not isinstance(pdd['spk'][grp_index], np.ndarray):
       print("Unacceptable data type for pdd['spk'][grp_index]")
       sys.exit(0)
    
    ntimes, ncells = pdd['spk'][grp_index].shape
    self.ncells=ncells
    self.ntimes=ntimes
    
    # assign cells and their indices
    self.cellcoords = pdd['cellcoords'][grp_index]
    if self.verbose>3: print("pdd['target'].dtype=", pdd['target'].dtype)
    targettabI1=pdd['target'][:, grp_index].astype('int32')
    # .target_cells lists indices of the 32 target cells for that group
    self.target_cells=targettabI1-1
    self.target_cells[self.target_cells < 0]=-11111
    istimcells=targettabI1[targettabI1>0]-1 # list of stimulated cell indices
    target_ensembles=pdd['target_ensemble'][grp_index]-1
    self.ensemble_cells=targettabI1.reshape([-1, 2**grpid])-1
    
    self.icells_stim=istimcells # list of stimulated cell indices (icells_ are indices into raster data)
    self.n_stimcells=len(istimcells)
    self.cells_select=np.ones(ncells).astype(bool) # boolean array for cell selection
    self.cells_select[istimcells]=False
    self.icells_nonstim=np.where(self.cells_select)[0]
    self.n_nonstimcells=len(self.icells_nonstim)
    
    # assign frames and trial variables cells and their indices
    if 'frametimes' in pdd:
      self.frame_times=pdd['frametimes'][grp_index]
    else:
      print('Variable "frametimes" missing! Setting frames_times based on Nframes and framerate in pdd')
      self.frame_times=np.arange(pdd['Nframes'][grp_index])/pdd['framerate'][grp_index]
    opertimes=pdd['optostimtimes'][grp_index]
    nstims=len(opertimes)
    self.nstims=nstims
#    opindsI1=pdd['StimOrder'][grp_index]['Eind'][:nstims]
    opindsI1=pdd['StimInfo'][grp_index]['Eind'][:nstims]
    self.trial_stim_inds=opindsI1-1
    t0=time.time()
    # targettab is 1-indexed, from matlab
    ibins1, self.pstoffs = find_nearest_bin_indices(opertimes, self.frame_times, return_offsets=True)
    if verbose>3: print("Time for calculating lbins1=", time.time()-t0)
    if False: # compare with other methods
#     frmtimes=pdd['frametimes'][grp_index]
      frmtimes=self.frame_times
      t0=time.time()
      ibins2=find_nearest_bin_indices2(opertimes, frmtimes)
      print("Time for lbins2=", time.time()-t0)
      t0=time.time()
      ibins3=find_nearest_bin_indices_slow(opertimes, frmtimes)
      print("Time for lbins3=", time.time()-t0)
    # Assign trial frames

    tvalid=ibins1>=0
    if tvalid.sum()<len(ibins1):
      self.pstoffs=self.pstoffs[tvalid]
      ibins1=ibins1[tvalid]
      self.trial_frames=ibins1
      self.trial_stim_inds=self.trial_stim_inds[tvalid]
      
    idurations=np.diff(ibins1)
    if min(idurations)==0:
      print("WARNING: Repeated indices!!! Removing them!")
      print("opertimes[-10:]=", opertimes[-10:])
      frmtimes=self.frame_times
      keyboard('check opertimes and frmtimes! This shouldnt happend so check whats going on! Will use unique')
      ibins1=np.unique(ibins1)
      idurations=np.diff(ibins1)
    duration=find_optimal_interstimultus_duration(idurations, "*** Finding duration during initial trial times", verbose=self.verbose)

    if padd_unfinished:
      self.trial_frames=ibins1
    else:
      iibinlastgood = -1
    # frame indices occuring after the onset of a stimulus
      nrasterbins=pdd['spk'][grp_index].shape[0]
      while (ibins1[iibinlastgood]+duration+trextend) >= nrasterbins: iibinlastgood-=1
      print("iibinlastgood=", iibinlastgood)
      if iibinlastgood<-1:
          lastindx=(iibinlastgood+1)
          print("Omitting stimulations due to early ending raster! Last index=%d" % lastindx)
          self.trial_frames=ibins1[:lastindx]
          self.trial_stim_inds=self.trial_stim_inds[:lastindx]
      else:
          self.trial_frames=ibins1
  
    self.trial_cell_inds=targettabI1[self.trial_stim_inds]-1 # translation of stimulus indices into cell indices
    self.good_trial_select=self.trial_cell_inds>=0
    self.isi_min_duration=duration

# START collecting the data matrix(ices)
#    if len(fwindow) != 2: fwindow=[0, 6]
    if self.verbose>1: print("self.isi_min_duration=", self.isi_min_duration)
    self.get_trials_raster(adtype=adtype, trwindow=trwindow, troffset=troffset, trextend=trextend, grpid=grpid)
    
# if nshuffles # 
#  def get_shuffled_ml_data(self, nshuffles=1, ntimef=None, astride=None, arange=None, adtype='spk', fwindow=[0,6], grpid=None, omitstims=[], keep_xdata=False, precise_time=False, keep_raster=True):
#    self.get_ml_data_tr(nshuffles=nshuffles, adtype=adtype, grpid=grpid, fwindow=fwindow, omitstims=[], keep_xdata=keep_xdata)
    self.get_ml_data_tr(nshuffles=nshuffles, adtype=adtype, grpid=grpid, fwindow=fwindow, omitstims=[], keep_xdata=keep_xdata)

          
    if False:
        self.Xdatas={}
        self.Xuse_all={}
        print("stoffs=", stoffs)
        print("endoffs=", endoffs)
        for addt in adtypes:
           print("Procesing %s:" % addt)
           Xdata=np.zeros((nstims, ncells))
           rasterdata=rasterdatas[addt]
           for iprt,ibin in enumerate(self.trial_frames):
               sti=ibin+stoffs
               endi=ibin+endoffs
               Xdata[iprt,:]=rasterdata[sti:endi,:].sum(0)
        
           self.fcells=np.arange(ncells, dtype='int')
           if self.exclude==1:
              Xuse=Xdata[self.good_trial_select,:]
              youse=yodata[self.good_trial_select]
           elif self.exclude==2:
              self.fcells=self.inonstimcells.copy()
              Xuse=Xdata[:,tnonstim]
              Xuse=Xuse[self.good_trial_select,:]
              youse=yodata[self.good_trial_select]
           else:
              Xuse=Xdata
              youse=yodata
              
           print("Xdata.shape=", Xdata.shape)
           print("yodata.shape=", yodata.shape)
           print("Xuse.shape=", Xuse.shape)
           print("yuse.shape=", yuse.shape)
           self.Xdatas[addt]=Xdata
           self.Xuse_all[addt]=Xuse
           
        for adt in adtypes[1:]:
          assert self.Xuse_all[adt].shape == ( self.nsamples, self.nf )

  def get_psarrays(self, adtype='spk', fwindow=None, grpid=None, omitstims=[], nshuffles=0, keep_xdata=False):
    """ get post-stimulus array for Perturbation experiments """
    if not self.ncells: self.assign_basic_info(verbose=self.verbose)
    dtlen1=self.ps_dtlen
    if self.maxrep<1:
      print("self.maxrep=", self.maxrep)
      keyboard('sta to bi sa maxrep-om?')
    self.psarray=np.zeros((self.maxrep, self.nclasses, self.psduration, self.ncells))
    self.istarr=np.zeros((self.maxrep, self.nclasses), dtype=int)
    self.nclasstrials=np.zeros(self.nclasses, dtype=int)
    try:
      for ist,sclass in zip(self.istim,self.stimlabels):
#        print("ist=", ist, " sclass=", sclass)
        self.psarray[self.nclasstrials[sclass], sclass, :, :]=self.rfdata[ist:(ist+dtlen1),:]
        self.istarr[self.nclasstrials[sclass], sclass]=ist
        self.nclasstrials[sclass]+=1
    except:
      keyboard('failed psarray sh12')

  def assign_trial_select(self, omitstims=[], select_trials=None):
      if select_trials is None:
        if self.exclude>0: self.trial_select = self.good_trial_select
        else: self.trial_select = np.ones_like(self.good_trial_select).astype(bool)
      elif select_trials == 'good': self.trial_select = self.good_trial_select
      else:
        if isinstance(select_trials, np.ndarray) and select_trials.dtype == 'bool': self.trial_select = select_trials
        else: raise ValueError("select_trials must be 'good', boolean array")
      for omitstim in omitstims:
        self.trial_select=self.trial_select & (self.trial_stim_inds != omitstim)
    
  def get_ml_data(self, **kwargs):
      self.get_ml_data_tr(**kwargs)
    
  def get_ml_data_old(self, adtype='spk', fwindow=[0,6], grpid=None, omitstims=[], nshuffles=0, keep_xdata=False):
    
      self.ydata=self.trial_stim_inds

      if grpid is None:
        grp_index=self.default_grp_index
      elif isinstance(grpid,(int,np.integer)):
        grp_index=self.pdd['grp1indx']+grpid
        if grp_index != self.default_grp_index:
          print("WARNING: For now only the default grpid can be used, which was group id specified when loading the data! Nothing done!")
          return None

      self.assign_trial_select(omitstims)
      labels=self.ydata[self.trial_select]
      self.orig_classlabels, self.yuse = np.unique(labels, return_inverse=True)
      self.norigclasses=len(self.orig_classlabels)
#      keyboard('check labels')
      
      if len(fwindow)==2: stoffs, endoffs=fwindow
      else: stoffs, endoffs=0, 6
      
      if self.exclude>1:
        self.nf=self.n_nonstimcells
        assert self.nf == self.cells_select.sum()
      else: self.nf=self.ncells

      print("self.trial_select.sum()=", self.trial_select.sum())
# maybe do it here:
# self.frames_select=self.trial_stim_inds>0
# collect original one and then don't do this if it matches the original
      rasterdata=self.pdd[adtype][grp_index]
      if nshuffles and False: # Don't do it on raster
        self.shuffled_raster=True
        vstim=np.zeros(self.ntimes)
        vstim[self.trial_frames]=1
        rasterdata, shuff_matrix = trial_shuffle_rawdata(rasterdata, vstim, retmat=True)
        self.trial_shuffle_matrix=shuff_matrix
        print('trial_shuffle_rawdata(rfdata, vstim, stimlabels=None, retmat=False)')
        keyboard('test trial shuffle')
      else:
        self.shuffled_raster=False
      Xdata=np.zeros((self.nstims, self.ncells))
      print("After self.trial_select.sum()=", self.trial_select.sum())
      print("Xdata.shape=", Xdata.shape)
      for itrial,iframe in enumerate(self.trial_frames):
#        if self.frames_select[ifrm]:
          sti=iframe+stoffs
          endi=iframe+endoffs
#          Xdatalist.append(rasterdata[sti:endi,:].sum(0))
          Xdata[itrial,:]=rasterdata[sti:endi,:].sum(0)
        
      self.fcells=np.arange(self.ncells, dtype='int')
      self.Xuse=Xdata[self.trial_select,:]
      if self.exclude==2:
        self.Xuse=self.Xuse[:,self.cells_select]
        self.fcells=self.fcells[self.cells_select]

      self.nsamp, self.nf = self.Xuse.shape
      self.iclasses, self.class_counts= np.unique(self.yuse, return_counts=True)
      self.nclasses=len(self.iclasses)
      if nshuffles: self.do_trial_shuffle()

      print("self.Xuse.shape=", self.Xuse.shape)
      print("nsamp=%d nf=%d" % (self.nsamp, self.nf))
      if keep_xdata:
        self.Xdata = Xdata
        print('To match Xdata samples to Xuse samples you can use: sel_Xdata = self.Xdata[self.trial_select,:]')
#        self.sel_Xdata = Xdata[self.trial_select,:]
        
  def get_trials_raster(self, adtype='spk', trwindow=None, troffset=0, trextend=0, grpid=None, omitstims=[], nshuffles=0, select_trials=None, keep_xdata=True, precise_time=False):

      if self.verbose>-1: print("Getting trials raster!")
      self.ydata=self.trial_stim_inds
      self.troffset=troffset
      self.trextend=trextend
      # Don't allowing shuffling of the original trials raster anymore
      assert nshuffles==0


#      if grpid is None: grp_index=self.pdd['grp1indx']
      if grpid is None:
        grp_index=self.default_grp_index
      elif isinstance(grpid,(int,np.integer)):
        grp_index=self.pdd['grp1indx']+grpid
        if grp_index != self.default_grp_index:
          print("WARNING: For now only the default grpid can be used, which was group id specified when loading the data! Nothing done!")
          return None

      if precise_time:
        print("ERROR: Precise-time not implemented due to the problems of interpolating spike signals! Proceeding with discretized bins!")
        print("Setting precise_time to False")
        precise_time=False
        
      usetrsel=False # This is now permanently set to False
      if usetrsel:
        self.assign_trial_select(omitstims, select_trials=select_trials)
        print("self.trial_select=", self.trial_select)
      else:
        if select_trials is None: select_trials = np.ones_like(self.good_trial_select).astype(bool)  
        elif select_trials == 'good': select_trials = self.good_trial_select
        for omitstim in omitstims:
          select_trials= select_trials & (self.trial_stim_inds != omitstim)

      self.classlabels, self.orig_classcounts= np.unique(self.ydata, return_counts=True)
      self.maxrep=max(self.orig_classcounts)
      
      if usetrsel:
        labels=self.ydata[self.trial_select]
        self.select_trials=self.trial_select
      else:
        labels=self.ydata[select_trials]
        self.select_trials=select_trials
        
      self.orig_classlabels_raster, self.tr_iclass = np.unique(labels, return_inverse=True)
      self.tr_labels=labels
#     keyboard('check that labels == self.orig_classlabels_raster[self.tr_iclass]')
      self.norigclasses=len(self.orig_classlabels_raster)
      self.nclasses=self.norigclasses
      nclasses=self.norigclasses
      
      if self.exclude>1:
        self.nf=self.n_nonstimcells
        assert self.nf == self.cells_select.sum()
      else:
        self.nf=self.ncells
      
      rasterdata=self.pdd[adtype][grp_index]
      
      if nshuffles and False: # Don't do it on raster
        self.shuffled_raster=True
        vstim=np.zeros(self.ntimes)
        vstim[self.trial_frames]=1
        rasterdata, shuff_matrix = trial_shuffle_rawdata(rasterdata, vstim, retmat=True)
        self.trial_shuffle_matrix=shuff_matrix
        print('trial_shuffle_rawdata(rfdata, vstim, stimlabels=None, retmat=False)')
        keyboard('test trial shuffle')
      else:
        self.shuffled_raster=False

      if self.verbose>1: 
        if self.trial_select is None: print("trial_select is not assigned!!!")
        else: print("After self.trial_select.sum()=", self.trial_select.sum())
      
      ntrials=len(self.trial_frames)
      mintrdist=100000000
      actblocks=[]
      durations=[]
      assert ntrials==len(self.trial_frames)
      for itrial, iframe in enumerate(self.trial_frames):
#        if self.frames_select[ifrm]:
          rsti=iframe+self.troffset
          if  itrial==(ntrials-1):
              duration=find_optimal_interstimultus_duration(np.array(durations,int), "*** Finding duration while collecting blocks!", verbose=self.verbose)
              self.tr_duration=duration
              if duration != self.isi_min_duration and verbose>1:
                print('Duration of block is not the same as isi_min_duration! This is normal if troffset or trextend are not zero!')
              rendi=rsti+duration+self.trextend
              if rendi>rasterdata.shape[0]:
                rrblock=rasterdata[rsti:-1,:]
                if self.verbose>0:
                  print('WARNING: This trials starting at iframe=%d will be omitted or padded (see "padd_unfinished" flag), since it exceeds the end of raster!' % iframe)
                  if self.verbose>2: 
                    print("end index=%d and rasterdata.shape=" % rendi, rasterdata.shape)
                    print("Other blocks have shapes tally:", tally([a.shape for a in actblocks]))
                    print("Current block shape=", rrblock.shape)
                if padd_unfinished:
                  nrow1, ncol1 = rrblock.shape
                  nrowspadd=duration - nrow1
                  padding_rows = max(0, nrowspadd)
                  padded_rrblock=np.pad(rrblock, ((0, padding_rows), (0, 0)), 'constant')
                  print("WARNING: itrial=%d needed padding of %d rows" % (itrial, nrowspadd))
                  actblocks.append(padded_rrblock)
#                print("duration=", duration)
                print("padded_rrblock.shape=", padded_rrblock.shape)
#                keyboard('padded_rrblock')
#                keyboard('will be omitted since it exceeds the end of rasterdata!')
              else:
                rrblock=rasterdata[rsti:rendi,:]
                actblocks.append(rrblock)
          else:
            rendi=self.trial_frames[itrial+1]+self.trextend
            durations.append(rendi-rsti)
#          Xdatalist.append(rasterdata[sti:endi,:].sum(0))
            rrblock=rasterdata[rsti:rendi,:]
#          print("rrblock.shape=", rrblock.shape)
            actblocks.append(rrblock)

# removing bad trials
      self.ntrials=len(actblocks)
      if self.ntrials != ntrials:
        print("There were bad trials that will be eliminated! n_good_trials=%d n_orig_trials=%d" % (self.ntrials, ntrials))
        ntrials=self.ntrials
        self.trial_frames=self.trial_frames[:ntrials]
        self.trial_stim_inds=self.trial_stim_inds[:ntrials]
        self.good_trial_select=self.good_trial_select[:ntrials]
        self.ydata=self.trial_stim_inds
        select_trials=select_trials[:ntrials]
        self.select_trials=select_trials
        if self.trial_select is not None:
            keyboard('check self.trial_select since this was testing only when usetrsel was False!')

      if self.verbose>0: print("Default inter-stimulus duration=", duration)
      
      if trwindow is None:
        sti=0
        endi=self.tr_duration
        self.trwin_duration=self.tr_duration
      else:
        sti=trwindow[0]
        endi=trwindow[1]
        self.trwin_duration=endi-sti

      self.trwindow=[sti, endi]
      if self.verbose>3: 
        print("trials raster duration=", self.tr_duration)
        print("ntrials=", ntrials)
        print("sti=", sti)
        print("endi=", endi)
      if use_sparse_tr:
        self.trials_raster = [sp.sparse.lil_matrix((self.tr_duration, self.ncells), dtype='uint8') for _ in range(self.ntrials)]
      else:
        self.trials_raster=np.zeros((self.tr_duration, self.ncells, self.ntrials), 'uint8')
      self.class_trials=np.zeros((self.maxrep, self.nclasses), dtype=int)
      self.numclasstrials=np.zeros(self.nclasses, dtype=int)
#      try:
      if 1:
        mismatched_reports={}
        for itrial,blk in enumerate(actblocks):
          itarget=self.trial_stim_inds[itrial]
          if blk.shape[0] != self.tr_duration:
            mmkey=blk.shape
            if mmkey in mismatched_reports:
              mismatched_reports[mmkey].append(itrial)
            else:
              mismatched_reports[mmkey]=[itrial]
#            self.trials_raster[:, :, itrial]=blk[sti:endi,:]
          if use_sparse_tr:
            self.trials_raster[itrial][:,:]=blk[sti:endi,:]
          else:
            if (endi-sti)  != self.tr_duration:
              print("endi=", endi)
              print("sti=", sti)
              print("blk.shape=", blk.shape)
              print("self.tr_duration=", self.tr_duration)
              keyboard('check what is wrong on this one')
            
            if isinstance(blk, np.ndarray):
              try:
                self.trials_raster[:, :, itrial]=blk[sti:endi,:]
              except:
                print("endi=", endi)
                print("sti=", sti)
                print("self.tr_duration=", self.tr_duration)
                print("blk.shape=", blk.shape)
                # keyboard('FAILED assignment check it')
                
            else:
              self.trials_raster[:, :, itrial]=blk[sti:endi,:].todense()
          self.class_trials[self.numclasstrials[itarget], itarget]=itrial
          self.numclasstrials[itarget]+=1
        if len(mismatched_reports)>0 and self.verbose>0:
          print("There were %d groups of mis-matched blocks! Set verbose to >1 to see the details!" % len(mismatched_reports))
          if self.verbose>1: 
             for mmkey in mismatched_reports:
               print("WARNING: trials raster epoch block with shape %s different from tr_duration=%d in %d trials!" % (str(mmkey), self.tr_duration, len(mismatched_reports[mmkey])))
               if self.verbose>2: print("Mismatched trial ids=", mismatched_reports[mmkey])
#      except:
      else:
        print("FAILED!!!")
        print("itrial=", itrial)
        print("sti=", sti)
        print("endi=", endi)
        if use_sparse_tr: print("self.trials_raster has %d sparse matrices with shape=" % len(self.trials_raster), self.trials_raster[0].shape)
        else: print("self.trials_raster.shape=", self.trials_raster.shape)
        print("blk.shape=", blk.shape)
        keyboard('check failure! Check shapes of blk and trials_raster')

  def shuffle_trials_raster(self, nshuffles=1, adtype='spk', grpid=None, omitstims=[], trwindow=None, select_trials=None, keep_xdata=True, ttfrac=0, **gtrkwds):
      if self.trials_raster is None:
        print("Trials_raster not found! Obtaining the default one!")
        self.get_trials_raster(adtype=adtype, grpid=grpid, omitstims=omitstims, trwindow=trwindow, select_trials=select_trials, keep_xdata=keep_xdata, precise_time=precise_time, **gtrkwds)
      self.nshuffles=nshuffles
      self.ttfrac=ttfrac
      if ttfrac: # if ttfrac is not zero return already split data set!
        print("Splitting data before shuffling!!")
        if use_sparse_tr: ntottrials=len(self.trials_raster)
        else: ntottrials=self.trials_raster.shape[2]

        itrain, itest = get_train_test_indices(ntottrials, ttfrac=ttfrac, random_train=True)
        shuff_train_rasts=[]
        shuff_test_rasts=[]
        iclass_train=self.tr_iclass[itrain]
        iclass_test=self.tr_iclass[itest]
        if use_sparse_tr:
          print("WARNING: Shuffling NOT tested for sparse trials_raster!")
          train_trials_raster=[self.trials_raster[itr1] for itr1 in itrain]
          test_trials_raster=[self.trials_raster[itr1] for itr1 in itest]
        else:
          train_trials_raster=self.trials_raster[:,:,itrain]
          test_trials_raster=self.trials_raster[:,:,itest]
        self.iclass_shuff_train_raster = np.tile(iclass_train, nshuffles)
        self.iclass_shuff_test_raster = np.tile(iclass_test, nshuffles)
        self.itrain=itrain
        self.itest=itest
        for ishuff in range(nshuffles):
          shuff_train_rasts.append(trial_shuffle_raster(train_trials_raster, iclass_train))
          shuff_test_rasts.append(trial_shuffle_raster(test_trials_raster, iclass_test))
        self.trial_shuffled_train_raster=np.concatenate(shuff_train_rasts, axis=2)
        self.trial_shuffled_test_raster=np.concatenate(shuff_test_rasts, axis=2)
      else:
        self.trial_shuffled_train_raster=[]
        self.trial_shuffled_test_raster=[]
        self.itrain=None
        self.itest=None
        self.iclass_shuff_raster = np.tile(self.tr_iclass, nshuffles)
        shuff_rasts=[]
        for ishuff in range(nshuffles):
          shuff_rasts.append(trial_shuffle_raster(self.trials_raster, self.tr_iclass))
        self.trial_shuffled_raster=np.concatenate(shuff_rasts, axis=2)
        
  def get_ml_data_tr(self, aoffset=0, astride=None, arange=None, ntimef=None, nshuffles=0, adtype='spk', trwindow=None, fwindow=None, grpid=None, omitstims=[], select_trials=None, keep_xdata=False, precise_time=False, keep_raster=True, ttfrac=0):
    self.nshuffles=nshuffles
    
    if nshuffles: # WITHOUT shuffling
      if self.trial_shuffled_raster is None or  self.nshuffles != nshuffles or ttfrac != self.ttfrac or ( ( trwindow is not None ) and trwindow != self.trwindow):
        print("Shuffled raster not found or new # shuffles requests! Obtaining the shuffled raster!")
        self.shuffle_trials_raster(nshuffles=nshuffles, adtype=adtype, trwindow=trwindow, grpid=grpid, omitstims=omitstims, select_trials=select_trials, keep_xdata=keep_xdata, precise_time=precise_time, ttfrac=ttfrac)
        self.fwindow=fwindow
        
      if ttfrac: 
        tsr_train=self.trial_shuffled_train_raster
        tsr_test=self.trial_shuffled_test_raster
        self.ydata_train=self.iclass_shuff_train_raster
        self.ydata_test=self.iclass_shuff_test_raster
        self.ydata=None
        self.Xdata=None
      else:
        trialrasterdata=self.trial_shuffled_raster
        self.ydata=self.iclass_shuff_raster
    else: # WITHOUT shuffling
      if self.trials_raster is None:
        print("In get_shuffled_ml_data trials_raster not found! Obtaining the default one!")
        self.get_trials_raster(adtype=adtype, trwindow=None, grpid=grpid, omitstims=omitstims, select_trials=select_trials, keep_xdata=keep_xdata, precise_time=precise_time)
        
      trialrasterdata=self.trials_raster
      self.ydata=self.tr_iclass

    if arange is None:
      if fwindow: arange=fwindow[1]-fwindow[0]
      else: arange=6

    if astride is None: astride=self.tr_duration*2
    if ntimef is None: self.ntimef=1
    else: self.ntimef=ntimef
    
    if self.verbose>0: print("Using arange=%d astride=%d aoffset=%d to obtain Xtrain and Xtest data from trials_raster" % (arange, astride, aoffset))
    if ttfrac:
      print("Splitting data while acquiring it!")
      
      tdur, ncells, ntrtrn = tsr_train.shape
      tdur2, ncells2, ntrtst = tsr_test.shape
      assert ncells == ncells2
      assert tdur==tdur2
      if self.verbose>4: 
        print("ttfrac SPLITTING arange=", arange)
        print("ttfrac SPLITTING astride=", astride)
        print("ttfrac SPLITTING aoffset=", aoffset)
      
      if use_sparse_tr: rla=rolling_average_ndarray(tsr_train, arange, astride, offs=aoffset, axis=0, sparse_tr=True)
      else: rolling_average_ndarray(tsr_train, arange, astride, offs=aoffset, axis=0)
      if self.ntimef>rla.shape[0]:
        print("ntimef=%d is to big for the first folded dimension averaged array with dimensions:" % ntimef, rla.shape)
        print("Reducing ntimef to the maximal allowable size: ntimef=%d" % rla.shape[0])
        self.ntimef=rla.shape[0]
      self.Xtrain_data=rla[:self.ntimef,:,:].reshape([self.ntimef*ncells, -1]).T
      try:
        self.Xtrain=self.Xtrain_data[np.tile(self.good_trial_select[self.itrain], self.nshuffles),:][:,np.tile(self.cells_select, self.ntimef)]
      except:
        keyboard('Assignment of self.Xtrain FAILED: Check self.itrain here!')
      labels=self.ydata_train[np.tile(self.good_trial_select[self.itrain], self.nshuffles)]
      self.orig_train_classlabels, self.ytrain = np.unique(labels, return_inverse=True)
      if use_sparse_tr: rla=rolling_average_ndarray(tsr_test, arange, astride, offs=aoffset, axis=0, sparse_tr=True)
      else: rla=rolling_average_ndarray(tsr_test, arange, astride, offs=aoffset, axis=0)
      
      self.Xtest_data=rla[:self.ntimef,:,:].reshape([self.ntimef*ncells, -1]).T
      self.Xtest=self.Xtest_data[np.tile(self.good_trial_select[self.itest], self.nshuffles),:][:,np.tile(self.cells_select, self.ntimef)]
      labels=self.ydata_test[np.tile(self.good_trial_select[self.itest], self.nshuffles)]
      self.orig_test_classlabels, self.ytest = np.unique(labels, return_inverse=True)
    else:
      if self.verbose>4: 
        print("NO ttfrac arange=", arange)
        print("NO splitting astride=", astride)
        print("NO splitting aoffset=", aoffset)
      
      self.Xtrain_data, self.Xtrain, self.ytrain, self.Xtest_data, self.Xtest, self.ytest = None, None, None, None, None, None

      if use_sparse_tr:
        ntrials = len(trialrasterdata)
        tdur, ncells = trialrasterdata[0].shape
      else: tdur, ncells, ntrials = trialrasterdata.shape
        
      rla=rolling_average_ndarray(trialrasterdata, arange, astride, offs=aoffset, axis=0, sparse_tr=use_sparse_tr)
      
      if self.ntimef>rla.shape[0]:
        print("ntimef=%d is to big for the first folded dimension averaged array with dimensions:" % ntimef, rla.shape)
        print("Reducing ntimef to the maximal allowable size: ntimef=%d" % rla.shape[0])
        self.ntimef=rla.shape[0]

      self.Xdata=rla[:self.ntimef,:,:].reshape([self.ntimef*ncells, ntrials]).T
      if self.nshuffles:
        self.Xuse=self.Xdata[np.tile(self.good_trial_select, self.nshuffles),:][:,np.tile(self.cells_select, self.ntimef)]
  #    labels=self.ydata[self.trial_select]
        labels=self.ydata[np.tile(self.good_trial_select, self.nshuffles)]
      else:
        self.Xuse=self.Xdata[self.good_trial_select,:][:,self.cells_select]
  #    labels=self.ydata[self.trial_select]
        labels=self.ydata[self.good_trial_select]
      self.orig_classlabels, self.yuse = np.unique(labels, return_inverse=True)
      
  def plot_trials_raster(self, istim, plotarr=(1,1,0), fold='mean', adtype='spk', trwindow=None, grpid=None, omitstims=[], nshuffles=0, select_trials=None, keep_xdata=True, precise_time=False):
      """ trials_raster: array with dimensions ((duration, ncells, ntrials))"""

      if use_sparse_tr:
        print("plot_trials_raster NOT implemented for sparse trials_raster!")
        return
        
      if self.trials_raster is None:
        self.get_trials_raster(adtype=adtype, trwindow=trwindow, grpid=grpid, omitstims=omitstims, nshuffles=0, select_trials=select_trials, keep_xdata=keep_xdata, precise_time=precise_time)
      mnact=self.trials_raster.mean(0)
      plt.plot(mnact[:, self.trial_stim_inds==istim].mean(1))
      plt.show()
        
  def do_trial_shuffle(self, rndseed=-1):
      if rndseed>0: np.random.seed(rndseed)
      self.Xshuf=self.Xuse.copy()
      self.yshuf=self.yuse.copy() # it is not going to change after shuffle; just use for now for testing, that shuffling d
      uclasses, ccounts = np.unique(self.yshuf, return_counts=True)
      ntrialsmax=max(self.class_counts)
      self.shuffle_matrix=-np.ones((self.nf, self.nclasses, ntrialsmax), dtype='int32')
      for iclss, (clss, nclss) in enumerate(zip(self.iclasses, self.class_counts)):
        clssindx=np.where(self.yshuf == clss)[0]
        for ifeature in range(self.nf):
          iperm=np.random.permutation(nclss)
          shuff_clssindx=clssindx[iperm]
          self.shuffle_matrix[ifeature, iclss, :nclss]=shuff_clssindx
          self.Xshuf[shuff_clssindx,ifeature]=self.Xuse[clssindx,ifeature]
          self.yshuf[shuff_clssindx]=self.yuse[clssindx]
          assert all(self.yshuf[shuff_clssindx]==clss)

  def do_trial_shuffle_Xdata(self, retmat=False, rndseed=-1):
      if retmat: self.Xshuf, self.shuffle_matrix = trial_shuffle_Xdata(self.Xuse, self.yuse, retmat=True, rndseed=rndseed)
      else: self.Xshuf=trial_shuffle_Xdata(self.Xuse, self.yuse, retmat=False, rndseed=rndseed)

  def show_stim_cells(self, scolor='r', ocolor='b'):
    plt.plot(self.cellcoords[0,self.icells_stim], self.cellcoords[1,self.icells_stim], 'o', color=scolor)
    plt.plot(self.cellcoords[0,self.icells_nonstim], self.cellcoords[1,self.icells_nonstim], 'o', color=ocolor)
    plt.show()

class VisStimData:
  def __init__(self, dataspec, ncells=None, fwindow=None, spktype='spikes', samptype='n-twb', kvalue=None, sortby='', mode=0, datainfo = gavdatainfo):
    self.compiled=False
    self.datainfo= datainfo
    self.avinf = None
    self.fwindow = None
    self.smptype = 'allsamp'
    self.kvalue = kvalue
    self.dataspec = dataspec
    self.df=None
    self.ncells=0 # will trigger assign_basic_info
    self.raster_data=[] # will store basic information contained in matlab files -- it will triggered get_raster_data
    self.psarray=[] # will trigger get_psarray
    self.istarr=[] # maybe not needed here -- check!
    self.rflabels=[] # will trigger assign class variables  from raster_data
    self.X=[]
    self.wX=[]
    self.ttfeatures = []
    self.isort={}
    self.info={}
#    if isinstance(dataspec, str):
#      if len(dataspec)>2 and dataspec[:3]=='vis':
#        datafile, labelsfile, dmode,coordfile = get_data_info(dataspec)
#        rfdata, vstim, frate, labels, dd = get_data(datafile, labelsfile=labelsfile)
#        psarray, ulabels, istim=get_responses_array(rfdata, vstim, labels)


    self.get_raster_data(dataspec, spktype=spktype)
    
    if dataspec in gavdatainfo:
        self.kvalue = gavdatainfo[dataspec][1]
        self.avinf=AvalancheInfo(self.dataspec, dd=self.dd)
    else:
        self.avinf=AvalancheInfo(None)
        if kvalue is not None: self.kvalue = kvalue
        else: self.kvalue=1
        
#    self.get_psarray()
#    self.get_ml_data(samptype=samptype, fwindow=fwindow)
    self.get_pswin_data(fwindow=fwindow)
    
    if sortby:
      self.sortby=sortby
      if sortby != 'dsi':
        self.isort['dsi'] = sort_cells_by_selectivity(self.wX, self.wy, sortby='dsi', psarray=self.psarray, ulabels=self.ulabels, nstimclasses=self.nstimclasses)
      self.isort[sortby] = sort_cells_by_selectivity(self.wX, self.wy, sortby=sortby, psarray=self.psarray, ulabels=self.ulabels, nstimclasses=self.nstimclasses)
      
  def assign_basic_info(self, stimlabels=None, verbose=0):
    if stimlabels is None: stimlabels=self.raster_data['stimlabels']
    ulabels,cnts=np.unique(stimlabels, return_counts=True)
    stimcounts=dict(zip(ulabels,cnts))
    nclasses=len(ulabels)
    self.ntrials=len(stimlabels)
    assert self.ntrials== self.raster_data['istim'].size
    self.ulabels=ulabels
    self.maxrep=max(cnts)
    self.nclasses=nclasses
    self.nstimclasses=nclasses
    self.nframes,self.ncells=self.raster_data['rfdata'].shape

  def  assign_optional_vsdvs(self, optvars=optraster_datavars_vstim):
    if not self.ncells: self.assign_basic_info(verbose=verbose)
    self.info['optvars'] = optvars
    nframes=self.nframes
    assert nframes == self.raster_data['rfdata'].shape[0]
    if 'ps_dtlen_max' not in self.raster_data:
      isdiffs=self.raster_data['istim']
      dtlenmin=int(np.min(isdiffs))
      dtlenmax=int(np.max(isdiffs))
      self.raster_data['ps_dtlen_max']=dtlenmax
      self.raster_data['ps_dtlen_min']=dtlenmin
    else:
      dtlenmax=self.raster_data['ps_dtlen_max']
    if 'rftimes' not in self.raster_data:
      self.raster_data['rftimes']=1000.*np.arange(nframes)/self.raster_data['frate']
    if 'rftrid' not in self.raster_data:
       rftrid=np.zeros(nframes)-1
       istims=self.raster_data['istim']
       for itr,istm1 in enumerate(istims):
         rftrid[istm1:istm1+dtlenmax]=itr
       self.raster_data['rftrid']=rftrid
    if 'rflabels' not in self.raster_data:
       rflabs=np.zeros(self.nframes).astype(int)+self.nclasses
       for istm1,lbl1 in zip(self.raster_data['istim'], self.raster_data['stimlabels']):
         rflabs[istm1:(istm1+dtlenmax)]=lbl1
       self.raster_data['rflabels']=rflabs
                
  def get_psarray(self, stimlabels=None, verbose=0):
    if not self.ncells: self.assign_basic_info(verbose=verbose)
#    repcounter=np.zeros(self.nclasses, dtype=int )
    dtlen1=self.ps_dtlen
    if self.maxrep<1:
      print("self.maxrep=", self.maxrep)
      keyboard('sta to bi sa maxrep-om?')
      
    self.psarray=np.zeros((self.maxrep, self.nclasses, self.ps_dtlen, self.ncells))
    self.istarr=np.zeros((self.maxrep, self.nclasses), dtype=int)
    self.nclasstrials=np.zeros(self.nclasses, dtype=int)
    try:
      for ist,sclass in zip(self.istim,self.stimlabels):
#        print("ist=", ist, " sclass=", sclass)
        self.psarray[self.nclasstrials[sclass], sclass, :, :]=self.rfdata[ist:(ist+dtlen1),:]
        self.istarr[self.nclasstrials[sclass], sclass]=ist
        self.nclasstrials[sclass]+=1
    except:
      keyboard('failed psarray sh12')

  def get_class_ists(self, sclass):
    return self.istarr[:self.nclasstrials[sclass],sclass]
    
  def get_raster_data(self, dataspec, spktype='default'):
    " use dataspec vis1, vis2, to load the data, or pass (rfdata, vstim, frate, stimlabels) tuple "
    self.spktype=spktype
    self.dataspec=dataspec
    if isinstance(dataspec,str):
      if 'sim' in dataspec:
        self.raster_data=get_simulated_data(dataspec)
      else:
        print('Getting %s data for %s' % (spktype, dataspec))
        datafile, labelsfile, dmode,coordfile = get_data_info(dataspec)
        self.datafile=datafile
        self.labelsrc=labelsfile
#       if dataspec in ['vis1','vis2','vis3','vis4','vis5','vis6']:
        if dmode==1:
          if spktype=='default': spktype='spikes'
          rfdata, vstiminfo, frate, stimlabels, dd = get_dataA(datafile, labelsfile=labelsfile, spktype=spktype)
          print("dataA: rfdata.shape=", rfdata.shape)
          self.dd=dd
          istim, jstim, dtlen, maxrep=vstiminfo[0:4]
          self.vstim=vstiminfo[3]
          dtlenmin=int(np.min(np.diff(istim)))
          dtlenmax=int(np.max(np.diff(istim)))
          self.raster_data = {'name': dataspec, 'rfdata': rfdata, 'istim': istim, 'jstim' :jstim, 'stimlabels' : stimlabels,  'frate':frate, 'ps_dtlen':dtlen}
#      elif dataspec in ['visB1']:
        elif dmode==2:
          if spktype=='default': spktype='spk'
          rfdata, vstiminfo, stimlabels, dd = get_dataB(datafile, spktype=spktype)
          self.dd=dd
          print("dataC: rfdata.shape=", rfdata.shape)
          istim, jstim, dtlen, frate, maxrep=vstiminfo[0:5]
          frametimes = vstiminfo[5]
          self.raster_data = {'name': dataspec, 'rfdata': rfdata, 'istim': istim, 'jstim' :jstim, 'stimlabels' : stimlabels,  'frate':frate, 'ps_dtlen': dtlen, 'rftimes': frametimes }
        elif dmode==3:
          if spktype=='default': spktype='spk'
          rfdata, vstiminfo, stimlabels, dd = get_data_cw(datafile, spktype=spktype)
          self.dd=dd
          print("dataC: rfdata.shape=", rfdata.shape)
#      vistiminfo=[istim, jstim, dtlen, frate, frametimes, vstim]
          istim, jstim, dtlen, frate, maxrep=vstiminfo[0:5]
#         self.vstim=vstiminfo[3]
          self.frame_times = vstiminfo[5]
          self.raster_data = {'name': dataspec, 'rfdata': rfdata, 'istim': istim, 'jstim' :jstim, 'stimlabels' : stimlabels,  'frate':frate, 'ps_dtlen':dtlen}
    elif isinstance(dataspec, dict):
       self.dataspec="dict_input"
       self.raster_data=dataspec
    elif (isinstance(dataspec, list) or isinstance(dataspec, tuple)): # old method -- should remove this eventually
       self.dataspec="raw_input"
       if len(dataspec)==5:
         rfdata, istim, jstim, stimlabels, odinfo = dataspec
#         odinfo=={'frate':frate, 'ps_dtlen':dtlen, 'maxrep': maxrep} )
         jstim, dtlen= -1, int(np.min(np.diff(istim)))
         self.raster_data = {'name': dataspec, 'rfdata': rfdata, 'istim': istim, 'jstim' :jstim, 'stimlabels' : stimlabels}
         self.raster_data.update(odinfo)
    else:
      raise ValueError("Unrecognized data specification in get_raster_data")
    
    self.raster_data2classvars()
    self.clean_nans(nanvalue=0)
     
  def raster_data2classvars(self, verbose=0):
    # will assign basic info if self.ncells is 0
    self.assign_optional_vsdvs()
    for vsdv in raster_datavars+optraster_datavars:
      if self.verbose>3: print('Assigning calss variable %s from raster_data ' % vsdv)
      exec("self.%s = self.raster_data['%s']" % (vsdv, vsdv))

  def get_pswin_data(self, fwindow=None, gathermode=1, verbose=0):
    if len(self.psarray)==0: self.get_psarray(verbose=verbose)
    if fwindow is None:
      if self.fwindow is None: pstim1,pstim2=0,None
      else: pstim1, pstim2=self.fwindow
    else:
      pstim1,pstim2=fwindow
      self.fwindow=[pstim1,pstim2]
    if pstim2 is None: pstim2i=self.ps_dtlen
    else: pstim2i=pstim2
    if pstim2>self.ps_dtlen:
      warning
    if gathermode==1:
      xlist=[]
      ylist=[]
      irfindexlist=[]
      ipstimf=[]
      for icls in range(self.nclasses):
         ntotreps=self.nclasstrials[icls]
         xcur=self.psarray[:ntotreps, icls, pstim1:pstim2,:].reshape([-1,self.ncells])
         xlist.append(xcur)
         ylist.extend([icls]*xcur.shape[0])
         for itrc in range(self.nclasstrials[icls]):
           irfindexlist.extend(self.istarr[itrc, icls]+np.arange(pstim1, pstim2i))
           ipstimf.extend(np.arange(pstim1, pstim2i))
         
      self.wX=np.concatenate(xlist)
      self.wy=np.array(ylist)
      self.irfwX=np.array(irfindexlist)
      self.ipstimf=np.array(ipstimf)
      self.cell_columns=['c%d' % ii for ii in range(self.ncells)]
      self.df=pd.DataFrame(self.wX, columns=self.cell_columns)
      nsamples=len(self.wy)
      rs1=self.wy+np.random.randn(nsamples)*0.1
      rs2=self.wy+np.random.randn(nsamples)*0.2
      rs3=self.wy+np.random.randn(nsamples)*0.3
      zs1=rs1*np.exp(np.random.rand(nsamples)*2j*np.pi)
      zs2=rs2*np.exp(np.random.rand(nsamples)*2j*np.pi)
      zs3=rs3*np.exp(np.random.rand(nsamples)*2j*np.pi)

      addcolsd={'y': self.wy,
               'ftimes': self.rftimes[self.irfwX],
                'trid': self.rftrid[self.irfwX].astype(int),
                'ipstimf': self.ipstimf,
                'fakeA1': self.wy//2,
                'fakeA2': self.wy%2,
                'fakeB1': np.sqrt(self.wy//2) - 2*self.wy%2,
                'fakeB2': 1-2*self.wy%2,
                'fake1x': zs1.real,
                'fake1y': zs1.imag,
                'fake2x': zs2.real,
                'fake2y': zs2.imag,
                'fake3x': zs3.real,
                'fake3y': zs3.imag,
                }
      
      self.df=pd.concat([self.df, pd.DataFrame(addcolsd)], axis=1)
      self.df_features = self.cell_columns+['ftimes','trid','ipstimf']
      self.ndffeatures = len(self.df_features)     # self.ncells+3
    else:
      self.get_train_test_data(fwindow=fwindow, ttfrac=ttfrac, verbose=verbose)

  def set_features(self, mode=1, ttfeatures='all', fwindow=None, ttfrac=0.7, verbose=0, balance=False):
    if ttfeatures=='all':
      self.ttfeatures = self.df_features
    elif ttfeatures=='cells':
      self.ttfeatures = self.cell_columns
    elif ttfeatures=='cpst':
      self.ttfeatures = self.cell_columns+['ipstimf']
    elif ttfeatures[1:5]=='fake':
        if ttfeatures[0]=='c': addcols=self.cell_columns
        else: addcols=self.df_features
        if ttfeatures[5]=='A':
          self.ttfeatures = addcols+['fakeA1','fakeA2']
        elif ttfeatures[5]=='B':
          self.ttfeatures = addcols+['fakeB1','fakeB2']
        elif ttfeatures[5]=='1':
          self.ttfeatures = addcols+['fake1x','fake1y']
        elif ttfeatures[5]=='2':
          self.ttfeatures = addcols+['fake2x','fake2y']
        elif ttfeatures[5]=='3':
          self.ttfeatures = addcols+['fake3x','fake3y']
    else:
      print('Unknown ttfeature! Using all')
      self.ttfeatures = self.df_features
      
    self.nfeatures = len(self.ttfeatures)
      
  def get_train_test_data(self, mode=1, ttfeatures='all', fwindow=None, ttfrac=0.7, verbose=0, balance=False):
    """ generate training and testing sets based on mode (0 -random 1- beggining vs end 2- blocks ) and
        ttfeatures : 'all', 'cells', 'cpst', 'cfake', 'afake'
    """
    if len(self.psarray)==0: self.get_psarray(verbose=verbose, fwindow=fwindow)
    
    if mode==0:
         n_trials=self.ntrials
         ntrain1=int(ttfrac*n_trials)
         ntest1=n_trials-ntrain1
         iprmreps=np.random.permutation(n_trials)
         itrtrain=iprmreps[:ntrain1]
         itrtest=iprmreps[ntrain1:]
         self.df_train=self.df[self.df['trid'].isin(itrtrain)]
         self.df_test=self.df[self.df['trid'].isin(itrtest)]
    elif mode==1:
         n_trials=self.ntrials
         ntrain1=int(ttfrac*n_trials)
         ntest1=n_trials-ntrain1
         iprmreps=np.arange(n_trials)
         itrtrain=iprmreps[:ntrain1]
         itrtest=iprmreps[ntrain1:]
         self.df_train=self.df[self.df['trid'].isin(itrtrain)]
         self.df_test=self.df[self.df['trid'].isin(itrtest)]
    elif mode=='psarray':
      xtrainlist=[]
      ytrainlist=[]
      xtestlist=[]
      ytestlist=[]
      if fwindow is None:
        if self.fwindow is None:
          pstim1,pstim2=0, None
        else:
          pstim1,pstim2 = self.fwindow
      else:        
        pstim1,pstim2 = fwindow
        keyboard('This is not going to be compatible with wX and wy fwindow')
      for icls in range(self.nclasses):
         ntotreps=self.nclasstrials[icls]
         ntrain1=int(ttfrac*ntotreps)
         print("ntotreps=", ntotreps)
         print("ttfrac=", ttfrac)
         print("ntrain1=", ntrain1)
         ntest1=ntotreps-ntrain1
         iprmreps=np.random.permutation(ntotreps)
         xcur=self.psarray[iprmreps[:ntrain1],icls,pstim1:pstim2,:].reshape([-1,self.ncells])
         xtrainlist.append(xcur)
         ytrainlist.extend([icls]*xcur.shape[0])
         xcur=self.psarray[iprmreps[ntrain1:ntotreps],icls,pstim1:pstim2,:].reshape([-1,self.ncells])
         xtestlist.append(xcur)
         ytestlist.extend([icls]*xcur.shape[0])
    if balance:
      print('Balancing data NO implemented yet!')
    print('Assigning Xtrain, ytrain, Xtest, ytest')
    self.Xtrain=self.df_train[self.ttfeatures]
    self.ytrain=self.df_train['y']
    self.Xtest=self.df_test[self.ttfeatures]
    self.ytest=self.df_test['y']
    
  def get_trialids(self, **kwargs):
    if len(self.istarr)==0:
      self.get_psarray(**kwargs)
      
  def get_rframes(self, **kwargs):
    if len(self.istarr)==0:
      self.get_psarray(**kwargs)
      
  def label_frames(self):
    try:
      self.rflabels=np.zeros(self.nframes).astype(int)+self.nclasses
      for istm1,lbl1 in zip(self.istim, self.labels):
        self.rflabels[istm1:(istm1+self.ps_dtlen_max)]=lbl1
    except:
      print('Need self.istim, self.labels, and self.ps_dtlen_max to do this')

  dimredlist=['pca_2_wX', 'pca_3_wX', 'lda_2_wX', 'lda_3_wX']
  
  def dimred(self, method='lda_3_df', smptype=None, features=[], fX=[], fy=[], return_proj=False, fwindow=None):
    ldinfo=method.split('_') # first items is a method and the rest are method parameters, e.g., pca_3_t
    methodname=ldinfo[0]
    if smptype is None: smptype=self.smptype
    if len(ldinfo)>1: ncomp=int(ldinfo[1])
    else: ncomp=3
    if len(ldinfo)>2: dstr=ldinfo[2] # dstr 'X' or 'R' for sample of raster frames
    else: dstr='wX'
    if methodname == 'lda':
      self.dimredname=method
      self.dimredalg = LinearDiscriminantAnalysis(n_components=ncomp)
    elif methodname == 'pca':
      if len(ldinfo)>1: ncomp=int(ldinfo[1])
      self.dimredalg = PCA(n_components=ncomp)

    if fwindow is None: fwindow=self.fwindow
    
    if dstr=='df':
        if features: fX=self.df[features]
        else: fX=self.df[self.df_features]
        fy=self.df['y']
    elif dstr=='wX':
        if len(self.wX)==0:
          self.get_pswin_data(fwindow=fwindow)
        fX,fy=self.wX,self.wy
    elif dstr=='R':
        self.dimredalg.fit(self.rfdata, self.rflabels)
        self.lsproj=self.dimredalg.transform
        if return_proj: return self.dimredalg.transform(self.rfdata)
    else:
        print('Unknow dstr=', dstr)
        return None
    assert len(fX)>0
    self.dimredalg.fit(fX, fy)
    self.lsproj=self.dimredalg.transform
    if return_proj: return self.dimredalg.transform(fX)

  def clean_nans(self, nanvalue=0):
      self.info['nanvalue']=nanvalue
      self.rfdata[np.isnan(self.rfdata)]=nanvalue
      if self.psarray:
        self.psarray[np.isnan(self.psarray)]=nanvalue
    
  def plot_trajectories(self, dmethod='pca_3_wX', ptype='l', ndim=None, astart=0, frmstep=1):
    ldinfo=method.split('_') # first items is a method and the rest are method parameters, e.g., pca_3_t
    self.dimred(method=dmethod)
    print("self.rfdata.shape=", self.rfdata.shape)
    if len(self.rflabels)==0:
      self.label_frames()
    if userf:
      dimreddata=self.rfdata
      dimredlabels=self.rflabels
      rproj = self.dimred_transform(dimreddata)
    if ncomp==2:
        if len(ldinfo)>2: pcolor=ldinfo[2]
        else: pcolor='time'
        # decided how to desplay trajectory based on pcolor (thir
        if pcolor in ['time', 't']:
          acolors = np.linspace(0, 1, rproj.shape[0])
          plt.scatter(rproj[:, 0], rproj[:, 1], c=acolors, cmap='jet')
          # Add a colorbar to the plot to indicate the time points
          cbar = plt.colorbar()
          cbar.set_label('Time')
        elif pcolor in ['l', 'lbl', 'label']:
          npts=rproj.shape[0]
          lcolorslist=list('krgbcm')+mpcolors
          lcolors=[lcolorslist[ilbl] if ilbl>=0 else (0.8,0.8,0.8) for ilbl in self.rflabels]
          print("Animating npts=", npts)
          assert npts == len(lcolors)
          if astart:
            if astart<0:
              astart1=self.i
            pltpoint, = plt.plot([],[], marker="o", c='r')
            keeploop=True
            ncheckloop=1000
            t0=time.time()
            i0=0
            ianim=0
            for ipt in range(1,npts,frmstep):
              plt.plot(rproj[ipt-1:ipt+1, 0], rproj[ipt-1:ipt+1, 1], color=lcolors[ipt])
              pltpoint.set_data(rproj[ipt,0], rproj[ipt,1])
              pltpoint.set_color(lcolors[ipt])
              plt.title('Frame %d' % ipt)
              plt.pause(0.01)
              ianim+=1
              if (ianim%ncheckloop == 0) and keeploop:
                 print("ncheckloop=", ncheckloop)
                 tela=time.time()-t0
                 print("There is another %g seconds to go" % ((npts-ipt)*tela/(ipt-i0)))
                 ans=input('Do you want to break the loop? (b, y), or increase check frequency (i), or continue (n,c), or set loop size #integer!')
                 try:
                   ncheckloop=int(ans)
                 except:
                   if ans[0] in ['y','b']:
                     print('Exiting animation!')
                     break
                   elif ans[0] =='i':
                     ncheckloop*=2
                   elif ans[0] in ['n','c']:
                     ncheckloop=npts
                 t0=time.time()
                 i0=0
          else:
            for ipt in range(1,npts):
              plt.plot(rproj[ipt:(ipt+2), 0], rproj[ipt:(ipt+2), 1], 'o', color=lcolors[ipt])
          
          # Add axis labels and a title to the plot
        plt.xlabel('PCA component 1')
        plt.ylabel('PCA component 2')
        plt.title('Colored Trajectories of PCA Components')
      # Show the plot
        plt.show()

    if 0:
        for ipt in range(1,npts):
          plt.plot(rproj[ipt:(ipt+2), 0], rproj[ipt:(ipt+2), 1], color=lcolors[ipt%10])
          plt.pause(0.1)
        if 0:
          plt.scatter(rproj[:, 0], rproj[:, 1], c=colors, cmap='jet')
        plt.show()

  def get_pafa_info(self, avinf=None, nshiftexplore=None, reloadai=False):
      pass
            
  def get_yparab_tally(self, ishuffle=0):
  #  garbage below
      for ii in range(ntypes):
        print("ii=", ii)
        if ii in smptally:
          print("ii in=", ii)
          runinfo['nframes'][ii,ishf]=smptally[ii]
        else:
          runinfo['nframes'][ii,ishf]=0
   # maybe use this
      self.avinf.avshinfo[ishuffle]['smplabels'].reshape([-1,self.dtlen])
      for icl in range(nstimclasses):
        if icl not in yparabtally:
          yparabtally[icl]=0
          
      iparabavs = ( self.smplabels==1 )
      yparabs=y[iparabavs]
      self.yparabtally=tally(yparabs)

  def evaluate_simple(self, cmodel, ttfeatures=None, ttmode=None, ttfrac=None, dimred='', ndim=0):
      if ttfeatures is None: ttfeatures=self.ttfeatures
      if ttmode is None: ttmode=self.ttmode
      if ttfrac is None: ttfrac=self.ttfrac
      self.get_train_test_data(mode=ttmode, ttfrac=ttfrac, ttfeatures=ttfeatures)

      if len(dimred)>0 and ndim>0:
        self.dimred(self, method=dimred, smptype='allsamp', fX=[], fy=[], return_proj=False) # not finished apparently
        
      if cmodel[:2]=='tf':
        tf.keras.backend.clear_session()
        klsb=Klasifajer(cmodel, {'hiddenls': [200, 50]}, data=self.Xtrain, labels=self.ytrain, nclasses=self.nstimclasses)
        klsb.fit(self.Xtrain, self.ytrain, nepochs=1000, verbose=2)
#        klsb.model.fit(Xdatabig, y_train, epochs=neadd)
        p2=klsb.evaluate(self.Xtest, self.ytest)
        print("paccuracy=%f" % (p2))
      elif cmodel[:2]=='rf':
        if False:
          klsb=Klasifajer(cmodel, {'mxf': 'sqrt', 'mxdepth': None}, nclasses=self.nstimclasses)
          klsb.fit(self.df_train[self.df_features], self.df_train['y'])
          p2=klsb.evaluate(self.df_test[self.df_features], self.df_test['y'])
          print("paccuracy=%f" % (p2))
        klsb=Klasifajer(cmodel, {'mxf': None, 'mxdepth': None}, nclasses=self.nstimclasses)
        klsb.fit(self.Xtrain, self.ytrain)
        p2=klsb.evaluate(self.Xtest, self.ytest)
        print("paccuracy again=%f" % (p2))
      else:
        keyboard('wrong cmodel')
      return p2
    
  def evaluate_oob(self, cmodel='rf1', ttfeatures='all', noob=0, ttfrac=0.8, franks=None, nselect=np.array([20, 50, 100]), sortby='lda1', namplify=1, eprmexplore=[None], eprmname='', datarray=None, ulabels=None, random_train=True, lsdim=3, ntotclasses=None, getmodels=False, verbose=0):
  # eprmname='nepochs'
    info={}
    if self.verbose>0: print("calling evaluate_prediction_noob with cmodel=", cmodel)

    if noob<=0:
      random_train=False
      noob=1
    
    if sortby in ['shf1', 'shf2']:
      print('SHUFFLING SHUFFLING!')
      np.random.shuffle(yev)
      
    nans=np.isnan(Xev)
    nnans=nans.sum()
    if nnans>0:
      print('WARNING: There were %d NaN in %s' % (nnans, dataname))
      print('Setting them to 0')
      Xev[nans]=0.
      time.sleep(1)
    ynans=np.isnan(yev)
    nynans=ynans.sum()
    if nynans>0:
      print("nynans=", nynans)
      raise ValueError('NaNs not allowed in labels!')

    nne=len(eprmexplore)
    if cmodel[:2]=='tf':
      nepochs=eprmexplore
      nne=len(nepochs)
      epdiffs=[nepochs[0]]+list(np.diff(nepochs))
      print("epdiffs=", epdiffs)
    if cmodel[:2] in ['rf','dt']:
      mxdepthexplore= [3,5,None]
      mxdepthexplore=eprmexplore
    if cmodel[:3] in ['svm']:
      gammaexplore= ['auto','scale', 0.1]
      gammaexplore=eprmexplore
    
    nselvals=len(nselect)
    if self.verbose>1:
      print("nselect=", nselect)
      print("nselvals=", nselvals)
    paccuracy=np.zeros((noob,nselvals,nne))-1
    nsamps,nfeatures=Xev.shape
    print("nfeatures=", nfeatures)
    print("nsamps=%d nfeatures=%d" % (nsamps, nfeatures))
    if namplify==1:
      Xnew, ynew = Xev, yev
    else:
      Xnew, ynew =amplify_noise(Xev, yev, namplify, ampf=ampnoisef)

    if ulabels is None:
      ulabels,cnts= np.unique(yev, return_counts=True)
      for iiii in range(100):
         print("ulabels=", ulabels)

    if getmodels: models=[]
    print('Starting new OOB loop! ')
    for inoob in range(noob):
      print("Starting inoob=%d of %d" % (inoob+1,noob))
      ulab1,cnts1= np.unique(ynew, return_counts=True)
      if self.verbose>3:
        print("ulab1=", ulab1)
        print("cnts1=", cnts1)

      Xtrain, ytrain, Xtest, ytest = split_train_test(Xnew, ynew, ttfrac=ttfrac, random_train=random_train)
          
      ulab2,cnts2= np.unique(ytrain, return_counts=True)
      ulab3,cnts3= np.unique(ytest, return_counts=True)
      
      if self.verbose>3:
        print("ulab2=", ulab2)
        print("cnts2=", cnts2)
        print("ulab3=", ulab3)
        print("cnts3=", cnts3)
      
      if len(ytrain)<2:
        print("len(ytrain)=", len(ytrain))
        keyboard('insied evaluate nob-ove --  check sta je ovo')

#      isortedf = select_features_dsi(Xtrain, ytrain, sortby=sortby, datarray=datarray, ulabels=ulabels, lsdim=lsdim, ntotclasses=ntotclasses)
# check the difference between nclasses and nstimclasses
      isortedf = sort_cells_by_selectivity(Xtrain, ytrain, sortby=sortby, nclasses=ntotclasses, psarray=datarray, ulabels=ulabels, lsdim=3, nstimclasses=ntotclasses, retsortvals=False)
#                          select_features_dsi(Xtrain, ytrain, sortby=sortby, datarray=datarray,                                 ulabels=ulabels, lsdim=lsdim, ntotclasses=ntotclasses)
      #y_train = tf.keras.utils.to_categorical(ytrain, 8)
      y_train = ytrain.astype('uint64')
      print("nselect=", nselect)
      for isl,nsel in enumerate(nselect):
        print('Starting loop for nselect: %d of %d' % (isl+1, nselvals))
        sys.stdout.flush()
        if nsel>0:  
          isel=isortedf[:nsel]
          Xdatabig=Xtrain[:,isel]
          Xtestbig=Xtest[:,isel]
        else:
          Xdatabig=Xtrain
          Xtestbig=Xtest

        if cmodel[:2]=='tf':
          tf.keras.backend.clear_session()
          klsb=Klasifajer(cmodel, {'hiddenls': [200, 50]}, data=Xdatabig, labels=y_train, nclasses=ntotclasses)
          nepc=0
          for iep,neadd in enumerate(epdiffs):
            nepc+=neadd
#            klsb.mprms['nepochs']=neadd
#            del klsb.mprms['nepochs'] # mprms takes precedence in the fit -- should I change this
            print("Incremental fitting for nepochs=%d (total %d)" % (neadd, nepc))
            klsb.fit(Xdatabig, y_train, nepochs=neadd)
            klsb.model.fit(Xdatabig, y_train, epochs=neadd)
            p2=klsb.evaluate(Xtestbig, ytest)
            print("%s: %d -> p2=%f" % (eprmname, nepc, p2))
            paccuracy[inoob, isl, iep]=p2
            print("nepochs=%d vs nsel=%d predval=%g" % (nepc, nsel, p2))
        else:
          for ipe, prmv in enumerate(eprmexplore):
            print("eprmname=", eprmname)
            print("prmv=", prmv)
            klsb=Klasifajer(cmodel, {eprmname: prmv}, data=Xdatabig, labels=y_train, nclasses=ntotclasses)
            klsb.fit(Xdatabig, y_train)
            p2=klsb.evaluate(Xtestbig, ytest)
            print("%s: %s -> p2=%f" % (eprmname, str(prmv), p2))
            paccuracy[inoob, isl, ipe]=p2

        print('Gotten this accuracy:', p2)
# to save space append only the latest of explored ones (that one should have the highest accuracy          
        if getmodels:
           models.append(klsb)
    #savearray=np.array([paccuracy, nselect, nepochs],dtype=object)
    if getmodels: info['models']=models
    info['macc']=paccuracy.max()
    return paccuracy, nselect, {eprmname : eprmexplore}, info

    pass
  
  def plot_feature_pairs(self, featurepair=['fake2x','fake2y']):
    nplots=len(featurepair)//2
    nrow,ncol=getNxM(nplots)
    fig,axs=plt.subplots(nrows=nrow, ncols=ncol)
    if min([nrow,ncol])==1:
        for iplt in range(nplots):
          axs[iplt].scatter(self.df[featurepair[2*iplt]], self.df[featurepair[2*iplt+1]], c=self.df['y'])
    else:
        for iplt in range(nplots):
          irow=iplt//ncol
          icol=iplt%ncol
          axs[irow,icol].scatter(self.df[featurepair[2*iplt]], self.df[featurepair[2*iplt+1]], c=self.df['y'])
    plt.show()
  
############## END OF VisStimData ############## END OF VisStimData ############## 
############## END OF VisStimData ############## END OF VisStimData ############## 
############## END OF VisStimData ############## END OF VisStimData ############## 
############## END OF VisStimData ############## END OF VisStimData ############## 
############## END OF VisStimData ############## END OF VisStimData ############## 
############## END OF VisStimData ############## END OF VisStimData ############## 

def get_simulated_data(dataspec):
  if dataspec=='simtiny':
      ntt=10
      ncells=3
#      rfdata = np.tile(np.arange(ntt),(ncells,1)).T
      rfdata = np.arange(ntt)[:, None] + np.arange(ncells)*ntt
      istim=np.arange(0,4)*2 + 1
      jstim=istim+1
      stimlabels=np.array([0, 1, 0, 1])
      frate=11.11
      dtlen=2
      return {'name': dataspec, 'rfdata': rfdata, 'istim': istim, 'jstim' :jstim, 'stimlabels' : stimlabels, 'frate' : frate, 'ps_dtlen' : dtlen}
  elif dataspec=='simmini':
      ntt=100
      ncells=5
      rfdata=np.tile(np.arange(ntt),(ncells,1)).T
      istim=np.arange(1,9)*11
      jstim=istim+5
#      labels=list('abcdefghijk')
      stimlabels=np.arange(len(istim))%3
      frate=11.11
      dtlen=7
      return {'name': dataspec, 'rfdata': rfdata, 'istim': istim, 'jstim' :jstim, 'stimlabels' : stimlabels, 'frate' : frate, 'ps_dtlen' : dtlen}

def loadnpy(filename, enc='latin1'):
    if filename.strip()[-7:] == '.npy.gz':
      aload=np.load(gzip.open(filename, 'rb'), encoding=enc, allow_pickle=True)
    else:
      aload=np.load(filename, encoding=enc, allow_pickle=True)
    if aload.dtype == np.dtype('O'):
      return aload.tolist()
    else:
      return aload
    
def getNxM(n, aspectratio=1):
  print("getNxM n=", n)
  print("getNxM aspectratio=", aspectratio)
  nr=np.ceil(np.sqrt(n*aspectratio))
  if nr==0: nr=1
  nc=(np.ceil(n/nr))
  nblank=nr*nc-n
  print(nr, nc)
  print([nr/2.,nc/2.])
  print(min([nr/2.,nc/2.]))
  if nblank>min([nr/2,nc/2]):
    nexplr=min([nr-1,2])
    nrs=np.arange(nr-nexplr,nr+nexplr+0.5)
    nblanks=[x*np.ceil(n/x)-n for x in nrs]
    print('blanks:', nblanks)
    nr=nrs[np.argmin(nblanks)]
    nc=(np.ceil(n/nr))
    nblank=nr*nc-n
  print('AR=%g n=%d nr=%d nc=%d nblank=%d' % (aspectratio, n, nr, nc, nblank))
  return [int(nr), int(nc)]
    
from numpy.lib.stride_tricks import as_strided

def rolling_average_ndarray_simple(arr, nrng, nstride, offs=0, axis=0):
      arr = np.asarray(arr)
      arr = np.moveaxis(arr, axis, 0)
      if (nrng+offs)> arr.shape[0]:
        print("Unable to do rolling average! Reduce offs or nrng! offs=%d nrng=%d arr.shape[0]=%d" % ( offs, nrng, arr.shape[0]))
        return None
      if offs: arr = arr[offs:]
      # Calc the shape and strides for as_strided along the first dimension
      shape = ((arr.shape[0] - nrng) // nstride + 1, nrng) + arr.shape[1:]
      strides = (arr.strides[0] * nstride, arr.strides[0]) + arr.strides[1:]
      strided_view = as_strided(arr, shape=shape, strides=strides)
      avresult = strided_view.mean(axis=1)
      avresult = np.moveaxis(avresult, 0, axis)
      return avresult
    
def rolling_average_ndarray(arr, nrng, nstride, offs=0, axis=0, sparse_tr=False, messg=''):
    if sparse_tr:
      if isinstance(arr, list):
         if axis==2:
           print("Cannot use axis=2 for a list of sparse matrices, where sparse_tr=True!")
           return None
         
         avresults=[rolling_average_ndarray(arr1.todense(), nrng, nstride, offs=offs, axis=axis) for arr1 in arr]
         avres=np.array(avresults)
         print("avres.shape=", avres.shape)
         avresult = np.moveaxis(avres, 1, 0)
         print("avresult.shape=", avresult.shape)
         return avresult
      else:
        print("if using sparse trials raster arr needs to be a list!")
        return None
    else:
      arr = np.asarray(arr)
      arr = np.moveaxis(arr, axis, 0)
      if (nrng+offs)> arr.shape[0]:
        print("Unable to do rolling average! Reduce offs or nrng! offs=%d nrng=%d arr.shape[0]=%d" % ( offs, nrng, arr.shape[0]))
        return None
      if offs: arr = arr[offs:]
      # Calc the shape and strides for as_strided along the first dimension
      shape = ((arr.shape[0] - nrng) // nstride + 1, nrng) + arr.shape[1:]
      strides = (arr.strides[0] * nstride, arr.strides[0]) + arr.strides[1:]
      strided_view = as_strided(arr, shape=shape, strides=strides)
      avresult = strided_view.mean(axis=1)
      avresult = np.moveaxis(avresult, 0, axis)
      return avresult
    
def fold_trial_raster(tsr, ntimef, arange, astride=-1, offs=0):
  
  if False and use_sparse_tr:
      if isinstance(tsr, list):
         foldedres=[fold_trial_raster(tsr1, ntimef, arange, astride=astride, offs=offs) for tsr1 in tsr]
         fldres=np.array(foldedres)
         print("fldres.shape=", fldres.shape)
         keyboard('check the shape')
  else:
      if astride<1: astride=arange
      if  use_sparse_tr:
        ntrials = len(tsr)
        tdur, ncells = tsr[0].shape
      else: tdur, ncells, ntrials=tsr.shape
      rla=rolling_average_ndarray(tsr, arange, astride, axis=0, offs=offs, sparse_tr=use_sparse_tr)
      if ntimef>rla.shape[0]:
          print("ntimef=%d is to big for the first folded dimension averaged array with dimensions:" % ntimef, rla.shape)
          print("Reducing ntimef to the maximal allowable size: ntimef=%d" % rla.shape[0])
          ntimef=rla.shape[0]
      newXdata=rla[:ntimef,:,:].reshape([ntimef*ncells, -1]).T
      return newXdata, ntimef
  
def get_samptype_from_smptype(smptype, pstim1, pstim2):
  if smptype=='avr': samptype='n-twa_%d_%d' % (pstim2-pstim1, pstim1)
  elif smptype=='wsamp': samptype='n-twb_%d_%d' % (pstim2-pstim1, pstim1)
  elif smptype=='allsamp': samptype='n-twb'
  else: samptype='n-twb'

def vstimes2istim(frametimes, vstimes):
  istm=np.zeros(len(vstimes)).astype(int)
  for ivst1,vst1 in enumerate(vstimes):
    istm[ivst1]=np.abs(frametimes-vst1).argmin()
  dtlen=int(np.min(np.diff(istm)))
  return istm, dtlen

def stimts2istim(vstim, thresh=0.99):
   diffstim=np.diff(vstim)
   istim=np.where(diffstim>thresh)[0]+1
#   dtlen=int(np.median(np.diff(istim)))
   dtlen=int(np.min(np.diff(istim)))
   return istim, dtlen
 
def stimts2ijstim(vstim, thresh=0.9):
   diffstim=np.diff(vstim)
   istim=np.where(diffstim>thresh)[0]+1
   jstim=np.where(diffstim<(-thresh))[0]+1
#   dtlen=int(np.median(np.diff(istim)))
   dtlen=int(np.min(np.diff(istim)))
   return istim, jstim, dtlen

def plot_responses(dataarray, stim, icell=None, itrial=None, tplot=1, nclasses=8):
    npts, ncells, ntrials1, nclasses=dataarray.shape
    if icell is None and itrial is None:
       icell=stim
       scolors=list('krgbcmy')+[(1,0.6,0.2)]
       for iss in range(nclasses):
           clr=scolors[iss]
           plt.plot(dataarray[:,icell,:,iss].mean(axis=1), color=clr, linewidth=2, label='stim=%d' % iss)
           if tplot>0:
             for itr in range(ntrials1):
               plt.plot(dataarray[:,icell,itr,iss], color=clr, linewidth=0.5,linestyle='--')
       plt.legend()
       plt.show()
    elif isinstance(icell,int) and itrial is None:
      if tplot%10==1:
        mntrls=dataarray[:,icell,:,stim].mean(axis=1)
        plt.plot(mntrls, color='k', linewidth=3)
        if tplot>10:
          for itr in range(ntrials1):
            plt.plot(dataarray[:,icell,itr,stim], color=rand(3), linewidth=0.5)
        plt.show()
      elif tplot==2:
        mntrls=dataarray[:,icell,:,stim].mean(axis=1)
        mntrls[mntrls<0.0001]=0.0001
        plt.semilogy(mntrls, color='k', linewidth=3)
        for itr in range(ntrials1):
          vals=dataarray[:,icell,itr,stim]
          vals[vals<0.0001]=0.0001
          plt.plot(vals, color=rand(3), linewidth=0.5)
        plt.show()
      else:
        plt.imshow(dataarray[:,icell,:,stim])
        plt.show()
    elif isinstance(itrial,int) and icell is None:
        plt.imshow(dataarray[:,:,itrial,stim])
        plt.show()
    else:
        plt.plot(dataarray[:,icell,itrial,stim], color='k', linewidth=1.5)
        plt.show()

def get_data_info(dataname, datainfo='input_data.txt', spktype='spikes'):
  """ dataname is the abbreviated name for the data - label pairs
       use 'mix-'dname1'-'dname2' to use data from dname1 and labels from dname2
  """
  if isinstance(datainfo, str):
     datainfo=file2dict(datainfo)
#     print("datainfo=", datainfo)
     
  if 'mix-' in dataname:
    mx, datadata, labeldata = dataname.split('-')
    datafile=datainfo[datadata][0]
    labelfile=datainfo[labeldata][1]
    if len(datainfo[labeldata])>2: coordfile=datainfo[labeldata][2]
    else: coordfile='none'
    mode=data2mode(datadata)
  else:
    if dataname in datainfo:
      datafile, labelfile, coordfile = datainfo[dataname]
      mode=data2mode(dataname)
      if verbose>1: print("mode=", mode)
    else:
      print('Couldnt find dataname=', dataname)
  return datafile, labelfile, mode, coordfile

def guess_labels_file(matfilename, labelsinfo=[('normdata_', 'angleindex_531'),('VisOnly','NoOpto'),('.mat','.txt')]):
   if isinstance(labelsinfo, list):
      reprules=labelsinfo
      labelsfile=matfilename
      for rrule in reprules:
         labelsfile=labelsfile.replace(*rrule)
   elif isinstance(labelsinfo, str):
     labelsfile=labelsinfo # implement search for the most similar textfiles
   return labelsfile
  
def find_nearest_bin_indices(t_events, t_bins, when='after', return_offsets=False):
  """Finds the index of the bin in t_bins that occurs immediately after each event time in t_events.
  Args:
    t_events: A 1D numpy array of event times.
    t_bins: A 1D numpy array of bin times.
  Returns:
    A 1D numpy array of indices into t_bins, where each index corresponds to the bin
    that occurs immediately after the corresponding event time in t_events.
  """
  if when == 'before': side='left'
  else: side='right'
  ibns = np.searchsorted(t_bins, t_events, side=side)
  ibns[ibns == len(t_bins)] = -1  # Handle corner case where event is after last bin
  if return_offsets:
    tfirstbin=t_bins[ibns]
    toffsets = tfirstbin - t_events
    return ibns, toffsets
  else:
    return ibns

def find_nearest_bin_indices_slow(t_events, t_bins):
  ibins=[]
#  nevents=len(t_events)
  for te in t_events:
      igb=np.where(t_bins>te)[0]
      if len(igb)>0:
        ibins.append(np.min(igb))
      else:
        ibins.append(-1)
  return np.array(ibins, 'int')

def find_nearest_bin_indices2(t_events, t_bins, when='after'):
    """
    Find the indices of bins that occur immediately after each event time.
    Parameters:
    t_events (ndarray): 1D array containing times of events.
    t_bins (ndarray): 1D array containing times of sample bins.
    Returns:
    ndarray: 1D array containing indices of bins for each event.
    """
    if when == 'before': side='left'
    else: side='right'
    indices = np.searchsorted(t_bins, t_events, side=side)
    # Ensure indices are within the bounds of the array
    indices = np.clip(indices, 0, len(t_bins) - 1)
    # Check if the bin at the found index is actually after the event time
#    mask = t_bins[indices] >= t_events
    # Adjust indices where the bin is not actually after the event time
#    indices += ~mask
    return indices
  
#was load_mat_cell_arrays
def load_mat_ensemble_stim_data(filename, dataset=None):
    """Loads cell arrays from a MATLAB .mat file into lists of NumPy arrays and prepares the data for the perturbation
       analysis.
       filename: The path to the .mat file.
    """
    try:
      indata=sp.io.loadmat(filename)
    except:
      if dataset is None:
        with h5py.File(filename, 'r') as f:
          indata = {}
          for key, value in f.items():
              indata[key] = convert_to_numpy(value, f, key)
        matmode=0
      elif isinstance(dataset,(int,np.integer)):
        matmode=1
        keyboard('Do THIS outside the .mat file which is huge and takes a long time to load')
        sys.exit(0)
        acceptable_inds=np.array([2,3,4,6,7,8,35,36,37])-1
        if dataset in acceptable_inds:
          fulldata=load_mat_file_with_cell_arrays(filename)
          indata0=fulldata['Data'][0][dataset]
        else:
          print("We are not using this index for now!")
          return None
#    keyboard('check after 1')
    
# post-process the raw             
    data={}
    var1d=['optostimtimes', 'frametimes']
    array_vars=['target']
    
#    OLD_important_vars=['spk', 'prob', 'fluo', 'cellcoords', 'target', 'frametimes', 'optostimtimes', 'target_ensemble', 'StimOrder']
    important_vars=['spk', 'prob', 'fluo', 'cellcoords', 'target', 'frametimes', 'optostimtimes', 'target_ensemble', 'StimOrder', 'Nframes', 'framerate']
#    OLD_cell_vars=['spk', 'prob', 'fluo', 'cellcoords', 'frametimes', 'optostimtimes', 'StimOrder', 'target_ensemble']
    cell_vars=['spk', 'prob', 'fluo', 'cellcoords', 'frametimes', 'optostimtimes', 'StimOrder', 'target_ensemble', 'Nframes', 'framerate']
    array_vars=['target']
    for datname in cell_vars:
        if datname in var1d:
            data[datname]=[x.ravel() for x in indata[datname][0]]
        elif datname == 'StimOrder':
            ldata=[]
            igrp1=-1
            for ix, x in enumerate(indata[datname][0]):
              ldict={}
              try:
                if isinstance(x, dict):
                    if igrp1<0: igrp1=ix
                    ldict['targets']=x['targets'].ravel().astype(int)
                    ldict['Eind']=x['Eind'].ravel().astype(int)
                else:
                  ldict['targets']=None
                  ldict['Eind']=None
                ldata.append(ldict)
              except:
                print("There was an error for the element #%d of StimOrder" % ix)
                keyboard('Problem with StimOrder (Eind, targets)')
            data[datname]=ldata
            data['grp1indx']=igrp1
        elif datname in ['target_ensemble']:
           data[datname]=[x.astype(int) for x in indata[datname][0]]
        else:
#          keyboard('check %s' % datname)
           print("Loading data ", datname)
           if datname == 'Nframes':
             data[datname]=[x for x in indata[datname]]
           elif datname == 'framerate':
             keyboard('check')
             data['framerate']=[indata[datname] for _ in range(len(indata['spk'][0]))]
           else:
             data[datname]=[x for x in indata[datname][0]]

    for datname in array_vars:
        if datname in ['target']:
          data[datname]=indata[datname].astype(np.int32)
        else:
          data[datname]=indata[datname]
          
    data['StimInfo']=data['StimOrder']  # to have a universal StimOrder use StimInfo key! StimOrder differs from one experiment to other
#    keyboard('check after 2')
    return data

def can_convert_to_integer(arr, error_tolerance=1e-12):
    return np.all(np.abs(arr - np.round(arr)) < error_tolerance)

def convert_to_numpy(value, f, key, verbose=0):  # Pass the file object 'f'
    """Recursively converts HDF5 objects, until dereferenced."""
    if key=='highresp_trials':
      print("HERE HERE HERE HERE HERE HERE")
      print("value=", value)
    if isinstance(value, h5py.Dataset):
        if value.shape == (0, 1) or value.shape == (1, 0):  # Check for empty cell representation
            print("Hello 1 Hello 1\n"*30)
            return np.array([], dtype=value.dtype)
        if value.dtype == 'object':
           return [convert_to_numpy(cell, f, key, verbose=verbose) for cell in value]
        else:
           if verbose>1: print("key=", key, " value.dtype=", value.dtype)
#           if value.dtype!='f8': keyboard('check value')
           narr=np.array(value, dtype=value.dtype)
           if value.ndim==2:
             if value.shape == (0, 1) or value.shape == (1, 0):
               print("Hello 3 Hello 3\n"*30)
               narr=np.array([], dtype=value.dtype)
               return narr
             elif min(value.shape)==1:
               narr=np.array(value, dtype=value.dtype).ravel()
           if False and np.issubdtype(narr.dtype, np.floating) and can_convert_to_integer(narr):
             if verbose>0: print("Convert %s to integer array!" % key, " shape=", narr.shape)
             mxval=np.abs(narr).max()
             if mxval > 2147483646: return narr.astype(np.int64)
             elif np.abs(narr).max() > 32766: return narr.astype(np.int32)
             else: return narr.astype(np.int16)
           else:
             return narr
    elif isinstance(value, h5py.Group):
#OLD        return {k: convert_to_numpy(v, f, key) for k, v in value.items()}
        return {k: convert_to_numpy(v, f, k, verbose=verbose) for k, v in value.items()}
    elif isinstance(value, h5py.Reference):
        return convert_to_numpy(f[value], f, key, verbose=verbose)  # Recursive dereferencing
    elif isinstance(value, np.ndarray):
        if value.shape == (0, 1) or value.shape == (1, 0):  # Check for empty cell representation
            print("Hello 2 Hello 2\n"*30)
            return np.array([], dtype=value.dtype)
        if value.dtype == 'object':
            return [convert_to_numpy(cell, f, key, verbose=verbose) for cell in value]
          
    print("Returning raw value for key=", key)
    return value
                    
def load_mat_file_with_cell_arrays(filename, itemind=None, verbose=0):
    """Loads cell arrays from a MATLAB .mat file into lists of NumPy arrays and prepares the data for the perturbation
       analysis.
       filename: The path to the .mat file.
    """
    if itemind is None:
      with h5py.File(filename, 'r') as f:
        indata = {}
        for key, value in f.items():
            indata[key] = convert_to_numpy(value, f, key)
    else:
      f=h5py.File(filename, 'r')
      llist=[(kk,vv) for kk, vv in f.items()]
      ddds=llist[1][1][0]
      indata=convert_to_numpy(ddds[itemind],f, key=None, verbose=verbose)
    return indata

def load_npy_pst_data_file_A(filename, dataset=None):
    """Loads indata arrays from a numpy file (obtained from a bit .mat file)
      #2 A_1: 10 targets, 50 trials per target, 194 cells
      #3 A_2: 10 targets, ~77 trials per target, 164 cells
      #4 A_3: 9 targets, ~95 trials per target, 117 cells
      #6 A_5: 10 targets, ~75 trials per target, 186 cells
      #7 A_6: 7 targets, ~198 trials per target, 150 cells
      #8 A_7: 7 targets, 196 trials per target, 164 cells
      #35 A_34: 10 targets, ~149 trials per target, 127 cells
      #36 A_35: 10 targets, ~149 trials per target, 299 cells
      #37 A_36: 10 targets, 149 trials per target, 281 cells
    """
    indata=loadnpy(filename)
    # post-process the raw             
    data={}
    direct_vars=['spk', 'prob', 'fluo', 'cellcoords', 'frametimes', 'Nframes', 'framerate']
    for impvar in direct_vars:
      if impvar in indata:
        data[impvar]=[indata[impvar]]
      else:
        data[impvar]=[]
    
    var1d=['optostimtimes', 'frametimes']
    for impvar in var1d:
      if impvar in indata:
        data[impvar]=[indata[impvar].ravel()]
      else:
        if impvar == 'frametimes': # this is because frametimes was missing in some datasets
          print('frametimes missing! Setting frames_times based on Nframes and framerate in indata')
          data[impvar]=[np.arange(indata['Nframes'])/indata['framerate']]

    data['ntargets']=len(indata['target'])
    data['target']=indata['target'][:,None]
    new_vars=['StimOrder', 'target_ensemble'] # Use StimInfo from now on
    data['target_ensemble']=[np.arange(data['ntargets'])+1]
    data['grp1indx']=0
    data['StimInfo']=[{}]
    data['StimInfo'][0]['Eind']=indata['StimOrder']['ROI_ind'].ravel().astype('int32')
    data['StimInfo'][0]['targets'] = None # change this later; need clarification from Tiago
    data['other']={}
    for kyind in indata:
      if kyind not in data:
        data['other'][kyind]=indata[kyind]
    return data
  
def read_matlab_vistim_file(datafilename, mode=1, spktype='spikes'):
#  matdata=sp.io.loadmat(matfilename)
    hdf5dict=h5py.File(datafilename, 'r')
    print(hdf5dict.keys())
    dictvals={}
    for ky in hdf5dict.keys():
      dictvals[ky]=hdf5dict[ky][()]
    if mode==1:
      rfdata=dictvals[spktype]
      vistim=dictvals['visualstimulus'].ravel()
      frate=dictvals['framerate'].ravel()[0]
    elif mode==2:
      rfdata=dictvals[spktype]
      vistim=dictvals['onset']
      frate=dictvals['framerate'].ravel()[0]
    else:
      raise ValueError('Unknown mode for reading matlab vistim file!')
    return rfdata, vistim, frate, dictvals
   
def read_npy_file(datafilename, labelsfile='', mode=1):
      rfdata, vistim, frate = myload(datafilename)
      dictvals={}
      dictvals['spikes']=rfdata
      dictvals['vistim']=vistim
      dictvals['framerate']=frate
      return rfdata, vistim, frate, dictvals

def get_dataA(datafilename, labelsfile='', smode=1, spktype='spikes'):
#  matdata=sp.io.loadmat(matfilename)
    froot,fext = os.path.splitext(datafilename)
    if smode==1:
      if fext == '.mat':
        rfdata, vistim, frate, dictvals = read_matlab_vistim_file(datafilename, mode=1, spktype=spktype)
      elif fext == '.npy':
        rfdata, vistim, frate, dictvals = read_npy_file(datafilename, mode=1)
    elif smode==2:
       # deal with this directly from the VisStimData class
      print('NOT IMPLEMENTED FOR SMODE=2! Use get_dataXXX() for data visXXX#')
      pass
    
    try:
        if 'matfile:' in labelsfile:
          print("dictvals.keys()=", dictvals.keys())
          if smode==2:
            labels=dictvals['angleindex'].astype(int)
          else:
            raise ValueError('Specify labels for sub-mode=%d!' % smode)
        else:
          labels=np.loadtxt(labelsfile, dtype=int)
    except:
        labels=[]
#       dictvals = sp.io.loadmat('contrast_wave/817_22.10.10_1_thrS_z0.mat')
#       frametimes=matfiledd['frametimes'].ravel()
#       vistim=matfiledd['visstimtimes'].ravel()
#       probs=matfiledd['prob']
#       spk=matfiledd['spk']
#       avstarts=matfiledd['av_start']

    istim, jstim, dtlen=stimts2ijstim(vistim)
    vistiminfo=[istim, jstim, dtlen, 20, vistim]
    return rfdata, vistiminfo, frate, labels, dictvals

def get_dataB(datafilename, labelsfile='', smode=1, spktype='spk'):
#        dd = sp.io.loadmat('data1/contrast_wave/817_22.10.10_1_thrS_z0.mat')
    if len(datafilename)<8:
      dataspec=datafilename
      datafilename,labelsfile,dmode,coordfile=get_data_info(datafilename)
#    avalinfo=AvalancheInfo(dataspec)
    if spktype in ['def', 'default']: spktype='spikes'
    
    if smode==1:
      dd = myloadmat(datafilename)
      rfdata=dd[spktype]
      rfdata[np.isnan(rfdata)]=0
      nframes1=int(dd['Nframes'].ravel()[0])
      assert nframes1 == rfdata.shape[0]
      istim=dd['onset'].ravel().astype(int)
      stimlabels=dd['angleindex'].ravel().astype(int)
#      iireps=np.where(onsets[:-1]==onsets[1:])[0]
      diffs=np.diff(istim)
      dtlen=int(diffs[diffs>0].min())
      iireps=np.where(diffs==0)[0]
      idiff=np.where(diffs>0)[0]
      nistim=np.append(istim[idiff], istim[-1])
      if False:
        for iii1,i1 in enumerate(iireps):
          print('At %d = %d %d' % (i1, istim[i1], stimlabels[i1]))
          print('istims: %d vs %d ' % (istim[i1], istim[i1+1]))
          print('labels: %d vs %d ' % (stimlabels[i1], stimlabels[i1+1]))
          print(' ... ', stimlabels[(i1-2):(i1+4)], ' ... ')
          print(' ... ', istim[(i1-2):(i1+4)], ' ... ')
          print(' ... ', nistim[(i1-2-iii1):(i1+4-iii1)], ' ... ')
  
      istim=nistim[(nistim+dtlen)<nframes1]
      stimlabels=stimlabels[:len(istim)]
      frate=dd['framerate'].ravel()[0]
      frametimes=None
      frametimes=1000.*np.arange(nframes1)/frate # frametimes in ms
      vstim=None
    jstim=-1
    maxrep=-1
    vistiminfo=[istim, jstim, dtlen, frate, maxrep, frametimes, vstim]
    return rfdata, vistiminfo, stimlabels, dd

def get_data_cw(datafilename, labelsfile='', smode=1, spktype='spk'):
#        dd = sp.io.loadmat('data1/contrast_wave/817_22.10.10_1_thrS_z0.mat')
    if len(datafilename)<8:
      dataspec=datafilename
      datafilename,labelsfile,dmode,coordfile=get_data_info(datafilename)
#    avalinfo=AvalancheInfo(dataspec)
#    keyboard('check avalinfo at first')

    if spktype in ['def', 'default', 'spikes']: spktype='spk'
    
    if smode==1:
        dd = sp.io.loadmat(datafilename)
        if spktype not in dd:
          if spktype in ['fluo', 'fluor']:
            if 'fluo' in dd: spktype='fluo'
            elif 'fluor' in dd: spktype='fluor'
        frametimes=dd['frametimes'].ravel()
        vstim=None
        vstimes=dd['visstimtimes'].ravel()
        istim, dtlen = vstimes2istim(frametimes, vstimes)
#        probs=dd['prob']
#        fluor=dd['fluor']
#        spk=dd['spk']
        rfdata=dd[spktype].T
        stimlabels=dd['dirindex'].ravel()-1
        nframes1=dd['Nframes'].ravel()[0]
#        frmtimes=
        assert nframes1 == rfdata.shape[0]
        frate=(nframes1-1)/(frametimes[-1]-frametimes[0])
    jstim=-1
    maxrep=-1
    vistiminfo=[istim, jstim, dtlen, frate, maxrep, frametimes, vstim]
    return rfdata, vistiminfo, stimlabels, dd

#def simdata2tsimage(rfdata, labels, cellids, trialids, ulabels):
def simdata2tsimage(rfdata, trialids):
  alldata=[]
  for itrial in range(max(trialids)+1):
     alldata.append( rfdata[trialids==itrial, :] )
  return np.concatenate(alldata,axis=1)

def generate_fake_trialindexed_rfdata(nstims=20, ntbins=10, ncells=7, celloffset=0):
  rfdata1=np.repeat(np.repeat(np.arange(nstims)+1,ntbins)[:,None], ncells, axis=1)
  rfdata=np.concatenate([[np.zeros(ncells)], rfdata1])
  if celloffset:
    for icell in range(ncells):
      rfdata[:,icell]+= icell*celloffset
  vstim=np.zeros(nstims*ntbins+1)
  vstim[[i*ntbins+1 for i in range(nstims)]]=1
  return rfdata, vstim

def generate_fake_trialindexed_xdata(nrepeats=10, ntbins=1, nclasses=4, ncells=5, celloffset=0, order='c', label_cell=True):
  """
  order: specify order of trials:
            'c': vary the class first, then repeats
            'r': vary the repeats first, then class
            's': shuffle and present them randomly
  label_cell: include also the labeling of the cell

  """
  nstims=nrepeats*nclasses
  if order=='c': trials=np.tile(np.arange(nclasses),nrepeats)
  elif order=='r': trials=np.repeat(np.arange(nclasses),nrepeats)
  elif order=='s':
    trials=np.repeat(np.arange(nclasses),nrepeats)
    np.random.shuffle(trials)
  else:
    print("Unknown ordering")
  ntrials=len(trials)

  assert ntrials == (nclasses*nrepeats)
  nbins=ntbins*nstims
  Xdata=np.zeros((nbins,ncells), dtype=object)
  print("trials=", trials)
  ydata=np.repeat(trials, ntbins)
  print("ydata=", ydata)
  trcounts=np.zeros(nclasses)
  for itr, clss in enumerate(trials):
    trcounts[clss]+=1
    ixst=ntbins*itr
    ixend=ntbins*(itr+1)
    if label_cell:
      for icell in range(ncells):
        strng='%d-%d-%d' % (trcounts[clss], clss, icell)
        Xdata[ixst:ixend, icell]=strng
    else:
      strng='%d-%d' % (trcounts[clss], clss)
#     print("strng=", strng)
#     Xdata[ixst:ixend,:]=np.repeat(np.array([]) for _ in range(ntbins)]), ncells, axis=1)
      Xdata[ixst:ixend,:]=strng
  return Xdata, ydata

def trial_shuffle_matrix_rows(array_2d, trial_id):
    """
    Shuffle rows in 2d array matrix, corresponding to the same ids in trial_id.
    Parameters:
    - array_2d: 2D numpy array with shape (ndur, ntid)
    - trial_id: 1D numpy array with ntid trials ids
    Returns:
    - A new 2D numpy array with rows shuffled within the same id group.
    """
    unique_ids = np.unique(trial_id)  # Find unique IDs
    if array_2d.shape[0] != len(trial_id):
      keyboard('check shape and len(trial_id)')
    shuffled_array = array_2d.copy()  # Create a copy to shuffle
    for unique_id in unique_ids:
        indices = np.where(trial_id == unique_id)[0]
        try:
          rows_to_shuffle = array_2d[indices]
        except:
          print("indices=", indices)
          keyboard('check indices for row shuffle')
          
        np.random.shuffle(rows_to_shuffle)
        shuffled_array[indices] = rows_to_shuffle
    return shuffled_array

  
def trial_shuffle_matrix_columns(array_2d, trial_id):
    """
    Shuffle rows in 2d array matrix, corresponding to the same ids in trial_id.
    Parameters:
    - array_2d: 2D numpy array with shape (ndur, ntid)
    - trial_id: 1D numpy array with ntid trials ids
    Returns:
    - A new 2D numpy array with columns shuffled within the same id group.
    """
    unique_ids = np.unique(trial_id)  # Find unique IDs
    if array_2d.shape[1] != len(trial_id):
      keyboard('check shape and len(trial_id)')
    shuffled_array = array_2d.copy()  # Create a copy to shuffle
    for unique_id in unique_ids:
        indices = np.where(trial_id == unique_id)[0]
        try:
          columns_to_shuffle = array_2d[:, indices]
        except:
          print("indices=", indices)
          keyboard('check indices for column shuffle')
          
        np.random.shuffle(columns_to_shuffle.T)
        shuffled_array[:, indices] = columns_to_shuffle
    return shuffled_array
  
def trial_shuffle_raster(trialraster3d, trial_id):
  assert trialraster3d.ndim == 3
  ndur, ncells, ntrials = trialraster3d.shape
  shuffraster3d=np.zeros_like(trialraster3d)
  for icell in range(ncells):
    shuffraster3d[:,icell,:]=trial_shuffle_matrix_columns(trialraster3d[:,icell,:], trial_id)
  return shuffraster3d

def trial_shuffle_Xdata(Xuse, yuse, ntbins=1, retmat=False, rndseed=-1):
    if ntbins != 1: raise ValueError("Shuffle of time-series Xdata not implemented yet!")
    if rndseed>0: np.random.seed(rndseed)
    nsamp, nf = Xuse.shape
    Xshuffled=Xuse.copy()
    yshuffled=yuse.copy() # it is not going to change after shuffle; just use for now for testing, that shuffling d
    uclasses, ccounts = np.unique(yshuffled, return_counts=True)
    ntrialsmax=max(ccounts)
    nclasses=len(uclasses)
    shuffle_matrix=-np.ones((nf, nclasses, ntrialsmax), dtype='int32')
    for iclss, (clss, nclss) in enumerate(zip(uclasses, ccounts)):
      clssindx=np.where(yshuffled == clss)[0]
      assert len(clssindx)==nclss
      for ifeature in range(nf):
          iperm=np.random.permutation(nclss)
          shuff_clssindx=clssindx[iperm]
          shuffle_matrix[ifeature, iclss, :nclss]=shuff_clssindx
          Xshuffled[shuff_clssindx, ifeature]=Xuse[clssindx,ifeature]
          yshuffled[shuff_clssindx]=yuse[clssindx]
          assert all(yshuffled[shuff_clssindx]==clss)
    assert all(yshuffled==yuse)
    if retmat: return Xshuffled, shuffle_matrix 
    else: return Xshuffled
  
def trial_shuffle_rawdata(rfdata, vstim, stimlabels=None, retmat=False):
  raise ValueError("Don't use trial_shuffle_rawdata for now ... needs correction!")
  ntt,ncells=rfdata.shape
  istim, dtlen=stimts2istim(vstim)
  nstims=len(istim)
#  trshuffrfdata=np.zeros_like(rfdata)
  trshuffrfdata=rfdata.copy()
  if stimlabels is None:                                               
    if retmat: stimmatrix=np.zeros((ncells,nstims), dtype='int32')
    for icell in range(ncells):
      iperm=np.random.permutation(nstims)
      if retmat: stimmatrix[icell, :]=iperm
      for iis, ist in enumerate(istim):
        istp=istim[iperm[iis]]
        trshuffrfdata[ist:(ist+dtlen),icell]=rfdata[istp:(istp+dtlen),icell]
  else:
    ulabels, nlabels = np.unique(stimlabels, return_counts=True)
    nclasses=len(ulabels)
    nstim1=max(nlabels)
    if retmat: stimmatrix=np.zeros((ncells,nclasses,nstim1), dtype=stimlabels.dtype)
    for icell in range(ncells):
      for ilbl, (lbl, nlbl) in enumerate(zip(ulabels, nlabels)):
        lblindx=np.where(stimlabels==lbl)
        iperm=np.random.permutation(nlbl)
        stimmatrix[icell, ilbl, :]=stimlabels[lblindx[iperm]]
        keyboard('You are using rawdata shuffle! Needs corrections to work properly')
        for iis, ist in enumerate(istim[lblindx]):
          istp=istim[lblindx[iperm[iis]]]
          trshuffrfdata[istp:(istp+dtlen),icell]=rfdata[istp:(istp+dtlen),icell]
  if retmat: return trshuffrfdata, stimmatrix
  else: return trshuffrfdata

def  datablock2sample(curblock, sampname):
    if False:
      print("curblock.shape=", curblock.shape)
      print("sampname=", sampname)
      keyboard('check curblcok')
      
    if sampname=='ntw-twb': return curblock.ravel()[None,:]
    elif sampname=='n-twb': return curblock
    elif sampname=='n-twa': return curblock.mean(0)[None,:]
    else:
      return None
                 
# plot the data
def plot_spike_data_basic(spikedata, visualstimulus, frate, ntimepts=-1, tpstart=0, vmin=0, vmax=1, gamma=0.4, savefig=''):
  fig = plt.figure(figsize=(16,8))
  spec = gridspec.GridSpec(ncols=1, nrows=2, hspace=0.0, height_ratios=[6,1])
  if ntimepts<0: ntimepts=spikedata.shape[0]
  nanpos=np.where(np.isnan(spikedata))
  spikedata[nanpos]=0.
  tpt1=tpstart
  tpt2=tpstart+ntimepts
  xtime=(np.arange(ntimepts)+tpstart+0.5)/frate
  ax0 = fig.add_subplot(spec[0])
  ax0.imshow((spikedata[tpt1:tpt2,:].T)**gamma, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
  ax0.set_ylabel("Spikes", fontsize=18)
  ax0.get_xaxis().set_visible(False)
  ax1 = fig.add_subplot(spec[1])
  ax1.plot(xtime,visualstimulus[tpt1:tpt2],'r')
  ax1.set_xlabel("Time [sec]", fontsize=16)
  ax1.set_ylabel("Stimulus", fontsize=14)
  ax1.set_xlim([xtime[0],xtime[-1]])
  ax1.set_ylim([-0.01, 1.1])
  if savefig:
    plt.savefig(savefig)
  plt.show()

def sort_cells_by_selectivity(Xss, yss, sortby='lda1', nclasses=None, psarray=None, ulabels=None, Xtest=None, ytest=None, lsdim=3, nstimclasses=None, retsortvals=False):
    
    if lsdim>0 and sortby in ['lda1', 'lda2', 'shf1', 'shf2']:
      try:
        nlclasses = len(np.unique(yss))
        if lsdim >= nlclasses: dimredalg = LinearDiscriminantAnalysis(n_components=nlclasses-1)
        else: dimredalg = LinearDiscriminantAnalysis(n_components=lsdim)
        dimredalg.fit(Xss,yss)
        ldata=dimredalg.transform(Xss)
        ldacoefs=dimredalg.coef_
        lcoeff=np.abs(ldacoefs).sum(0)
      except:
        keyboard('dimredalg problem inside sort_cells_by_selectivity')
    
    #plot LDA analysis
    if lsdim in [2,3] and sortby in ['lda1'] and False:
      colrs=list('rgbcmyk')+[(1.,0.7,0.1)]
    
      ldata=dimredalg.transform(Xss)
      plot_reduced_dim(ldata, yss, 8, dim=lsdim, subplot=121, loc=1, colors=colrs, title="Train data", show=False)
    
      ldatatest=dimredalg.transform(Xtest)
      plot_reduced_dim(ldatatest, ytest, 8, dim=lsdim, subplot=122, loc=1, colors=colrs, title="Test data", show=True)

    if sortby in ['lda1']:
      isortlda1=np.argsort(np.abs(lcoeff))[::-1]
      if verbose>1: print("lcoeff[isorted]=", lcoeff[isortlda1])
      isorted=isortlda1
      
    t0=time.time()
    if ( sortby in ['dsi', 'osi']):
      if len(psarray)==0:
        keyboard('must declare psarray for dsi')
      else:
        t0=time.time()
        DSI, OSI, PD = get_angle_selectivity(psarray)
        print("Time for DSI OSI CALCS=", time.time()-t0)

    sortvals=None
    if sortby=='dsi':
      sortvals=np.abs(DSI)
    elif sortby=='osi':
      sortvals=np.abs(OSI)
    elif sortby=='lda1':
      sortvals=np.abs(lcoeff)
    elif sortby=='lda2':
      sortvals=np.abs(lcoeff)[::-1]
    elif sortby in ['shf1', 'shf2']:
      sortvals=np.abs(lcoeff)
    elif sortby=='rnd1':
      isorted=np.random.permutation(self.nfeatures)
      
    if sortvals is not None:
      isorted=np.argsort(sortvals)[::-1]
      if verbose>1: print("Sorted values for %s" % sortby, sortvals[isorted])

    if retsortvals:
      return isorted, sortvals
    else:
      return isorted

def get_per_class_accuricies(train, test):
    pass
    krkrk

  
def get_impurity_function(criterion):
    def impf_gini(v):
        p = v/v.sum()
        return np.multiply(p, 1-p).sum()
    def impf_entropy(v):
        return sp.stats.entropy(pk=v)
    def impf_misclassification(v):
        p = v/v.sum()
        return 1 - p.max()
    if criterion == 'gini': return impf_gini
    elif criterion == 'entropy': return  impf_entropy
    elif criterion == 'misclassification': return impf_misclassification
    else: raise ValueError("Unknown impurity criterion: %s" % criterion)
  
def get_tree_pcfim(tree_clf):
    """
    get the per-class feature importance matrix (PCFIM)
    Arguments:
        tree_clf - The random forest, xgboost, or other classifier to calculate the importance matrix for.
        importance_matrix - The importance matrix with the importance of each predictor in predicting a class.
        rows are different classes, columns different features
    """
    # get the number of classes being predicted by the random forest
    classes = tree_clf.classes_
    n_classes = len(classes)
    importance_matrix = []

    for dec_tree in tree_clf.estimators_:
        criterion = dec_tree.get_params()['criterion']
        impurity_function=get_impurity_function(criterion)
        feature = dec_tree.tree_.feature
        n_features = dec_tree.tree_.n_features
        n_nodes = dec_tree.tree_.__getstate__()['node_count']
        nodes = dec_tree.tree_.__getstate__()['nodes']
        parent_node_ind = -np.ones(shape=n_nodes, dtype='<i8')
        #parent_node_ind[0] = n_nodes + 1
        #print(parent_node_ind)
        for par_ind,node in enumerate(nodes):
            if node[0] != -1:
                parent_node_ind[node[0]] = par_ind
            if node[1] != -1:
                parent_node_ind[node[1]] = par_ind

        # identify the leaves of the tree
        is_leaves = np.array([node[0]==-1 and node[1]==-1 for node in nodes])
        leaves_index = np.nonzero(is_leaves)[0]

        values_sorted = dec_tree.tree_.__getstate__()['values']
        node_pred = np.argmax(values_sorted[:,0,:], axis=1)
        leaves_class_index = node_pred[is_leaves]
        node_unvisited = np.ones((n_classes, n_nodes), dtype=bool)
        tree_importances = np.zeros((n_classes, n_features))
        for leaf_i,leaf_c_i in zip(leaves_index,leaves_class_index):
            current_i = parent_node_ind[leaf_i]
            while current_i != -1 and node_unvisited[leaf_c_i,current_i]:
                current_node = nodes[current_i]
                left_node = nodes[current_node['left_child']]
                right_node = nodes[current_node['right_child']]
                current_feature = current_node['feature']
                
                current_values = values_sorted[current_i,0,:]
                left_values = values_sorted[current_node['left_child'],0,:]
                right_values = values_sorted[current_node['right_child'],0,:]

                current_values_class = np.array([
                    current_values[leaf_c_i],
                    current_values[np.arange(len(current_values)) != leaf_c_i].sum()
                ])
                left_values_class = np.array([
                    left_values[leaf_c_i],
                    left_values[np.arange(len(left_values)) != leaf_c_i].sum()
                ])
                right_values_class = np.array([
                    right_values[leaf_c_i],
                    right_values[np.arange(len(right_values)) != leaf_c_i].sum()
                ])

                tree_importances[leaf_c_i,current_feature] += (
                        current_node['weighted_n_node_samples'] * impurity_function(current_values_class) -
                        left_node['weighted_n_node_samples'] * impurity_function(left_values_class) -
                        right_node['weighted_n_node_samples'] * impurity_function(right_values_class)
                        )
                node_unvisited[leaf_c_i,current_i] = False
                current_i = parent_node_ind[current_i]
                #print('next current is ', current_i)
        importance_matrix.append(tree_importances/nodes[0]['weighted_n_node_samples'])

    # average the predictor importances for each class by all of the trees in the forest
    importance_matrix = np.mean(importance_matrix, axis = 0)
    #should we normalize importance over each class
    importance_matrix = (importance_matrix.T / np.sum(importance_matrix, axis=1)).T
    return(importance_matrix)

def get_ovr_pcfim(clf_model, X_train, y_train, n_classes=None):
  if n_classes is None:
     ulbls=np.unique(y_train)
     n_classes=len(ulbls)
  from sklearn.multiclass import OneVsRestClassifier
  ovr_classifier = OneVsRestClassifier(clf_model)
  ovr_classifier.fit(X_train, y_train)
  class_feature_importances = []
  for i in range(n_classes):
    class_importances = ovr_classifier.estimators_[i].feature_importances_
    class_feature_importances.append(class_importances)

  # Create a DataFrame with feature importances for each class
#  importance_df = pd.DataFrame({f'Class {i}': imp for i, imp in enumerate(class_feature_importances)}, index=feature_names)
  return np.array(class_feature_importances)

def get_binary_pcfim(clf_model, X_train, y_train, n_classes=None):
    if n_classes is None:
      ulbls=np.unique(y_train)
      n_classes=len(ulbls)
      
    class_feature_importance = np.zeros((n_classes, X_train.shape[1]))
    for class_idx in range(n_classes):
        binary_y_train = np.where(y_train == class_idx, 1, 0)
        # Train a binary classifier
        clf_model.fit(X_train, binary_y_train)
        # Get feature importances for the class
        class_feature_importance[class_idx, :] = clf_model.feature_importances_
    return class_feature_importance

def get_pcfim(model, X_train, y_train, pcfim=1, n_classes=None, modelname=''):
  from xgboost import XGBClassifier
  from sklearn.ensemble import RandomForestClassifier
  if isinstance(model,str):
     if len(model)>2:
       try: nest=int(model[2:])*1000
       except:
         if model[3]=='p': nest=int(model[3:])*100
         else: nest=500
     if model[:2]=='xg':
        clf=XGBClassifier(objective='binary:logistic', n_estimators=nest)
        if pcfim==1: pcfim=2
     elif model[:2]=='rf':
        clf=RandomForestClassifier(n_estimators=nest)
        if pcfim==1: clf.fit(X_train, y_train)
  else:      
    if modelname:
      if modelname[:2]=='xg':
        clf = XGBClassifier(objective='binary:logistic', n_estimators=model.n_estimators)
        if pcfim==1:
          print("Cannot use pcfim=1 (tree) for XGBoost classifier")
          pcfim=2
      elif modelname[:2]=='rf':
        clf=model
    else:
        clf=model

  if pcfim==1:
    return get_tree_pcfim(clf)
  elif pcfim==2:
    return get_binary_pcfim(clf, X_train, y_train)
  elif pcfim==3:
    return get_ovr_pcfim(clf, X_train, y_train)

def test_pcfim(clfier='rf', pcfim=1, data=1):
  from xgboost import XGBClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.datasets import make_classification
  from sklearn.datasets import load_iris
  
  if data==1:
    XX, iyy = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=3, random_state=42)
  elif data==2:
    XX=np.random.randint(0,20,size=(1000,7))
    wts=np.array([0.1,0.2,1., 3., 2., -5., 0.0])
    yy=(XX @ wts)/wts.sum()
    yy=np.round(3*yy/yy.max()).astype('uint16')
    yyvals,iyy=np.unique(yy, return_inverse=True)
  elif data==3:
    iris = load_iris()
    XX, iyy = iris.data, iris.target
  else:
    XX, iyy = make_classification(n_samples=1000, n_features=20, n_informative=data, n_classes=3, random_state=42)
    
  if clfier=='rf': clf= RandomForestClassifier(n_estimators=500)
  elif clfier=='xg':
    clf = XGBClassifier(objective='binary:logistic', n_estimators=500)
#    clf = XGBClassifier(objective='multi:softmax', n_estimators=500)
    if pcfim==1:
      pcfim=2
      print("Cannot use tree_pcfim with xgboost! Using pcfim=%d" % pcfim)
  else:
    print("Unknown classifier:", clfier)
    return
  
  if pcfim==1: clf.fit(XX,iyy)
  print("Calculating pcfim")
  if pcfim==1: pcfim=get_tree_pcfim(clf)
  elif pcfim==2: pcfim=get_binary_pcfim(clf, XX, iyy)
  elif pcfim==3: pcfim=get_ovr_pcfim(clf, XX, iyy)
  else:
    print("Must use 1 or 2 for now!")
    return
  plt.imshow(pcfim)
  plt.show()
  keyboard('check')

def tryany(x):
   try:
      rv=int(x)
   except:
     try:
        rv=float(x)
     except:
        rv=x
   return rv

def center_class_means(X, y, cmean=0):
    X_cent= np.zeros_like(X)
    for i in np.unique(y):
        X_cent[y == i] = X[y == i] - np.mean(X[y == i], axis=0)
    return X_cent
  
