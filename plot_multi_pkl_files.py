#!/usr/bin/env python

from sputils import *

nargs=len(sys.argv)

rezdirs=['dropping_accuracy_results','daf_results', 'dafm2109_results']
nmulticurves=(nargs-1)//3
labeldict={'B09P33': 'SubC', 'B10066P026': 'Crit',  'B101P00045': 'SupC'}
dirlabel={0: 'All', 1: 'Filter', 2: 'Filt/Subsamp'}

if nargs<3 or int((nargs-1)/3) != nmulticurves:
   print('Usage: '+sys.argv[0]+' irezdir1 BPname1 thresh1  irezdir2 BPname2 thresh2 ...')
   sys.exit(0)

allmulticurves=[]
mclabels=[]

for icrv in range(nmulticurves):
    idir=int(sys.argv[icrv*3+1])
    BPname=sys.argv[icrv*3+2]
    thresh=sys.argv[icrv*3+3]
    replicate_curves=[]
    if thresh.lower()=='none':
       thresh=None
    for isimrep in range(10):
      npygzfilename='SimData300ctx'+BPname+'_%d.npy.gz' % (isimrep)
      for imlrep in range(100):
         if thresh is None:
            pklfilename='%s/SimData300ctx%s_%d.npy.gz_%d.pkl' % (rezdirs[idir],BPname, isimrep, imlrep)
         else:
           pklfilename='%s/SimData300ctx%s_%d.npy.gz_%s_%d.pkl' % (rezdirs[idir], BPname, isimrep, thresh, imlrep)
         if os.path.exists(pklfilename):
              with open(pklfilename, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                   try:
                      aa=data[npygzfilename][imlrep]['accuracy']
                   except:
                      aa=data[npygzfilename][imlrep+100]['accuracy']
                   replicate_curves.append(aa)
         else:
            print("File %s does not exist! Skipping" % pklfilename)
            
    #              for mouse, sessions in data.items():
    #              average them
    nrepcurves=len(replicate_curves)
    if nrepcurves>2:
       allmulticurves.append(np.array(replicate_curves))
       if thresh is None:
          mclabels.append('%s-all' % BPname)
       else:
          if idir==2: addstr='sample matched'
          else: addstr=''
          mclabels.append('%s-T%s %s' % (labeldict[BPname],thresh, addstr))
    else:
       print("Didn't find sufficient number (>2) of curves for %s T=%s in %s" % (BPname, str(thresh), rezdirs[idir]))
       
nmulticurves2=len(allmulticurves)
print("nmulticurves2=", nmulticurves2)

assert nmulticurves == nmulticurves2
if nmulticurves<2:
  curves=allmulticurves[0]
  nreps,nfeatures=curves.shape
  print("nreps=", nreps)
  print("nfeatures=", nfeatures)
  mncurve=curves.mean(0)
  sevals=curves.std(0)/np.sqrt(nreps)
  print("mncurve[0:5]=", mncurve[0:5])
  print("sevals[:10].mean()",sevals[:10].mean())
  plt.errorbar(np.arange(nfeatures), mncurve, yerr=sevals)
  plt.savefig(BPname+'_filt%s_dropping_accuracy_se.pdf' % thresh)
  plt.show()
  plt.plot(mncurve*100.)
  plt.plot([0,288],[10,10],'--',lw=1)
  plt.ylim(9.3,15)
  plt.xlim(0,285)
  plt.ylabel("Accuracy [%]")
  plt.savefig(BPname+'_filt%s_dropping_accuracy_mean.pdf' % thresh)
  plt.show()
else:
   
  useserr=False
  for icrv,curves in enumerate(allmulticurves):
    print("Processing curve=", icrv )
    clabel=mclabels[icrv]
    nreps,nfeatures=curves.shape
    print("nreps=", nreps)
    print("nfeatures=", nfeatures)
    mncurve=curves.mean(0)
    if useserr:
      sevals=curves.std(0)/np.sqrt(nreps)
      print("mncurve[0:5]=", mncurve[0:5])
      print("sevals[:10].mean()",sevals[:10].mean())
      plt.errorbar(np.arange(nfeatures), mncurve, yerr=sevals, label=clabel)
    else:
      plt.plot(mncurve*100., label=clabel)

  plt.plot([0,288],[10,10],'--k',lw=1)
  plt.ylim(9.3,15)
  plt.xlim(0,285)
  plt.ylabel("Accuracy [%]")
       
  if useserr:
    plt.legend()
    plt.savefig(BPname+'_multiplot%d_dropping_accuracy_se.pdf' % nmulticurves)
    plt.show()
  else:
    plt.legend()
    plt.savefig(BPname+'_multiplot%d_dropping_accuracy_mean.pdf' % nmulticurves)
    plt.show()

