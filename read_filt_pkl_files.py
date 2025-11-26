#!/usr/bin/env python

from sputils import *

nargs=len(sys.argv)

threshs=['1.0']

if nargs<2:
   print('Usage: '+sys.argv[0]+' BPname threshs...')
   BPname="B03P223"
   BPname="B05P160"
else:
   BPname=sys.argv[1]

iarg=2
if nargs>iarg: threshs=sys.argv[iarg:]
iarg+=1

rezdir='dropping_accuracy_filtered_results'
#rezdir='daf_results'
print("rezdir=", rezdir)
print("threshs=", threshs)

nthresh=len(threshs)

alltcurves={}
for thresh in threshs:
  alltcurves[thresh]=[]
  for isimrep in range(10):
    npygzfilename='SimData300ctx'+BPname+'_%d.npy.gz' % (isimrep,)
    for imlrep in range(100):
       pklfilename='%s/SimData300ctx%s_%d.npy.gz_%s_%d.pkl' % (rezdir, BPname, isimrep, thresh, imlrep)
       if os.path.exists(pklfilename):
            with open(pklfilename, 'rb') as f:
              try:
                 data = pickle.load(f)
              except:
                 print("Cannot find this replicate %d - %d" % (isimrep, imlrep))
                 continue
              if isinstance(data, dict):
                 subdatadict=data[npygzfilename]
                 if imlrep in subdatadict:
                   aa=subdatadict[imlrep]['accuracy']
                 elif (imlrep+100) in subdatadict:
                    print("For original thresh=%s isimrep=%d  imlrep=%d ADDING 100" % (thresh, isimrep, imlrep))
                    aa=data[npygzfilename][imlrep+100]['accuracy']
                 else:
                    print("Failed to find data for thresh=%s isimrep=%d  imlrep=%d! Adding 100" % (thresh, isimrep, imlrep))
                    continue
                 alltcurves[thresh].append(aa)
#              for mouse, sessions in data.items():
#              average them
  
if nthresh<2:
  curves=np.array(alltcurves[threshs[0]])
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
  plt.plot(mncurve)
  plt.savefig(BPname+'_filt%s_dropping_accuracy_mean.pdf' % thresh)
  plt.show()
else:
   
  useserr=False
  for thresh in threshs:
    print("Processing thresh=", thresh)
    curves=np.array(alltcurves[thresh])
    tlabel='T%s' % thresh
    nreps,nfeatures=curves.shape
    print("nreps=", nreps)
    print("nfeatures=", nfeatures)
    mncurve=curves.mean(0)
    if useserr:
      sevals=curves.std(0)/np.sqrt(nreps)
      print("mncurve[0:5]=", mncurve[0:5])
      print("sevals[:10].mean()",sevals[:10].mean())
      plt.errorbar(np.arange(nfeatures), mncurve, yerr=sevals, label=tlabel)
    else:
      plt.plot(mncurve, label=tlabel)
       
  if useserr:
    plt.legend()
    plt.savefig(BPname+'_multifilt%d_dropping_accuracy_se.pdf' % nthresh)
    plt.show()
  else:
    plt.legend()
    plt.savefig(BPname+'_multifilt%d_dropping_accuracy_mean.pdf' % nthresh)
    plt.show()
    
