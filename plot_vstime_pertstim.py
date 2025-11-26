#!/usr/bin/env python

from sputils import *
from pstimutils import *

nargs=len(sys.argv)

prmsd={'name': 'defaultrun', 'verbose':0, 'figfile': '', 'ymin': 0, 'ymax': None, 'xmin': None, 'xmax': None, 'title': '', 'nextstim': 0, 'yerr': 'std', 'usefilt': 0}

oargs=parse_arguments_dict(prmsd, sys.argv)

figfile = prmsd["figfile"]
ymin = prmsd['ymin']
ymax = prmsd["ymax"]
xmin = prmsd['xmin']
xmax = prmsd["xmax"]
title = prmsd["title"]
nextstim = prmsd['nextstim']
yerrplt = prmsd['yerr']
usefilt = prmsd['usefilt']

print("yerrplt=", yerrplt)

nargs=len(oargs)    

stimalpha=0.2
plotherr=0
addfineticks=False

binms=1000/45.5
usems=True
if usems:
   tscale=binms
   tunitname='ms'
   tstimoffset=0.
else:
   tscale=1.
   tunitname = 'bin #'
   tstimoffset = -0.5

if nargs<2:
   print('Usage: '+sys.argv[0]+' vstimefiles')
   print("prmsd=", prmsd)
   sys.exit(0)
   
iarg=1
if nargs>iarg: filenames=oargs[iarg:]
iarg+=1

colors=list('krgbcmy')+[np.random.rand(3) for _ in range(50)]
colors=list('krgbcmy')+[(1, 0.5, 0), (0.5, 1, 0), (0.5, 0., 1), (0, 1, 0.5),  (1, 0., 0.5), (1, 0.5, 0), (0.5, 1, 0.)]
colors=20*colors
nfiles=len(filenames)
print("Processing %d files:" % len(filenames))
print(filenames)
rtxts1,commons1=remove_longest_seqmatch(filenames, minlen=5, nrepeat=3)

idata=0
missingfiles=[]
already_plotted=[]
xlimin, xlimax=0, 0
ylimin, ylimax=0, 0

for ifile, rezfile in enumerate(filenames):
    try:
       dd=loadnpy(rezfile)
    except:
       print("Cannot find file %s! SKIPPING!" % rezfile)
       missingfiles.append([ifile, datname, model])
       continue
       
    rezmat=dd[0]
    timeoffsets, naver, troffset, chancerr, ttfrac, nclasses = dd[1]
    methods=dd[2]
    noob,ntimesteps,nmethods=rezmat.shape
    print("noob=", noob)
    print("ntimesteps=", ntimesteps)
    print("nmethods=", nmethods)
    ittomax=max([2, int(chancerr)-2])
    
    assert nmethods == len(methods)
    print("naver=", naver)
    
    for imth in range(nmethods):
       if nmethods>1:
          plt.subplot('%d1%d' % (nmethods, imth+1))
       mmeans=rezmat[:,:,imth].mean(0)
       mmmax=mmeans.max()
       mmmin=mmeans.min()
       
       if mmmin<ylimin: ylimin=mmmin
       if mmmax>ylimax: ylimax=mmmax
       
       mstds=rezmat[:,:,imth].std(0)
       mses=rezmat[:,:,imth].std(0)/np.sqrt(noob)
       tts=np.array(timeoffsets)+troffset+naver/2.
       print("yerrplt=", yerrplt)
       
       if yerrplt == 'std': useyerr=mstds
       else: useyerr=mses
       
       ttsx=tscale*(tts+tstimoffset)
       
       if nfiles==1:
          plt.errorbar(ttsx, mmeans, yerr=useyerr, elinewidth=0.5, color='k', label='accuracy')
       else:
          plt.errorbar(ttsx, mmeans, yerr=useyerr, elinewidth=0.5, color=colors[ifile], label=rtxts1[ifile].strip())

       if plotherr:
         for itto, ttor in enumerate(timeoffsets):
            if nfiles==1: clr=colors[itto]
            else: clr=colors[ifile]
            tto=ttor+troffset
            if plotherr>1: ittop=(itto%ittomax)+1
            else: ittop=mmeans[itto]
            plt.plot([tto,tto+naver], [ittop, ittop], '--', color=clr, lw=0.5)
            plt.plot([tto,tto+naver], [ittop, ittop], color=clr, marker='|', lw=0.5)
       if nclasses not in already_plotted:
              mintbx=ttsx[0]-naver/2
              maxtbx=tts[-1]+naver/2
              mintbx=tts[0]
              maxtbx=tts[-1]
              if mintbx< xlimin: xlimin=mintbx
              if maxtbx> xlimax: xlimax=maxtbx
              plt.plot([tscale*mintbx-1, tscale*maxtbx+1], [chancerr, chancerr], '--', color=(0.5, 0.2, (nclasses*0.1)%1.), lw=1, label='chance (n=%d)' % nclasses)
              already_plotted.append(nclasses)
       if ifile == 0:
              plt.axvspan(tstimoffset*tscale, (4.55+tstimoffset)*tscale, facecolor='r', alpha=stimalpha, label='stimulation')
              for iavrng in range(naver):
                 naway=iavrng+1
                 naway=naver-iavrng
                 falpha=0.8*stimalpha*(naver-naway)/naver
                 xleft=-naway+tstimoffset
                 xright=xleft+1
                 plt.axvspan(tscale*xleft, tscale*xright, facecolor='r', alpha=falpha)
                 xleft=4.55+naway-1+tstimoffset
                 xright=xleft+1
                 plt.axvspan(tscale*xleft, tscale*xright, facecolor='r', alpha=falpha)
                 
                 
                 
    #   plt.plot([tts[0], tts[-1]], [chancerr, chancerr], '--', color=(0.95,0.3,0.1), lw=1, label='chance')
    #   plt.plot([0, 0], [0, mmeans.max()], '--', color=(0.05,0.95,0.1), lw=1, label='onset')
       if nmethods>1:
          plt.title('method=%s' % methods[imth])

if True:

  if usems:
     ntickskp=100
     mntoff=ntickskp*int(min(tscale*(tts-naver/2.))//ntickskp)
     mxtoff=ntickskp*int(max(tscale*(tts+naver/2.))//ntickskp)
     fticks=list(range(mntoff, mxtoff,ntickskp))
  else:
     if max(np.abs(tts))>10: ntickskp=5
     else: ntickskp=2
     mntoff=ntickskp*int(min(tts-naver/2.)//ntickskp)
     mxtoff=ntickskp*int(max(tts+naver/2.)//ntickskp)
     fticks=list(range(mntoff, mxtoff,ntickskp))
     if ntickskp==5 and False:
        fticks.remove(-5)
        fticks.remove(5)
        fticks=sorted(fticks+[-6, -4, -2, 2, 4, 6, 8])
  
  print("missingfiles=", missingfiles)
  
  # Sort legends
  handles, labels = plt.gca().get_legend_handles_labels()
  sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])  # Sort by label name
  sorted_labels, sorted_handles = zip(*sorted_handles_labels)
  fntszs=[9, 10, 11, 16]
  plt.legend(sorted_handles, sorted_labels, fontsize=fntszs[0])
  plt.xticks(fticks, fontsize=fntszs[1])
  plt.yticks(np.arange(0,80,10), fontsize=fntszs[2])
  plt.ylabel('Accuracy [%]', fontsize=fntszs[3])
  plt.xlabel('Time [%s]' % tunitname, fontsize=fntszs[3])
  if title: plt.title(title)
  if ymax is None: ymax=ylimax+2
  plt.ylim([ymin,ymax])
  print("xlimin=", xlimin)
  print("xlimax=", xlimax)
  if xmin is None: xmin=tscale*(xlimin-1.)
  if xmax is None: xmax=tscale*(xlimax+1.)
  plt.xlim([xmin, xmax])
  plt.tight_layout()
  if figfile: plt.savefig(figfile)
  plt.show()
  
else:
  if (tts[-1]-tts[0])>150: ntickskp=20
  elif (tts[-1]-tts[0])>70: ntickskp=10
  elif (tts[-1]-tts[0])>20: ntickskp=5
  else: ntickskp=2
  
  mntoff=ntickskp*int(min(tts-naver/2.)//ntickskp)
  mxtoff=ntickskp*int(max(tts+naver/2.)//ntickskp)
  
  fticks=list(range(mntoff, mxtoff,ntickskp))
  
  if ntickskp==5 and addfineticks:
     fticks.remove(-5)
     fticks.remove(5)
     fticks=sorted(fticks+[-6, -4, -2, 2, 4, 6, 8])
     
  plt.xticks(fticks)
  plt.xlabel('Time bins')
  if title: plt.title(title)
  
  handles, labels = plt.gca().get_legend_handles_labels()
  sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])  # Sort by label name
  sorted_labels, sorted_handles = zip(*sorted_handles_labels)
  
  if ymax is None: ymax=ylimax+2
  
  if nextstim:
     plt.annotate('Next Stim.', 
              xy=(nextstim, 0), 
              xytext=(nextstim, ylimax/2.),  # Adjust vertical position of label
              arrowprops=dict(arrowstyle="<-", color="red", alpha=0.5),
              ha='center', 
              va='bottom')
  
  fntszs=[10, 12, 12, 16]
  plt.legend(sorted_handles, sorted_labels, fontsize=fntszs[0])
  plt.xticks(fticks, fontsize=fntszs[1])
  plt.yticks(np.arange(0, ylimax,10), fontsize=fntszs[2])
  plt.ylabel('Accuracy [%]', fontsize=fntszs[3])
  plt.xlabel('Time bins', fontsize=fntszs[3])
  print("xlimin=", xlimin)
  print("xlimax=", xlimax)
  if xmin is None: xmin=xlimin-1.2
  if xmax is None: xmax=xlimax+1.2
  plt.xlim([xmin, xmax])
  plt.ylim([ymin,ymax])
  
  plt.tight_layout()
  if figfile: plt.savefig(figfile)
  plt.show()
  
       
