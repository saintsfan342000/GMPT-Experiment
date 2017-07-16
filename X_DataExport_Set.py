import numpy as n
L = n.genfromtxt
fit = n.polyfit
abs=n.abs
from numpy import nanmax, nanmin
import os
import pandas as pd
import shutil
import matplotlib.pyplot as p

'''
For generating PT set figures for Kaleidagraph.
Each expt has a sheet
Also a LL sheet
'''
plot = True
expts = n.array([2,3,4,7,8,10,11])
FSSS = 'FS30SS10'
# [0]Expt No., [1]Mon.Day, [2]Material, [3]Tube No., [4]Alpha, [5]Alpha-True    , [6]Mean Radius, [7]Thickness, [8]Eccentricity
key = n.genfromtxt('../ExptSummary.dat', delimiter=',')
# Sort by alpha
key = key[ n.in1d(key[:,0],expts) ]
key = key[ key[:,5].argsort() ]
expts = key[:,0].astype(int)
alpha = key[:,4]

fid = pd.ExcelWriter('../GMPT_SetData.xlsx')
exporthead = 'Stage, Force, Pres(ksi), SigX, SigQ, EpsX(%), EpsQ(%)'
exporthead = exporthead.split(',')
LL = n.empty((len(expts),9)) # Expt, alpha, Stage, Force, Press, 
#                              sigx_LL, sigq_LL, epx_LL, epq_LL

for k,X in enumerate(expts):
    
    path = '../GMPT-{}_{}'.format(X,FSSS)
    print(path)
    
    # STLP: Stage, Force, Press, AxSts, HoopSts
    STPF = L(path+'/STPF.dat', delimiter=',', usecols=(0,2,3,4,5))

    # [0] Stage, [1,2,3]eps_x,q,xq(point avg), [4]eps_x(1"ext), [5]eps_q(BFcirc@mid), [6]d/L
    E = 100*L( path+'/Results.dat', delimiter=',', usecols=(1,2))

    D = n.c_[STPF,E]

    # Limit load
    loc = L( path + '/zMisc/prof_stages.dat', delimiter=',', dtype=int)[2]
    LL[k] = X, alpha[k], *D[loc]
  
    pd.DataFrame( D ).to_excel( fid, sheet_name='Ex{}-{}'.format(X, alpha[k]), 
                                index=False, header=exporthead)

    if plot == True:
        p.figure(1)
        p.plot(D[:,4],D[:,3])
        p.figure(2)
        p.plot(D[:,6],D[:,5])

exporthead = ['Expt', 'Alpha'] + exporthead
pd.DataFrame(LL).to_excel( fid, sheet_name='Limit Load'.format(X, alpha[k]), 
                            index=False, header=exporthead)

fid.save()

if plot:
    p.figure(1)
    p.plot(LL[:,6],LL[:,5],'o')
    p.figure(2)
    p.plot(LL[:,8],LL[:,7],'o')
    p.show('all')