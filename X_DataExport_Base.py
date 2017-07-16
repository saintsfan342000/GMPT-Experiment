import numpy as n
L = n.genfromtxt
fit = n.polyfit
abs=n.abs
from numpy import nanmax, nanmin
import os
import pandas as pd
import shutil
import matplotlib.pyplot as p

plot = False
expts = [3]
FSSS = 'FS15SS5'

exporthead = 'Stage, Force, Pres(ksi), SigX, SigQ, EpsX(%), EpsQ(%)'
exporthead = exporthead.split(',')

for k,X in enumerate(expts):
   
    path = '../GMPT-{}_{}'.format(X,FSSS)
    print(path)
    
    fid = pd.ExcelWriter('../GMPT{}_ExptData.xlsx'.format(X))
    
    # STLP: Stage, Force, Press, AxSts, HoopSts
    STPF = L(path+'/STPF.dat', delimiter=',', usecols=(0,2,3,4,5))
    
    profstg = L( path + '/zMisc/prof_stages.dat', delimiter=',', dtype=int)
    if X == 3:
        profstg = n.array([179, 272, 379, 400, 410, 420, 426], dtype=int)
       
# [0] Stage, [1,2,3]eps_x,q,xq(point avg), [4]eps_x(1"ext), [5]eps_q(BFcirc@mid), [6]d/L, Lg=4.292908
    A = L( path+'/Results.dat', delimiter=',', usecols=(1,2))*100
    
    d = n.hstack(( STPF,A ))
    
    # [0]NEq [1]NEx [2]Gamma [3]F11-1 [4]F22-1 [5]atan(F12/F22) [6]epeq [7]AramX [8]AramY
    # Amax = L(path+'/loc_mean.dat', delimiter=',', usecols=(0,1))*100

    pd.DataFrame(d).to_excel( fid, sheet_name='Exp{}'.format(X),
                                index=False, header=exporthead)
    
    d_stg = n.hstack(( profstg[:,None], d[profstg] ))
    pd.DataFrame(d_stg).to_excel( fid, sheet_name='Exp{}_stages'.format(X), 
                                  index=False, header=['Stage']+exporthead)
    
    profs = L( path+'/LEp_profiles.dat'.format(X), delimiter=',')
    profs = profs[:,n.hstack((0,profstg+1))]
    profs[:,0]*=(1/.05)
    pd.DataFrame(profs).to_excel( fid, sheet_name='Exp{}_LEpProfs'.format(X), 
                                index=False, header=n.hstack(( 's/t', profstg.astype(str) )) )
    if plot:
        p.plot(d[:,-1],d[:,1])
        p.plot(Amax[:,0],d[:,1])
        p.plot(d[profstg,-2],d[profstg,1],'o')
        p.figure()
        p.plot(profs[:,0],profs[:,1:])
    
fid.save()
p.show('all')
