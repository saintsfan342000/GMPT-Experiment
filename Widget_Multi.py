import numpy as n
from numpy import pi
from pandas import read_csv
import matplotlib.pyplot as p
from sys import argv
import figfun as f
p.style.use('mysty')
import glob
import os

expt = 7
size_factor = .8
FS = 15
SS = 5
path = '../GMPT-{}_FS{}SS{}'.format(expt,FS,SS)

try:
   os.chdir(path)
except FileNotFoundError:
    os.chdir(input('TYPE PATH HERE\n'))

key = n.genfromtxt('../ExptSummary.dat'.format(path), delimiter=',')
expt, date, material, tube, alpha, a_true, Rm, thickness, ecc = key[ key[:,0] == expt, : ].ravel()

if n.isnan(alpha):
    alpha = '$\\infty$'
    
#########
# Results.dat
# [0] Stage, [1,2,3]eps_x,q,xq(point avg), [4]eps_x(1"ext), [5]eps_q(BFcirc@mid), [6]d/L, Lg=4.330875
# STPF.dat
# [0]Stage, [1]Time, [2]Force(kip), [3]Pressure(ksi), [4]NomAxSts(ksi), [5]NomHoopSts(ksi), [6]LVDT(volt), [7]MTSDisp(in)

D = n.genfromtxt('Results.dat', delimiter=',')
stg, time, F, P, sigx, sigq, LVDT, Disp = n.genfromtxt('STPF.dat', delimiter=',').T
loc = P.argmax()
ur_prof = read_csv('./ur_profiles.dat', sep=',', comment='#', header=None, index_col=None).values
LEp_prof = read_csv('./LEp_profiles.dat', sep=',', comment='#', header=None, index_col=None).values
profStg = n.genfromtxt('./zMisc/prof_stages.dat', delimiter=',', dtype=int)

titlestring = 'GMPT-{:.0f}, $\\alpha$ = {}.  FS{:.0f}SS{:.0f}. Tube {:.0f}-{:.0f}'.format(expt,alpha,FS,SS,material,tube)

p.style.use('mysty-12')
p.rcParams['font.size'] = 18*size_factor
p.rcParams['axes.labelsize'] = 22*size_factor

W,H = 18,10
fvec = n.array([W,H,W,H])

# ax1:  LEp Profile
x,y,w,h = 1.5,1.1,10,4
ax1_loc = n.array([x,y,w,h]/fvec)
# ax2:  Ur Profile
x,y,w,h = 12.75,2.5,3.5,6
ax2_loc = n.array([x,y,w,h]/fvec)
# ax3:  Hooop-sts / hoop-stn
x,y,w,h = 7.25,6,4,3
ax3_loc = n.array([x,y,w,h]/fvec)
# ax4:  Ax-sts / hoop-sts
x,y,w,h = 1.75,6,4,3
ax4_loc = n.array([x,y,w,h]/fvec)
# ax_sl:  Slider
x,y,w,h = 4,.1,10,.25
ax_sl_loc = n.array([x,y,w,h]/fvec)

W,H = map(lambda x: x*size_factor,(W,H))
fig = p.figure(figsize=(W,H))
for i in [1,2,3,4,'_sl']:
    exec('ax{} = fig.add_axes(ax{}_loc)'.format(i,i))


# ax1:  LEp Profile
line1, = ax1.plot(LEp_prof[:,0]/thickness, LEp_prof[:,0+1], lw=3)
LL1, = ax1.plot(LEp_prof[:,0]/thickness, LEp_prof[:,loc+1], lw=3, alpha=0.0,zorder=-10)
ax1.axis([-8,8,0,LEp_prof[:,1:].max()*1.05])
ax1.set_xlabel('s/t$_\\mathsf{o}$')
ax1.set_ylabel('e$_\\mathsf{e}$')
f.myax(ax1,TW=.0025,HW=.22,HL=.07,OH=.3)

# ax2:  Ur Profile
line2, = ax2.plot(ur_prof[:,4*0+4]*100, ur_prof[:,0]*2/4, lw=3)
LL2, = ax2.plot(ur_prof[:,4*loc+4]*100, ur_prof[:,0]*2/4, lw=3, alpha=0.0,zorder=-10)
ax2.axis(xmin=0, ymin=-1,ymax=1)
ax2.set_xlabel('u$_\\mathsf{r}$/R$_\\mathsf{o}$ (%)')
ax2.set_ylabel('$\\frac{\\mathsf{2y}_\\mathsf{o}}{\\mathsf{L}_\\mathsf{g}}$')
f.myax(ax2, TW=.0045, HW=1, HL=0.035, nudge=('left',.3,-.5))

# ax3:  Hooop-sts / hoop-stn
line3, = ax3.plot(D[0,2]*100, sigq[0], 'o',  ms=8)
LL3, = ax3.plot(D[loc,2]*100, sigq[loc], '^', ms=8, alpha=0.0)
l3, = ax3.plot(D[:,2]*100, sigq, lw=2,zorder=-10)
if not expt in [1,4]:
    ax3.axis(xmin=0,ymin=0)
ax3.set_xlabel('$\\bar{\\epsilon}_\\theta$ (%)')
ax3.set_ylabel('$\\sigma_\\theta$\n($\\mathsf{ksi}$)')
f.myax(ax3, TW=.0035, HW=.45, HL=0.07)

# ax4:  Ax-sts / hoop-sts
line4, = ax4.plot(D[0,1]*100, sigx[0], 'o', ms=8)
LL4, = ax4.plot(D[loc,1]*100, sigx[loc], '^', ms=8, alpha=0.0)
l4, = ax4.plot(D[:,1]*100, sigx, label='Point Avg.', zorder=-10)
ax4.axis(ymax=1.05*sigx.max())    
if not expt in [1,4]:
    ax4.axis(xmin=0,ymin=0, ymax=1.05*sigx.max())
ax4.set_xlabel('$\\bar{\\epsilon}_\\mathsf{x}$ (%)')
ax4.set_ylabel('$\\sigma_{\\mathsf{x}}$\n($\\mathsf{ksi}$)')
f.myax(ax4, TW=.0035, HW=.45, HL=0.07)

fig.text(.5, .98, titlestring, ha='center', va='top', transform=fig.transFigure)
fig.text(.5, .95, 'P = {:.0f} psi'.format(P[0]*1000), ha='center', va='top', transform=fig.transFigure)

from matplotlib.widgets import Slider, Button, RadioButtons
slider = Slider(ax=ax_sl, label='', valmin=0, valmax=stg[-1], valinit=0, valfmt='%.0f', facecolor=line1.get_color())

def update(val):
    i = int(val)
    line1.set_ydata(LEp_prof[:,i+1])
    line2.set_xdata(ur_prof[:,4*i+4]*100)
    line3.set_data(D[i,2]*100,sigq[i])
    line4.set_data(D[i,1]*100,sigx[i])
    tex = fig.texts[-1]
    tex.remove()
    tex = fig.text(.5, .95, 'P = {:.0f} psi'.format(P[i]*1000), ha='center', va='top', transform=fig.transFigure)
    if i >= P.argmax():
        LL1.set_alpha(0.5)
        LL2.set_alpha(0.5)
        LL3.set_alpha(1.0)
        LL4.set_alpha(1.0)
    fig.canvas.draw_idle()

slider.on_changed(update)
p.show()

os.chdir('../AA_PyScripts')
