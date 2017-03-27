import numpy as n
import matplotlib.pyplot as p
import os
from sys import argv
import figfun as f


try:
    worthless, expt, FS, SS, path = argv
    expt = int(expt)
    FS = int(FS)
    SS = int(SS)
    savefigs = True
except ValueError:
    expt = 1
    FS = 30
    SS = 10
    path = '../GMPT-{}_FS{}SS{}'.format(expt,FS,SS)
    savefigs = False


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
profstg = n.genfromtxt('./zMisc/prof_stages.dat', delimiter=',', dtype=int)

##################################################
# Figure 1 - AxSts vs Hoop
##################################################
p.style.use('mysty')
fig1 = p.figure()
ax1 = p.gca()
ax1.plot(sigq, sigx)
m,b = n.polyfit(sigq, sigx, 1)
colors = []
for i in profstg:
    l, = ax1.plot(sigq[i], sigx[i], 'o', mew=0)
    colors.append(l.get_mfc())
ax1.set_xlabel('$\\sigma_{\\theta}$ ($\\mathsf{ksi}$)')
ax1.set_ylabel('$\\sigma_{\\mathsf{x}}$\n($\\mathsf{ksi}$)')
ax1.set_title('Nominal Stress Reponse')
f.eztext(ax1, '$\\alpha$ = {:.2f}\nP$_{{\\mathsf{{max}}}}$ = {:.0f} psi'.format(m,P.max()*1000), 'br')
f.myax(ax1, f.ksi2Mpa, '$\\sigma_{\\mathsf{x}}$\n($\\mathsf{MPa}$)')

##################################################
# Figure 2 - Sigx vs epsx, Sigq vs epsq
##################################################
p.style.use('mysty-12')
fig2, ax21, ax22 = f.make12()
ax21.plot(D[:,4]*1000, sigx, label='1" L$_g$')
ax21.plot(D[:,1]*1000, sigx, label='Point Avg.')
for k,i in enumerate(profstg):
    ax21.plot(D[i,1]*1000, sigx[i], 'o', color=colors[k])
ax21.set_xlabel('$\\epsilon_\\mathsf{x}$($\\mathsf{x10}^\\mathsf{3})$')
ax21.set_ylabel('$\\sigma_{\\mathsf{x}}$\n($\\mathsf{ksi}$)')
leg = f.ezlegend(ax21, loc='lower right')
f.myax(ax21, f.ksi2Mpa, '$\\sigma_{\\mathsf{x}}$\n($\\mathsf{MPa}$)')

ax22.plot(D[:,5], sigq, label='Circ. Fit')
ax22.plot(D[:,2], sigq, label='Point Avg.')
for k,i in enumerate(profstg):
    ax22.plot(D[i,5], sigq[i], 'o', color=colors[k])
ax22.set_xlabel('$\\epsilon_\\theta$')
ax22.set_ylabel('$\\sigma_\\theta$\n($\\mathsf{ksi}$)')
leg = f.ezlegend(ax22, loc='lower right')
f.myax(ax22, f.ksi2Mpa, '$\\sigma_{\\theta}$\n($\\mathsf{MPa}$)')

##################################################
# Figure 3 - Ax Stn vs Hoop Stn
##################################################
p.style.use('mysty')
fig3 = p.figure()
p.plot(D[:,2]*1000,D[:,1]*1000)
ax3 = p.gca()
for k,i in enumerate(profstg):
    ax3.plot(D[i,2]*1000, D[i,1]*1000, 'o', color=colors[k], ms=8)
ax3.set_xlabel('$\\epsilon_\\theta$ ($\\mathsf{x10}^\\mathsf{3}$)')
ax3.set_ylabel('$\\epsilon_\\mathsf{x}$\n($\\mathsf{x10}^\\mathsf{3}$)')
f.myax(ax3)


##################################################
# Figure 0 - Binder Figs
# P vs Time, P vs Machine Disp
##################################################
p.style.use('mysty-sub')
fig0, ax01, ax02 = f.make21()
ax01.plot(time, P*1000,'b')
ax01.set_xlabel('Time (s)')
ax01.set_ylabel('P\n(psi)')
datestr = '{:.2f}/17'.format(date).replace('.','/')
titlestring = 'GMPT-{:.0f}, $\\alpha$ = {}.  {}.  Tube {:.0f}-{:.0f}'.format(
        expt, alpha, datestr, material, tube)
ax01.set_title(titlestring)
ax01.axis(xmin=0,ymin=0)
pmax = P.argmax()
f.eztext(ax01, 'P$_\\mathsf{{max}}$ = {:.0f}'.format(P[pmax]*1000), 'br')
ax01.plot(time[pmax], P[pmax]*1000,'r^',mec='r')
f.myax(ax01)

ax02.plot(Disp, P*1000,'b')
ax02.set_xlabel('$\\delta_\\mathsf{MTS}$ (s)')
ax02.set_ylabel('P\n(psi)')
#ax02.axis(xmin=0,ymin=0)
pmax = P.argmax()
f.eztext(ax02, 'P$_\\mathsf{{max}}$ = {:.0f}'.format(P[pmax]*1000), 'br')
ax02.plot(Disp[pmax], P[pmax]*1000,'r^',mec='r')
f.myax(ax02)
# Calculate a_true and plop it in between
a_true = n.polyfit(d[:,3],d[:,4],1)[0]
ax02.text(.5, .48, '$\\alpha_\\mathsf{{true}}$ = {:.2f}'.format(a_true),
          ha='center', va='center', transform=fig0.transFigure) 

if savefigs:
    fig3.savefig('3 - StrainProfile.png',dpi=125)
    fig2.savefig('2 - StrainPath.png',dpi=125)
    fig1.savefig('1 - Sts-Delta-Rot.png',dpi=125)
    fig0.savefig('BinderFig.pdf',bbox_inches='tight')
    p.close('all')
else:
    p.show('all')

