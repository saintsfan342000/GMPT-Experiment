import numpy as n
import matplotlib.pyplot as p
from pandas import read_csv
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
    expt = 3
    FS = 15
    SS = 5
    path = '../GMPT-{}_FS{}SS{}'.format(expt,FS,SS)
    savefigs = True


    
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
E = n.genfromtxt('WholeFieldAverage.dat', delimiter=',')
stg, time, F, P, sigx, sigq, LVDT, Disp = n.genfromtxt('STPF.dat', delimiter=',').T
ur_prof = read_csv('./ur_profiles.dat', sep=',', comment='#', header=None, index_col=None).values
LEp_prof = read_csv('./LEp_profiles.dat', sep=',', comment='#', header=None, index_col=None).values
profStg = n.genfromtxt('./zMisc/prof_stages.dat', delimiter=',', dtype=int)

titlestring = 'GMPT-{:.0f}, $\\alpha$ = {}.  FS{:.0f}SS{:.0f}. Tube {:.0f}-{:.0f}'.format(expt,alpha,FS,SS,material,tube)

##################################################
# Figure 1 - AxSts vs Hoop
##################################################
p.style.use('mysty')
fig1 = p.figure()
ax1 = p.gca()
ax1.plot(sigq, sigx)
m,b = n.polyfit(sigq, sigx, 1)
colors = []
for i in profStg:
    l, = ax1.plot(sigq[i], sigx[i], 'o', mew=0)
    colors.append(l.get_mfc())
ax1.axis([0,1.05*sigq.max(),0,1.05*sigx.max()])    
ax1.set_xlabel('$\\sigma_{\\theta}$ ($\\mathsf{ksi}$)')
ax1.set_ylabel('$\\sigma_{\\mathsf{x}}$\n($\\mathsf{ksi}$)')
ax1.set_title('Nominal Stress Reponse\n{}'.format(titlestring), fontsize=14)
f.eztext(ax1, '$\\alpha$ = {:.2f}\nP$_{{\\mathsf{{max}}}}$ = {:.0f} psi'.format(m,P.max()*1000), 'br')
f.myax(ax1, f.ksi2Mpa, '$\\sigma_{\\mathsf{x}}$\n($\\mathsf{MPa}$)')

##################################################
# Figure 2 - Sigx vs epsx, Sigq vs epsq
##################################################
p.style.use('mysty-12')
fig2, ax21, ax22 = f.make12()
ax21.plot(D[:,4]*1000, sigx, label='1" L$_g$')
ax21.plot(D[:,1]*1000, sigx, label='Point Avg.')
for k,i in enumerate(profStg):
    ax21.plot(D[i,1]*1000, sigx[i], 'o', color=colors[k])
ax21.axis(ymax=1.05*sigx.max())    
ax21.set_xlabel('$\\epsilon_\\mathsf{x}$($\\mathsf{x10}^\\mathsf{3})$')
ax21.set_ylabel('$\\sigma_{\\mathsf{x}}$\n($\\mathsf{ksi}$)')
ax21.set_title(titlestring, fontsize=14)
leg = f.ezlegend(ax21, loc='lower right')
f.myax(ax21, f.ksi2Mpa, '$\\sigma_{\\mathsf{x}}$\n($\\mathsf{MPa}$)')

ax22.plot(D[:,5], sigq, label='Circ. Fit')
ax22.plot(D[:,2], sigq, label='Point Avg.')
for k,i in enumerate(profStg):
    ax22.plot(D[i,2], sigq[i], 'o', color=colors[k])
ax22.set_xlabel('$\\epsilon_\\theta$')
ax22.set_ylabel('$\\sigma_\\theta$\n($\\mathsf{ksi}$)')
ax22.set_title(titlestring, fontsize=14)
leg = f.ezlegend(ax22, loc='lower right')
f.myax(ax22, f.ksi2Mpa, '$\\sigma_{\\theta}$\n($\\mathsf{MPa}$)')

##################################################
# Figure 3 - Ax Stn vs Hoop Stn
##################################################
p.style.use('mysty')
fig3 = p.figure()
p.plot(D[:,2]*100,D[:,1]*100)
ax3 = p.gca()
for k,i in enumerate(profStg):
    ax3.plot(D[i,2]*100, D[i,1]*100, 'o', color=colors[k], ms=8)
ax3.set_xlabel('$\\overline{\\epsilon_\\theta}$ (%)')
ax3.set_ylabel('$\\overline{\\epsilon_\\mathsf{x}}$\n(%)')
ax3.set_title('Nominal strain response\n{}'.format(titlestring), fontsize=14)
ax3.axvline(0,color='k')
ax3.axhline(0,color='k')
f.myax(ax3)

##################################################
# Figure 3b - Ax Stn vs Hoop Stn, whole field data
##################################################
p.style.use('mysty')
C = [i['color'] for i in list(p.rcParams['axes.prop_cycle'])]
fig3b = p.figure()
ax3b = fig3b.add_subplot(111)
# Loop thru for .5", 1", 1.5", 1.9"
for z,j in enumerate([0.5, 1, 1.5, 1.9]):
    ax3b.plot(E[:,2*z+2]*100,E[:,2*z+1]*100, label='{}"'.format(j), color=C[z+5])
    for k,i in enumerate(profStg):
        ax3b.plot(E[i,2*z+2]*100, E[i,2*z+1]*100, 'o', color=colors[k], ms=4,mec='k')
ax3b.axis(xmin=0,ymin=0)
ax3b.set_xlabel('$\\epsilon_\\theta$ (%)')
ax3b.set_ylabel('$\\epsilon_\\mathsf{x}$\n(%)')
f.ezlegend(ax3b, title='$\\overline{{\\mathsf{{L}}}}$ (+/-)')
ax3b.set_title('Nominal strain response\n{}'.format(titlestring), fontsize=14)
f.myax(ax3b)

##################################################
# Figure 4 - Profiles
##################################################
p.style.use('mysty-sub')
p.rcParams['font.size'] = 18
p.rcParams['axes.labelsize'] = 22
fig4 = p.figure()
ax4 = fig4.add_subplot(111)
for k,i in enumerate(profStg):
    l1, = ax4.plot(ur_prof[:,3*i+3], ur_prof[:,0]*2/4, alpha=0.5, color=colors[k], lw=3)
    #ax4.plot(ur_prof[:,4*i+3], ur_prof[:,0]*2/4, alpha=0.35, color=colors[k])
    l2, = ax4.plot(ur_prof[:,3*i+2], ur_prof[:,0]*2/4, color=colors[k], lw=3)
leg4 = ax4.legend([l2,l1],
        ['u$_\\mathsf{r}$/R$_\\mathsf{o}$', 'BF Circ.'],
        loc='upper right'
        )
p.setp(leg4.get_lines(), color=colors[0], lw=3)
ax4.set_xlabel('u$_\\mathsf{r}$/R$_\\mathsf{o}$')
ax4.set_ylabel('$\\frac{\\mathsf{2y}_\\mathsf{o}}{\\mathsf{L}_\\mathsf{g}}$')
ax4.set_title('Radial Displacement Profiles\n{}'.format(titlestring), fontsize=14)
f.myax(ax4,autoscale='preserve')

##################################################
# Figure 5 - LEp Profile thru max point
##################################################
p.style.use('mysty-12')
p.rcParams['font.size'] = 18
p.rcParams['axes.labelsize'] = 22
fig5 = p.figure(figsize=(12,6))
ax5 = fig5.add_axes([.12,.12,.8,.78])
for k,i in enumerate(profStg):
    ax5.plot(LEp_prof[:,0]/thickness, LEp_prof[:,i+1], color=colors[k])
ax5.axis(xmin=-8,xmax=8)
ax5.set_xlabel('s/t$_\\mathsf{o}$')
ax5.set_ylabel('e$_\\mathsf{e}$')
ax5.set_title(titlestring)
f.myax(ax5)

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
infostr = ('P$_\\mathsf{{LL}}$ = {:.0f} psi\n'.format(P[pmax]*1000) + 
           'F$_\\mathsf{{LL}}$ = {:.0f} lb\n'.format(F[pmax]*1000) + 
           '$\\delta_\\mathsf{{LL}}$ = {:.3f}$\\mathsf{{x10}}^{{-\\mathsf{{2}}}}$ in'.format(Disp[pmax]*100)
           )
f.eztext(ax01, infostr, 'br')
ax01.plot(time[pmax], P[pmax]*1000,'r^',mec='r')
f.myax(ax01)

if alpha == 0.5:
    ax02.plot(Disp, P*1000,'b')
    ax02.plot(Disp[pmax], P[pmax]*1000,'r^',mec='r')
    ax02.set_ylabel('P\n(psi)')
else:
    ax02.plot(Disp, F*1000,'b')
    ax02.plot(Disp[pmax], F[pmax]*1000,'r^',mec='r')
    ax02.set_ylabel('F\n(lb)')
ax02.set_xlabel('$\\delta_\\mathsf{MTS}$ (in)')
f.eztext(ax02, infostr, 'br')
f.myax(ax02)
# Calculate a_true and plop it in between
a_true = n.polyfit(sigq, sigx,1 )[0]
ax02.text(.5, .48, '$\\alpha_\\mathsf{{true}}$ = {:.5f}'.format(a_true),
          ha='center', va='center', transform=fig0.transFigure) 

if savefigs:
    fig5.savefig('5 - LEp_profile.png',dpi=125)
    fig4.savefig('4 - Ur_profile.png',dpi=125)
    fig3.savefig('3 - Stn-Stn.png',dpi=125)
    fig3b.savefig('3b - Stn-Stn.png',dpi=125)
    fig2.savefig('2 - Sts-Stn.png',dpi=125)
    fig1.savefig('1 - Sts-Sts.png',dpi=125)
    fig0.savefig('BinderFig.pdf',bbox_inches='tight')
    p.close('all')
else:
    p.show('all')

os.chdir('../AA_PyScripts')