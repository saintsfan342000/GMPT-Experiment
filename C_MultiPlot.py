import numpy as n
from numpy.linalg import eigvalsh
from numpy import pi
import os
from sys import argv
from pandas import read_excel
import figfun as f
import os
import matplotlib.pyplot as p
p.close('all')

# Specify expts or alpha!

try:
    argv = argv[1:]
    if len(argv) == 1:
        # we only have alpha!
        alpha = float(argv[0])
        expts = n.array([])
    elif len(argv)>1:
        alpha = ...
        expts = n.array(argv).astype(int)
    else:
        raise
except:
    expts = n.array([2,3,4,7,8,10,11,12,13])
    alpha = ...

FS, SS = 15,5
savefigs = 1
savepath = '..'

# [0]Expt No., [1]Mon.Day, [2]Material, [3]Tube No., [4]Alpha, [5]Alpha-True    , [6]Mean Radius, [7]Thickness, [8]Eccentricity
key = n.genfromtxt('../ExptSummary.dat', delimiter=',')
# Sort by alpha
key = key[ n.in1d(key[:,0],expts) ]
key = n.flipud(key[ key[:,5].argsort() ])
expts = key[:,0].astype(int)


# Initialize profstg array:
# Row for each experiment
# Colums for sigx, sigq, epsx, epsq
limload = n.empty((len(expts),4))
lastpic = n.empty((len(expts),4))

for k,X in enumerate(expts):
    
    relpath  = '../GMPT-{}_FS{}SS{}'.format(X,FS,SS)
    expt, date, material, tube, alpha, a_true, Rm, thickness, ecc = key[ key[:,0] == X, : ].ravel()
  
    #########
    # Results.dat
    #   [0] Stage, [1,2,3]eps_x,q,xq(point avg), [4]eps_x(1"ext), [5]eps_q(BFcirc@mid), [6]d/L, Lg=4.330875
    # STPF.dat
    #   [0]Stage, [1]Time, [2]Force(kip), [3]Pressure(ksi), [4]NomAxSts(ksi), [5]NomHoopSts(ksi), [6]LVDT(volt), [7]MTSDisp(in)
    # ur_profiles
    #   First column:  Undef y-coord
    #   k+1 to k+4 column:  Stage K ur/Ro for vert proile 
    # LEp_profiles
    #   Col [0]: Undeformed RQ or y coord.
    #   Col [k+1]: Stage k LEp along profile

    D = n.genfromtxt('{}/Results.dat'.format(relpath), delimiter=',')
    stg, time, F, P, sigx, sigq, LVDT, Disp = n.genfromtxt('{}/STPF.dat'.format(relpath), delimiter=',').T
    ur_prof = n.genfromtxt('{}/ur_profiles.dat'.format(relpath), delimiter=',')
    LEp_prof = n.genfromtxt('{}/LEp_profiles.dat'.format(relpath), delimiter=',')
    s1, s2, LL, last = n.genfromtxt('{}/zMisc/prof_stages.dat'.format(relpath), delimiter=',', dtype=int)
    maxpt = n.genfromtxt('{}/MaxPt.dat'.format(relpath), delimiter=',')
      
    limload[k] = sigx[LL], sigq[LL], D[LL,1], D[LL,2]
    lastpic[k] = sigx[last], sigq[last], D[last,1], D[last,2]

    ##################################################
    # Figure 1 - Nominal Stres
    ##################################################
    if k == 0:
        p.style.use('mysty')
        fig1 =  p.figure(figsize=(7,7))
        ax1 = fig1.add_axes([1.5/7,1.5/7,4/7,4/7])

    masterline,  = ax1.plot(sigq, sigx, label='{} || {:.0f}'.format(a_true,X))
    mastercolor = masterline.get_color()
    masterlabel = masterline.get_label()

    if X == expts[-1]:
        p.style.use('mysty')
        l1, = ax1.plot(limload[:,1],limload[:,0],'r^',ms=4)
        l2, = ax1.plot(lastpic[:,1],lastpic[:,0],'rs',ms=4)
        ax1.axis([0,50,0,50])
        ax1.set_xlabel('$\\sigma_{\\theta}$ ($\\mathsf{ksi}$)')
        ax1.set_ylabel('$\\sigma_{\\mathsf{x}}$\n($\\mathsf{ksi}$)')
        f.myax(ax1, f.ksi2Mpa)
        leg2 = ax1.legend([l1,l2],['LL','Fail'],loc='upper left', handletextpad=.1)
        p.setp(leg2.get_texts(), color='r')
        ax1.add_artist(leg2)
        leg1 = f.ezlegend(ax1,title='$\\alpha\\prime$ || Exp.')

    ##################################################
    # Figure 2 - Axial Sts-stn, Hoop sts-stn
    ##################################################
    if k == 0:
        p.style.use('mysty-12')
        #p.rcParams['axes.prop_cycle'] = cycler('color',colors)
        fig2, ax21, ax22 = f.make12()
   
    ax21.plot(D[:,1]*100, sigx, label=masterlabel)
    if X == expts[-1]:
        p.style.use('mysty-12')
        ll21, = ax21.plot(limload[:,2]*100,limload[:,0],'r^')
        ax21.axis(ymin=0)
        ax21.set_xlabel('$\\epsilon_\\mathsf{x}$ (%)')
        ax21.set_ylabel('$\\sigma_{\\mathsf{x}}$\n($\\mathsf{ksi}$)')
        leg2 = ax21.legend([ll21],['LL'],loc='upper left', handletextpad=0.1)
        leg = f.ezlegend(ax21, loc='lower right')
        ax21.add_artist(leg2)
        p.setp(leg2.get_texts(), color='r')
        f.myax(ax21, f.ksi2Mpa, '$\\sigma_{\\mathsf{x}}$\n($\\mathsf{MPa}$)')

    ax22.plot(D[:,2]*100, sigq, label=masterlabel)
    if X == expts[-1]:
        p.style.use('mysty-12')
        ll22, = ax22.plot(limload[:,3]*100,limload[:,1],'r^')
        ax22.set_xlabel('$\\epsilon_\\theta$ (%)')
        ax22.set_ylabel('$\\sigma_\\theta$\n($\\mathsf{ksi}$)')
        ax22.axis(xmin=0)
        leg2 = ax22.legend([ll22],['LL'],loc='upper left', handletextpad=0.1)
        leg = f.ezlegend(ax22, loc='lower right')
        ax22.add_artist(leg2)
        p.setp(leg2.get_texts(), color='r')
        f.myax(ax22, f.ksi2Mpa, '$\\sigma_{\\theta}$\n($\\mathsf{MPa}$)')


    ##################################################
    # Figure 3 - Ax Strain vs Hoop Strain
    ##################################################
    if k == 0:
        p.style.use('mysty')
        #p.rcParams['axes.prop_cycle'] = cycler('color',colors)
        fig3 =  p.figure(figsize=(7,7))
        ax3 = fig3.add_axes([1.5/7,1.5/7,4/7,4/7])
    
    ax3.plot(D[:,2]*100, D[:,1]*100, label=masterlabel)
    if X == expts[-1]:
        p.style.use('mysty')
        ll3, = ax3.plot(limload[:,3]*100, limload[:,2]*100, 'r^')
        ax3.axis(xmin=0)
        ax3.set_xlabel('$\\epsilon_\\theta$ (%)')
        ax3.set_ylabel('$\\epsilon_\\mathsf{x}$\n(%)')
        leg2 = ax3.legend([ll3],['LL'],loc='upper left', handletextpad=0.1)
        leg = f.ezlegend(ax3, loc='center right')
        ax3.add_artist(leg2)
        p.setp(leg2.get_texts(), color='r')
        f.myax(ax3)
        
    ##################################################
    # Figure 4 - Failure Stn vs Triax
    ##################################################
    if k == 0:
        p.style.use('mysty-sub')
        fig4, ax4, ax421 = f.make21()
        ax422 = ax421.twiny()

    
    SIG = n.array([[0,0,0],[0,sigq[last],0],[0,0,sigx[last]]])
    sigm = SIG.trace()/3
    S = SIG - SIG.trace()*n.eye(3)/3
    sig3,sig2,sig1 = n.sort(eigvalsh(SIG))
    s3,s2,s1 = n.sort(eigvalsh(S))
    J2 = ( (sig1-sig2)**2 + (sig2-sig3)**2 + (sig3-sig1)**2 )/6
    J3 = s1*s2*s3
    if not n.isclose(J3, (s1**3+s2**3+s3**3)/3):
        raise ValueError("J3 isn't correct")
    sigeq = n.sqrt(3*J2)
    if not n.isclose(sigeq, (n.sqrt( ((sigx-sigq)**2 + (sigx**2) + (sigq**2))/2 ))[last]):
        raise ValueError("Equivalent stress isn't correct")
    # Barsoum=Faleskog lode parameter
    mu = (2*sig2-sig1-sig3)/(sig1-sig3)
    c3q = 27*J3/(2*sigeq**3)
    # Lode Ange
    Q = n.arccos(c3q)/3
    if not n.isclose(mu, n.sqrt(3)*n.tan(Q-pi/6)):
        raise ValueError("Lode Params are not correct")
    
    triax = sigm/sigeq
    eef = maxpt[last,-1]
    
    ax4.plot(triax, eef, 's', color=mastercolor)
    ax4.plot([],[],color=mastercolor, label=masterlabel)
    #ax4.text(triax,.35,'{}'.format(a_true),color=mastercolor,ha='center',va='top',size=10)

    ax421.plot(mu,eef, 's', color=mastercolor)
    ax421.plot([],[],color=mastercolor,label=masterlabel)
    #ax422.plot(Q,eef,'o', color=mastercolor)

    if X == expts[-1]:
        p.style.use('mysty')
        #ax4.axis(ymin=0, ymax=0.4,xmax=2.1)
        ax4.set_xlabel('$\\sigma_{\\mathsf{m}}/\\sigma_{\\mathsf{e}}$')
        ax4.set_ylabel('$\\mathsf{e}^{\\mathsf{p}}_{\\mathsf{e}}$')
        leg = f.ezlegend(ax4, title='$\\alpha\\prime$ || Exp.')
        f.myax(ax4)

        ax421.set_xlabel('B-F Lode Param $\\mu$')
        ax421.set_xlim([-1,1.05])
        ax421.set_ylabel('$\\mathsf{e}^{\\mathsf{p}}_{\\mathsf{e}}$')
        ax422.set_xticks(n.arange(0,5*pi/12,pi/12))
        ax422.set_xticklabels(['0',r'$\pi$/12',r'$\pi$/6',r'$\pi$/4',r'$\pi$/3'])
        ax422.set_xlim([0,1.05*pi/3])
        ax422.set_xlabel('Lode Angle $\\theta$')
        ax422.patch.set_alpha(0)
        f.ezlegend(ax421, title='$\\alpha\\prime$ || Exp.')
        f.myax(ax421)

if not savefigs:
    p.show('all')        
else:
    fig1.savefig('{}/1_StsSts.png'.format(savepath),dpi=125,bbox_inches='tight')
    #fig1.savefig('{}/1_StsSts.pdf'.format(savepath),dpi=125,bbox_inches='tight')
    fig2.savefig('{}/2_StsStn.png'.format(savepath),dpi=125,bbox_inches='tight')
    #fig2.savefig('{}/2_StsStn.pdf'.format(savepath),dpi=125,bbox_inches='tight')
    fig3.savefig('{}/3_StnStn.png'.format(savepath),dpi=125,bbox_inches='tight')
    #fig3.savefig('{}/3_StnStn.pdf'.format(savepath),dpi=125,bbox_inches='tight')
    fig4.savefig('{}/4_TriaxLode.png'.format(savepath),dpi=125,bbox_inches='tight')
    #fig4.savefig('{}/4_TriaxLode.pdf'.format(savepath),dpi=125,bbox_inches='tight')
    p.close('all')
