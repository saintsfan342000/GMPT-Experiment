import numpy as n
L = n.genfromtxt
abs=n.abs
import os
from sys import argv
from pandas import read_excel
from scipy.signal import detrend
import figfun as ff
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
    expts = n.array([1,2,3,4,5,7])
    alpha = ...


FS, SS = 19, 6
savefigs = True

key = n.genfromtxt('../ExptSummary.dat', delimiter=',')
# [0]Expt No., [1]Material, [2]Tube No., [3]Alpha, [4]Alpha-True, [5]Mean Radius, [6]Thickness, [7]Eccentricity

# Figure out if I've specifided alpha or expteriment
if (len(expts) >= 1) and (alpha == ...) :
    #expt, alpha, tube no, thickness, true alpha, ecc
    expinfo = key[ n.in1d(key[:,0], expts), :][:,[0,3,2,6,4,7]]
    savepath = '../ComparisonFigs/PaperSet'.format(alpha)
elif (type(alpha) in [int,float]) and (len(expts)==0):
    expinfo = key[ key[:,1]==alpha,: ][:,[0,3,2,6,4,7]]
    savepath = '../ComparisonFigs/Alpha-{}'.format(alpha)
    if n.isnan(alpha):
        expinfo = key[ n.isnan(key[:,1]),: ][:,[0,3,2,6,4,7]]
        savepath = '../ComparisonFigs/Alpha-Inf'.format(alpha)
else:
    raise ValueError('expts must be empty array OR alpha must be ellipsis')

savepath = '..'
    
# If I have just one expt, flatten the expinfo array
# otherwise, just order them w.r.t. alpha
if (len(expinfo.shape)==2) and (expinfo.shape[0]==1):
    expinfo = expinfo.ravel()
    expts = n.array([expinfo[0]]).astype(int)
    limloads = n.empty( (len(expts),4) ) #Force, torque, disp, rot
    stat2 = n.empty_like(limloads)
    expinfo = expinfo[None,:]
else:
    # A couple of bad expts we want to exclude
    order = n.flipud(n.argsort(expinfo[:,1]))
    expinfo = expinfo[order,:]
    expts = expinfo[:,0].astype(int)
    limloads = n.empty( (len(expts),4) ) #Force, torque, disp, rot
    stat2 = n.empty_like(limloads)
    
    
if (savefigs == True) and not (os.path.exists(savepath)):
    os.mkdir(savepath)

for G in range(len(expts)):
    
    relpath  = '../TTGM-{}_FS{}SS{}'.format(expts[G],FS,SS)
            
    #########
    #Max.dat
    #   '[0]Stage [1]Time [2]AxSts [3]ShSts [4]NEx [5]NEy [6]Gamma [7]F11-1 [8]F22-1 [9]atan(F12/F22) [10]epeq [11]AramX [12]AramY'
    #mean.dat
    #   [0]Stage [1]Time [2]NumPtsPassed [3]AxSts [4]ShSts [5]NEx [6]NEy [7]Gamma [8]F11-1 [9]F22-1 [10]atan(F12/F22) [11]epeq'
    #like_Scott
    #   [0]Stage [1]Time [2]SizeAveragingZone(in) [3]AxSts [4]ShSts [5]NEx [6]NEy [7]Gamma [8]F11-1 [9]F22-1 [10]atan(F12/F22) [11]epeq'
    # profStgs
    #   'Stages at which profiles were generated'
    # profUr
    #   'First Row Stage number.  Second row begin data.\n[0]Ycoord [1:] Stage Ur/Ro'
    #MaxPt.dat
    #   [0]Stage [1]Time [2]AxSts [3]ShSts [4]NEx [5]NEy [6]Gamma [7]F11-1 [8]F22-1 [9]atan(F12/F22) [10]epeq'

    STF = L('{}/STF.dat'.format(relpath),delimiter=',')
    profStg = L('{}/prof_stages.dat'.format(relpath),delimiter=',').astype(int)
    dmax = L('{}/max.dat'.format(relpath),delimiter=',')
    max10 = L('{}/Max10.dat'.format(relpath),delimiter=',')
    dmaxPt = L('{}/MaxPt.dat'.format(relpath),delimiter=',')
    dmean = L('{}/mean.dat'.format(relpath),delimiter=',')
    dscot = L('{}/like_Scott.dat'.format(relpath),delimiter=',')
    DR = L('{}/disp-rot.dat'.format(relpath),delimiter=',')
    #'[0]Stage [1]Time [2]AxSts [3]ShSts [4]Delta/L [5]Phi. Lg = {:.6f} inch'
    limloads[G] = DR[int(profStg[2]),2:]
    stat2[G] = DR[int(profStg[1]),2:]
    profLEp = L('{}/StrainProfiles.dat'.format(relpath),delimiter=',')[1:]
    profUr = L('{}/RadialContraction.dat'.format(relpath),delimiter=',')[1:]
    profUr = profUr[~n.any(n.isnan(profUr),axis=1), :]
    profUr[:,1:] = detrend(profUr[:,1:],axis=0)
    profUr[:,1:] -= n.nanmean( profUr[profUr[:,0]>=.28,:][:,1:], axis=0 )
    #profUr[:,1:] -= n.nanmax( profUr[:,1:], axis=0 )
    
    ##################################################
    # Figure 1 - AxSts-Delta and ShearSts-Rot
    ##################################################
    if G == 0:
        p.style.use('mysty-sub')
        #p.rcParams['axes.prop_cycle'] = p.cycler('color',colors)
        fig1 =  p.figure(1,facecolor='w',figsize=(8,12))
        ax11 = fig1.add_subplot(2,1,1)
        ax12 = fig1.add_subplot(2,1,2)
    
    masterline, = ax11.plot(DR[:,4],DR[:,2],label = '{:.0f}; {}; {:.0f}'.format(expts[G],expinfo[G,1],expinfo[G,2]))
    mastercolor = masterline.get_color()
    masterlabel = masterline.get_label()
    ax12.plot(DR[:,5],DR[:,3],label = '{:.0f}; {}; {:.0f}'.format(expts[G],expinfo[G,1],expinfo[G,2]), color=mastercolor)
    
    if G == len(expts)-1:
        for J in range(len(expts)):
            m11 = ax11.plot(limloads[J,2],limloads[J,0],'^',mec='r',mfc='r',ms=6)[0]
            m21 = ax12.plot(limloads[J,3],limloads[J,1],'^',mec='r',mfc='r',ms=6)[0]
            m12 = ax11.plot(stat2[J,2],stat2[J,0],'o',mec='r',mfc='r',ms=6)[0]
            m22 = ax12.plot(stat2[J,3],stat2[J,1],'o',mec='r',mfc='r',ms=6)[0]
        
        ax11.set_title('Nominal Response',size=18)
        ax11.set_xlabel('$\\delta/\\mathsf{L}$')
        ax11.set_ylabel('$\\Sigma$\n$(\\mathsf{ksi})$')
        ax11.set_ylim([0,1.2*n.max(limloads[:,0])])
        ax11.set_xlim(left=0)
        l1 = ax11.legend([m12,m11],["Station 2", "LL"],loc='upper right',numpoints=1,fontsize=10,frameon=False)
        p.setp(l1.get_texts(),color='r')
        l2 = ax11.legend(loc='lower right',fontsize=10,title='Expt; $\\alpha$; Tube')
        p.setp(l2.get_title(),fontsize=10)
        ax11.add_artist(l1)
        ax12.set_xlabel('$\\phi^\\circ$')
        ax12.set_ylabel('$\\mathcal{T}$\n$(\\mathsf{ksi})$')
        ax12.set_ylim([0,1.2*n.max(limloads[:,1])])
        ax12.set_xlim(left=0)
        l1 = ax12.legend([m22,m21],["Station 2", "LL"],loc='upper right',numpoints=1,fontsize=10,frameon=False)
        p.setp(l1.get_texts(),color='r')
        l2 = ax12.legend(loc='lower right',fontsize=10,title='Expt; $\\alpha$; Tube')
        l2 = ax12.legend(l2.get_lines()[::-1],[i.get_text() for k,i in enumerate(l2.get_texts()[::-1])],fontsize=10,title='Expt; $\\alpha$; Tube',loc='lower right')
        p.setp(l2.get_title(),fontsize=10)
        ax12.add_artist(l1)
        
        ff.myax(ax11,ff.ksi2Mpa,'$\\Sigma$\n$(\\mathsf{MPa})$')
        ff.myax(ax12,ff.ksi2Mpa,'$\\mathcal{T}$\n$(\\mathsf{MPa})$')

    ##################################################
    # Figure 2 - Epsilon-Gamma
    ##################################################
    if G == 0:
        p.style.use('mysty-sub')
        #p.rcParams['axes.prop_cycle'] = cycler('color',colors)
        fig2 = p.figure(2,facecolor='w',figsize=(8,12) )
        ax21 = fig2.add_subplot(2,1,1)
        ax22 = fig2.add_subplot(2,1,2)
    
    ax21.plot(abs(dmean[:,7]),dmean[:,6],'o',ms=4,mfc=mastercolor,mec='none',label=masterlabel)
    ax21.plot(abs(dmax[-1,6]),dmax[-1,5],'s',ms=8,mfc=mastercolor,mec='none')
    ax22.plot(abs(dmean[:,10]),dmean[:,9],'o',ms=4,mfc=mastercolor,mec='none',label=masterlabel)
    ax22.plot(abs(dmax[-1,9]),dmax[-1,8],'s',mfc=mastercolor,mec='none',ms=8)
    
    if G == len(expts)-1:
        ax21.set_title('Mean strain path; Aramis User manual Definitions',fontsize=14)
        ax21.set_xlabel('$\gamma$')
        ax21.set_ylabel('$\\epsilon_y$')
        ax21.set_ylim(bottom=0)
        ax21.set_xlim(left=0)
        l2 = ax21.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=10,numpoints=1,handletextpad=.1,title='Expt; $\\alpha$; Tube')
        p.setp(l2.get_title(),fontsize=10)
        #ax21.legend(loc='lower left')
        ax22.set_title('Mean strain path; Haltom 2013 Definitions',fontsize=14)
        ax22.set_xlabel('$\\gamma = atan(\\mathsf{F}_{\\mathsf{01}}/\\mathsf{F}_{\\mathsf{11}}$)')
        ax22.set_ylabel('$\\epsilon_{\\mathsf{y}}$\n$\\mathsf{F}_{\\mathsf{11}}-\\mathsf{1}$')
        ax22.set_ylim(bottom=0)
        ax22.set_xlim(left=-.015)
        l2 = ax22.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=10,numpoints=1,handletextpad=.1,title='Expt; $\\mathregular{\\alpha}}$; Tube')
        p.setp(l2.get_title(),fontsize=10)
        #ax22.legend(loc='lower left')
        
        ff.myax(ax21)
        ff.myax(ax22)
        
    ##################################################
    # Figure 3 - Radial contraction at Station 2
    ##################################################
    if G == 0:
        p.style.use('mysty')
        #p.rcParams['axes.prop_cycle'] = cycler('color',colors)
        fig3 = p.figure(3,facecolor='w',figsize=(12,6) )
        ax3 = fig3.add_axes([.12,.12,.8,.78])
    
    # Station 2
    ax3.plot(2*profUr[:,0]/0.62,profUr[:,1+1],'-',lw=1.5,color=mastercolor,label=masterlabel)

    if G == len(expts) - 1:
        ax3.set_title('Radial Contraction at Station 2',size=18)
        ax3.set_xlabel('$\\mathsf{2}\\mathsf{y}_{\\mathsf{o}}/\\mathsf{L}_{\\mathsf{g}}$')
        ax3.set_ylabel('$\\frac{\\mathsf{u}_{\\mathsf{r}}}{\\mathsf{R}_{\\mathsf{o}}}$')
        ax3.grid(True,which='both',axis='y',linestyle='--',alpha=0.5)
        l3 = ax3.legend(loc='lower right',title='Expt; $\\alpha$; Tube',frameon=True)
        p.setp(l3.get_title(),fontsize=12)
        p.setp(l3.get_frame(),lw=0)
        #ax3.yaxis.set_ticks(n.arange(-.024,0+.004,.004))
        ax3.axis(xmin=-1,xmax=1,ymax=.002)
        
        ff.myax(ax3,TW=.002,HW=.3,HL=.05,OH=.2)
        
    ##################################################
    # Figure 5 - Radial contraction at LL
    ##################################################
    if G == 0:
        p.style.use('mysty')
        #p.rcParams['axes.prop_cycle'] = cycler('color',colors)
        fig5 = p.figure(5,facecolor='w',figsize=(12,6) )
        ax5 = fig5.add_axes([.12,.12,.8,.78])
    
    # Station 2
    ax5.plot(2*profUr[:,0]/0.62,profUr[:,1+2],'-',lw=1.5,color=mastercolor,label=masterlabel)
   
    if G == len(expts) - 1:
        
        ax5.set_title('Radial Contraction at LL',size=18)
        ax5.set_xlabel('$\\mathsf{2}\\mathsf{y}_{\\mathsf{o}}/\\mathsf{L}_{\\mathsf{g}}$')
        ax5.set_ylabel('$\\frac{\\mathsf{u}_{\\mathsf{r}}}{\\mathsf{R}_{\\mathsf{o}}}$')
        ax5.grid(True,which='both',axis='y',linestyle='--',alpha=0.5)
        l3 = ax5.legend(loc='lower right',title='Expt; $\\alpha$; Tube',frameon=True)
        p.setp(l3.get_title(),fontsize=12)
        p.setp(l3.get_frame(),lw=0)
        #ax5.yaxis.set_ticks(n.arange(-.024,0+.004,.004))
        ax5.axis(xmin=-1,xmax=1,ymax=.002)
        
        ff.myax(ax5,TW=.002,HW=.3,HL=.05,OH=.2)
        
           
    ##################################################
    # Figure 4 - Strain profile
    ##################################################
    if G == 0:
        p.style.use('mysty')
        #p.rcParams['axes.prop_cycle'] = cycler('color',colors)
        fig4 = p.figure(4,facecolor='w',figsize=(12,6) )
        ax4 = fig4.add_axes([.12,.12,.8,.78])
    
    ax4.plot(profLEp[:,-3]/expinfo[G,3],profLEp[:,-1],color=mastercolor,label=masterlabel)
    if G == len(expts) - 1:
        
        ax4.set_title('$\\mathsf{e}^{\\mathsf{p}}_{\\mathsf{e}}$ at Failure',size=18)
        ax4.set_xlabel('y$_{\\mathsf{o}}$/t$_{\\mathsf{o}}$')
        ax4.set_ylabel('$\\mathsf{e}^{\\mathsf{p}}_{\\mathsf{e}}$')
        ax4.set_xlim([-8,8])
        l2 = ax4.legend(loc='upper right',title='Expt; $\\alpha$; Tube')
        p.setp(l2.get_title(),fontsize=12)
        ff.myax(ax4,TW=.0025,HW=.3,HL=.05,OH=.2)
        
    ##################################################
    # Figure 6 - Equiv Stn:  Max, mean at failure, and mean at LL
    ##################################################
    if G == 0:
        p.style.use('mysty')
        #p.rcParams['axes.prop_cycle'] = cycler('color',colors)
        fig6 = p.figure(6)
        ax6 = fig6.gca()
    
    sig, tau = dmax[profStg[2]][2:4]
    triax = (sig/2)/n.sqrt(.75*(sig**2+4*tau**2))
    LL, = ax6.plot(triax, dmean[profStg[2],10],'bs',mec='b',label='LL')
    sig, tau = dmax[profStg[-1]][2:4]
    triax = (sig/2)/n.sqrt(.75*(sig**2+4*tau**2))
    ENDmean, = ax6.plot(triax, dmean[profStg[-1],11],'gs',mec='g',label='$\\bar{\\mathsf{e}}_{\\mathsf{ef}}^\\mathsf{p}$')
    ENDmax, = ax6.plot(triax, dmax[profStg[-1],10],'rs',mec='r',label='$\\mathsf{e}_{\\mathsf{ef}}^\\mathsf{p}$')
    END2, = ax6.plot(triax, max10[1,0],'o',mec='purple',color='purple',label='$\\mathsf{e}_{\\mathsf{ef2}}^\\mathsf{p}$',alpha=0.7)
    tx = ax6.text(triax,1.7,'\n{}'.format(expinfo[G,1]),color=mastercolor,ha='center',va='top',size=10)
    if G == len(expts) - 1:
        oldexpts = n.array([35,20,24,17,34,32,9,30,16,15,18])
        olddata = n.empty((len(oldexpts),2))
        for m,q in enumerate(oldexpts):
            oldmax = n.genfromtxt('../../AAA_TensionTorsion/TT2-{}_FS19SS6/max.dat'.format(q),delimiter=',')
            sig, tau = oldmax[-1][2:4]
            triax = (sig/2)/n.sqrt(.75*(sig**2+4*tau**2))
            olddata[m] = triax, oldmax[-1,10]
        old, = ax6.plot(olddata[:,0],olddata[:,1],'ks',alpha=0.2,label='TT2')
        ax6.axis([0,0.6,0,1.7])
        ax6.set_title('$\\mathsf{e}^{\\mathsf{p}}_{\\mathsf{e}}$ at Limit Load and Failure',size=18)
        ax6.set_xlabel('$\\sigma_{\\mathsf{m}}/\\sigma_{\\mathsf{e}}$')
        ax6.set_ylabel('$\\mathsf{e}^{\\mathsf{p}}_{\\mathsf{e}}$')
        leg6 = ax6.legend([ENDmax,END2,ENDmean,LL,old],[m.get_label() for m in [ENDmax,END2,ENDmean,LL,old]],
                            loc='center left',bbox_to_anchor=(1.01,0.5),bbox_transform=ax6.transAxes,
                            fontsize=18,handlelength=0)
        [tx.set_color(leg6.get_lines()[k].get_color()) for k,tx in enumerate(leg6.get_texts())]
        ff.myax(fig6)

    ##################################################
    # Figure 7 - Equiv Stn vs Rotation
    ##################################################
    if G == 0:
        p.style.use('mysty')
        #p.rcParams['axes.prop_cycle'] = cycler('color',colors)
        fig7 = p.figure(7)
        ax7 = fig7.gca()
    
    path, = ax7.plot(DR[:,5],dmean[:,11],'o',ms=4,color=mastercolor,mfc=mastercolor,mec='none',label=masterlabel)
    LL7, = ax7.plot(DR[profStg[2],5],dmean[profStg[2],11],'rs',ms=6,color='r',mfc='r',mec='r')
    ENDmax7, = ax7.plot(DR[-1,5],dmax[profStg[-1],10],'x',ms=10,mew=2,color=mastercolor,mec=mastercolor)
    if G == len(expts) - 1:
        ax7.axis(xmin=0,ymin=0)
        ax7.set_xlabel('$\\phi^\\circ$')
        ax7.set_ylabel('$\\mathsf{e}^{\\mathsf{p}}_{\\mathsf{e}}$')
        leg7_2 = ax7.legend([LL7,ENDmax7],[LL.get_label(),ENDmax.get_label()],loc='upper left')
        p.setp(leg7_2.get_texts(),color='r')
        p.setp(leg7_2.get_lines(),color='r')
        p.setp(leg7_2.get_lines(),mec='r')
        leg7 = ax7.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=10,numpoints=1,handletextpad=.1,title='Expt; $\\mathregular{\\alpha}}$; Tube')
        p.setp(leg7.get_title(),fontsize=10)
        ax7.add_artist(leg7_2)
        ff.myax(fig7)
        
if not savefigs:
    p.show('all')        
else:
    fig1.savefig('{}/1 - Sts-Delta-Rot.png'.format(savepath),dpi=125)
    #fig1.savefig('{}/1 - Sts-Delta-Rot.pdf'.format(savepath),dpi=125)
    fig2.savefig('{}/2 - StrainPath.png'.format(savepath),dpi=125,bbox_inches='tight')
    #fig2.savefig('{}/2 - StrainPath.pdf'.format(savepath),dpi=125,bbox_inches='tight')
    fig3.savefig('{}/3 - RadialContraction.png'.format(savepath),dpi=125,bbox_inches='tight')
    #fig3.savefig('{}/3 - RadialContraction.pdf'.format(savepath),dpi=125,bbox_inches='tight')
    fig5.savefig('{}/3 - RadialContraction_LL.png'.format(savepath),dpi=125,bbox_inches='tight')
    #fig5.savefig('{}/3 - RadialContraction_LL.pdf'.format(savepath),dpi=125,bbox_inches='tight')
    fig4.savefig('{}/4 - Strain Profile.png'.format(savepath),dpi=125,bbox_inches='tight')
    #fig4.savefig('{}/4 - Strain Profile.pdf'.format(savepath),dpi=125,bbox_inches='tight')
    fig6.savefig('{}/6 - eeq_Triax.png'.format(savepath),dpi=125,bbox_inches='tight')
    #fig6.savefig('{}/6 - eeq_Triax.pdf'.format(savepath),dpi=125,bbox_inches='tight')
    fig7.savefig('{}/7 - eeq_Rot.png'.format(savepath),dpi=125,bbox_inches='tight')
    #fig7.savefig('{}/7 - eeq_Rot.pdf'.format(savepath),dpi=125,bbox_inches='tight')
    p.close('all')