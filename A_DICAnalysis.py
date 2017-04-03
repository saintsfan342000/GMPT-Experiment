import numpy as n
from numpy import (dstack, vstack, hstack,
                   linspace, array, nanmean, nanstd)
from numpy.linalg import eigvalsh
abs = n.abs
import matplotlib.pyplot as p
from scipy.interpolate import griddata, interp1d, LinearNDInterpolator
from scipy.spatial.distance import pdist
import numexpr as ne
from pandas import read_excel, read_csv
from mysqrtm import mysqrtm
from CircleFitByPratt import CircleFitByPratt as CF
from CircleFitByPratt import PrattPlotter
import os, glob, shutil

'''
Analysis code for GMPT experiments
sig_x = F/(2piRt) + PR/2t
sig_q = PR/t

Plots we need
- sig_x vs sig_q
- tau_x vs tau_q (true sts)
- sig_x vs eps_x
- sig_q vs eps_q

Things to calculate
- Hoop stn along length
- Calc axial stn using d/L from edges, d/L 1" GL, avg of point stn
- Calc hoop stn using ur/R and average of points

Questions
- What is average strain measurement?
	Delta/L, from edges?  In Kork. you used a 1" GL axial extens.
	
Outstanding
- Analysis of Localization needs special treatment

'''
proj = 'GMPT-2_FS30SS10'
BIN = False
makecontours = False
saveAram = True                      # Whether to save the missing removed array to a .npy binary.  Only does so if BIN = False

print(" Have you added this experiment to the summary yet?")
print(" Have you added this experiment to the summary yet?")
print(" Have you added this experiment to the summary yet?")
print(" Have you added this experiment to the summary yet?")
print(" Have you added this experiment to the summary yet?")

expt = int( proj.split('_')[0].split('-')[1])
FS = int( proj.split('_')[1].split('SS')[0].split('S')[1] )
SS = int( proj.split('_')[1].split('SS')[1] )

if BIN:
    arampath = '../{}/AramisBinary'.format(proj)
    prefix = '{}_'.format(proj)
    #last = len( glob.glob( '{}/{}*.npy'.format(arampath,prefix) ) ) - 1
    # calculation of last moved to STPF[-1,0] rather than last aramfile
else:
    arampath = 'D:/Users/user/Documents/AAA Scales/{}/AllPts'.format(proj)
    prefix = '{}-Stage-0-'.format(proj)
    #last = len( glob.glob( '{}/{}*.txt'.format(arampath,prefix) ) ) - 1       

savepath = '../{}'.format(proj)
saveprefix = '{}_'.format(proj)          #Prefix for the npy files

key = n.genfromtxt('../ExptSummary.dat', delimiter=',')
if not (expt in key[:,0]):
    raise ValueError('You need to update the experiment summary!')
else:
    expt, date, material, tube, alpha, a_true, Rm, thickness, ecc = key[ key[:,0] == expt, : ].ravel()

os.chdir(savepath)

###########################################################################
# Make STPF file
###########################################################################
if not os.path.exists('STPF.dat'):
    ST = n.genfromtxt('./zMisc/ST.dat', delimiter=',')
    last = int(ST[-1,0])
    ## Read csv, flexible enough for spaces and tabs
    #[0]Pressure(psi)	[1]LVDT(V)  [2]Force(lbf)   [3]Disp.(in)    [4]Time(s)
    LV = read_csv('./zMisc/GMPT-{:.0f}_LV.dat'.format(expt),header=None,skiprows=1,comment='#',index_col=None,skipinitialspace=True,sep='\s*',engine='python').values
    STPF = n.empty( (len(ST[:,0]),8) )
    # STPF will be:
    # [0]Stage, [1]Time, [2]Force(kip), [3]Pressure(ksi), [4]NomAxSts(ksi), [5]NomHoopSts(ksi), [6]LVDT(volt), [7]MTSDisp(in)
    STPF[:,[0,1]] = ST
    STPF[0,1] = LV[0,-1] # In case LV writes its first just after 0 sec
    LVint = interp1d(LV[:,-1],LV[:,:-1],axis=0).__call__(STPF[:,1])
    LVint[:,[0,2]]*=.001 #to kip, ksi)
    STPF[:,[3,6,2,7]] = LVint
    STPF[:,4] = (STPF[:,2]/(2*n.pi*Rm*thickness) + STPF[:,3]*Rm/(2*thickness))
    STPF[:,5] = STPF[:,3]*Rm/thickness
    STPF[:,[6,7]]-=STPF[0,[6,7]]    #Initialize MTS disp and LVDT
    headerline = '[0]Stage, [1]Time, [2]Force(kip), [3]Pressure(ksi), [4]NomAxSts(ksi), [5]NomHoopSts(ksi), [6]LVDT(volt), [7]MTSDisp(in)'
    n.savetxt('STPF.dat',X=STPF,fmt='%.0f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f',header=headerline)
else:
    STPF = n.loadtxt('STPF.dat',delimiter=',')    
    last = int(STPF[-1,0])


if BIN == True:
    A = n.load('{}/{}{:.0f}.npy'.format(arampath,prefix,last))
else:
    A = read_csv('{}/{}{:.0f}.txt'.format(arampath,prefix,last),sep=',',na_values=' ',skiprows=3,header=None,index_col=None).values
    A = A[ ~n.any(n.isnan(A),axis=1), :]
    #[0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) *)

'''
###########################################################################
# Make the contour in which we select the area we analyze for localization
# IGNORING FOR NOW
###########################################################################
if not os.path.exists('./zMisc/box_limits.dat'):
    x=A[:,2]
    y=A[:,3]

    #Calculation for each facet of the Logarithmic cumulative plastic strain
    F=A[:,-4:].reshape(len(A[:,0]),2,2) 
    #FtF = n.einsum('...ij,...jk',F.transpose((0,2,1)),F)
    FtF = n.einsum('...ji,...jk',F,F) #Same as that commented out above. Kept the comment to recall how to transpose a stack of matrices
    U = mysqrtm( FtF )
    eigU = eigvalsh(U) #dimension is len(A[:,0]) x 2.  Each row is the vector of eigenvalues for that row's matrix
    LE = n.log(eigU)
    LE0,LE1 = LE[:,0], LE[:,1]
    LEp = ne.evaluate('( 2/3 * ( LE0**2 + LE1**2 + (-LE0-LE1)**2 ) )**0.5')

    meanLEP = nanmean(LEp)

    laststgcontour = p.figure(1,facecolor='w',figsize=(16,12))
    p.tricontourf(x,y,LEp,linspace(meanLEP-1*stdLEP,meanLEP+5*stdLEP,256),extend='both',cmap='viridis')
    p.axis([n.min(x)*1.1, 1.1*n.max(x), n.min(y), n.max(y)])
    #p.axis('equal')
    p.grid(True,linestyle='-',linewidth=2)
    p.xlabel('Undeformed X (in)')
    p.ylabel('Undeformed Y (in)')

    crop = n.asarray( p.ginput(2) )
    Xmin = n.min( crop[:,0] )
    Xmax = n.max( crop[:,0] )
    Ymin = n.min( crop[:,1] )
    Ymax = n.max( crop[:,1] )
    
    headerline='Xmin, Xmax, Ymin, Ymax (all in inches!)'
    n.savetxt('./zMisc/box_limits.dat',X=array([Xmin, Xmax, Ymin, Ymax])[None,:],fmt='%.6f',delimiter=', ',header=headerline)
    
    box_x = array([Xmin , Xmax , Xmax , Xmin , Xmin])
    box_y = array([Ymin , Ymin , Ymax , Ymax , Ymin])
    p.plot(box_x,box_y,'w',linewidth=2.5)
    p.grid(False)
    p.draw()
else:
    Xmin,Xmax,Ymin,Ymax = n.genfromtxt('./zMisc/box_limits.dat',delimiter=',')
    box_x = array([Xmin , Xmax , Xmax , Xmin , Xmin])
    box_y = array([Ymin , Ymin , Ymax , Ymax , Ymin])
'''

# Calculate the facet size and spacing relative to thickness
if not os.path.exists('./zMisc/facetsize.dat'):
    Atemp = A[ (abs(A[:,2])<=0.2) & (abs(A[:,3])<=0.1) , :]
    min_disp = n.min( pdist(Atemp[:,[2,3,4]]) )
    SS_th = (min_disp/thickness)
    FS_th = FS / SS * (min_disp/thickness)
    n.savetxt('./zMisc/facetsize.dat', X=[SS_th, FS_th], fmt='%.6f', header='[0]Step/thickness, [1]Size/thickness')

# Create the 4 stations at which we'll plot profiles
# We want the LL to be number 3
LL = STPF[:,3].argmax()
if not os.path.exists('./zMisc/prof_stages.dat'):
    # [0]Stage, [1]Time, [2]Force(kip), [3]Pressure(ksi), [4]NomAxSts(ksi), [5]NomHoopSts(ksi), [6]LVDT(volt), [7]MTSDisp(in)
    STPF = read_csv('STPF.dat',sep=',',comment='#',header=None,index_col=None).values
    p.figure()
    p.plot(STPF[:,6], STPF[:,3])
    p.plot(STPF[LL,6], STPF[LL,3],'ro',ms=10)
    p.title('Click where the first\ncalc stage will be', fontsize=30)
    p.axis([0, 1.1*STPF[:,6].max(), 0, 1.1*STPF[:,3].max()])
    loc1 = n.asarray(p.ginput(1)).flatten()[0]
    p.close()

    #Two before the two before LL, LL, one at last stage
    x = hstack(( linspace(loc1,STPF[LL,6],3), STPF[-1,6]))
    profStg = n.empty(len(x), dtype=int)
    for i in range(len(x)):
        profStg[i] = n.where(STPF[:,6]>=x[i])[0][0]
        
    headerline = 'Stages at which profiles were generated'
    n.savetxt('./zMisc/prof_stages.dat',X=profStg[None,:],fmt='%.0f',delimiter=', ',header=headerline)
else:
    profStg = n.genfromtxt('./zMisc/prof_stages.dat',delimiter=',', dtype=int)
'''
# Go thru the stages and calc Ur/R
urR = n.empty(last+1)
for k in range(last+1):
    if BIN == True:
        A = n.load('{}/{}{:.0f}.npy'.format(arampath,prefix,k))
    else:
        A = read_csv('{}/{}{:.0f}.txt'.format(arampath,prefix,k),sep=',',na_values=' ',skiprows=3,header=None,index_col=None).values
        A = A[ ~n.any(n.isnan(A),axis=1), :]
        if saveAram:
            n.save('./AramisBinary/{}{:.0f}'.format(saveprefix,k),A)
    #[0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) *)
    A = A[ n.abs(A[:,3])<0.1, :]
    # Interp deformed x and zoord from a linspace of x coors
    xspace = linspace(.9*A[:,5].min(), .8*A[:,5].max(), n.unique(A[:,1]).shape[0])  
    xzint = griddata( A[:,[5,3]], A[:,[5,7]], (xspace[None,:],n.array([[0]])), method='linear')[0]
    R = CF(xzint)[-1]
    if k == 0:
        Ro = R
    urR[k] = R/Ro - 1
'''

if not os.path.exists('./zMisc/disp_limits.dat'):
    ## Now identify the upper and lower ranges for calculating delta of the thick edges
    #[0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) *)
    p.figure(figsize=(16,12))
    p.title('Click the four points that bound the cusp of the thick edges',size=20)
    p.plot(A[:,3],A[:,4],'.',alpha=0.5,ms=4,markevery=2)
    p.axis([-2.3, -1.6, 0.8, 1])
    rdlim = n.sort(n.asarray(p.ginput(2))[:,0])
    p.close()
    p.figure(figsize=(16,12))
    p.plot(A[:,3],A[:,4],'.',alpha=0.5,ms=4,markevery=2)
    p.axis([1.6, 2.3, 0.8, 1])
    rdlim = n.sort( hstack( (rdlim, n.asarray(p.ginput(2))[:,0]) ) )
    p.close()
    headerline='Lower sxn Ymin, Lower sxn Ymax, Upper sxn Ymin, Upper sxn Ymax'
    n.savetxt('./zMisc/disp_limits.dat',X=rdlim[None,:],fmt='%.6f',delimiter=', ',header=headerline)
else:
    rdlim = n.sort(n.genfromtxt('./zMisc/disp_limits.dat',delimiter=','))

# Initialize and define some things
# Main data array
# [0] Stage, [1,2,3]eps_x,q,xq(point avg), [4]eps_x(1"ext), [5]eps_q(BFcirc@mid), [6]d/L
D = n.empty((last+1, 7))
# Linspace for best-fit circle
yspace_bfc = linspace(-.1, .1, 3)[:,None]
# Vertical profiles
# First col:  Undeformed y-coords
# next 4 cols:  -45 deg, 0deg, +45 deg, pratt BFC
numpts = len( n.unique(A[:,0])) # number of yspace points
PROFS = n.empty(( numpts,4*(last+1) + 1))*n.nan
yspace_pro = linspace(-2,2,numpts)[:,None]
PROFS[:,[0]] = yspace_pro  # First column assigned

#Cycle through the stages
for itcount, k in enumerate(range(last,-1,-1)):
    print('{}. Stage {}'.format(proj,k))
    if BIN:
        A = n.load('{}/{}{:.0f}.npy'.format(arampath,prefix,k))
    else:
        A = read_csv('{}/{}{:.0f}.txt'.format(arampath,prefix,k),sep=',',na_values=' ',skiprows=3,header=None,index_col=None).values
        A = A[ ~n.any(n.isnan(A),axis=1), :]
        if saveAram:
            if not os.path.exists('./AramisBinary'):
                os.mkdir('./AramisBinary')
            n.save('./AramisBinary/{}{:.0f}'.format(saveprefix,k),A)
    #[0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) *)
    Q = n.arctan2(A[:,2], A[:,4])*180/n.pi
    q_rng = Q.max()-Q.min()
    q_mid = Q.min()+q_rng/2
    F = A[:,-4:].reshape(len(A[:,0]),2,2)   # A "stack" of 2x2 deformation gradients 
    FtF = n.einsum('...ji,...jk',F,F)
    U = mysqrtm( FtF )     #Explicit calculation of sqrtm
    #eigU = eigvalsh(U)  #Each row is the vector of eigenvalues for that row's matrix
    #LE = n.log(eigU) #Element-wise
    #LE0,LE1 = LE[:,0], LE[:,1]
    #LEp = ne.evaluate('( 2/3 * ( LE0**2 + LE1**2 + (-LE0-LE1)**2 ) )**0.5')
    NEq = U[:,0,0] - 1 #Hoop
    NEx = U[:,1,1] - 1 #Axial
    NExq = U[:,0,1]
    Ro = n.sqrt(A[:,2]**2 + A[:,4]**2)
    R =  n.sqrt(A[:,5]**2 + A[:,7]**2)
    # Append R, Ro, Q to A
    A = n.hstack(( A, Ro[:,None], R[:,None], Q[:,None] ))
    
    # Point Coordinate Range.  abs(Y)<0.5 and +/- 20 degrees from q_mid
    rng = (abs(A[:,3]) < 0.5) & (Q <= q_mid+20) & (Q >= q_mid-20)
    # rng = (abs(A[:,3]) < 2) & (Q <= q_mid+20) & (Q >= q_mid-20) & (abs(A[:,3]) >1)  #### Higher up analysis
    # Filter based on axial-hoop strain ratio
    ratmean = (NEx/NEq)[rng].mean()
    ratdev = (NEx/NEq)[rng].std()
    keeps = ((NEx/NEq)[rng]<=ratmean+1*ratdev) & ((NEx/NEq)[rng]>=ratmean-1*ratdev)
    # Assign
    D[k,0] = k
    D[k,1] = NEx[rng][keeps].mean()
    D[k,2] = NEq[rng][keeps].mean()
    D[k,3] = NExq[rng][keeps].mean()

    # Virtual axial extensometer 
    # Interp at +/- 0.5" undeformed for axial
    rng = (Q <= q_mid+10) & (Q >= q_mid-10)
    xspace = linspace( A[rng,2].min(), A[rng,2].max(), 2*len(n.unique(A[rng,1])) )
    Atemp = A[(A[:,3]>=0.4) & (A[:,3]<=0.6)]
    x_hi = griddata( Atemp[:,[2,3]], Atemp[:,6], (xspace[None,:],array([[.5]])))[0].mean()
    Atemp = A[(A[:,3]<=-0.4) & (A[:,3]>=-0.6)]
    x_lo = griddata( Atemp[:,[2,3]], Atemp[:,6], (xspace[None,:],array([[-.5]])))[0].mean()
    # Assign
    D[k,4] = (x_hi-x_lo)-1

    # BF circ at +0.1, 0, and -0.1
    rng = (Q <= q_mid + .3*q_rng) & (Q >= q_mid - .4*q_rng) & (abs(A[:,3])<1)
    xspace = linspace( A[rng,2].min(), A[rng,2].max(), 2*len(n.unique(A[rng,1])) )
    Atemp = A[(A[:,3]>=-0.2) & (A[:,3]<=0.2)]
    #Atemp = A[(A[:,3]>=1.2) & (A[:,3]<=2)]  #### Higher up analysis
    xzinterp = griddata( Atemp[:,[2,3]], Atemp[:,[2,4,5,7]], (xspace[None,:], yspace_bfc), method='linear')
    # This interpolatin is of shape ( len(yspace_bfc), len(xspace), len(Atemp[0]) )
    # i.e. (number of y pts, number of x pts, number of columns interpolated)
    Ro, R = 0.0, 0.0
    for i in range(yspace_bfc.shape[0]):
        Ro += CF(xzinterp[i,:,[0,1]].T)[-1]  # Have to transpose it b/c the slice makes it 2 x n, want nx2
        R += CF(xzinterp[i,:,[2,3]].T)[-1]
    # Assign
    D[k,5] = R/Ro - 1

    # BF Circles from -2 to +2
    # Consider finding out the min and max x by finding out the x coord of the most extreme but unbroken j-index column
    # The wider the arc over which we BF circle, the smoother the data
    '''
    Atemp = A[ n.abs(A[:,3] ) =< 2 ]
    length, mins, maxes = [], [], []
    for K,J in enumerate(n.unique(Atemp[:,1])):
        Atemp = A[ A[:,1] == J ]
    '''
    rng = (Q <= q_mid + 30) & (Q >= q_mid - 30)
    Atemp = A[(A[:,3]>=1.2) & (A[:,3]<=2)]  #### Higher up analysis
    xzinterp = griddata( A[:,[2,3]], A[:,[2,4,5,7]], (xspace[None,:], yspace_pro), method='linear')
    for i in range(yspace_pro.shape[0]):
        if n.any( n.isnan( xzinterp[i] ) ):
            pass
            print('fuck')
        else:
            Ro = CF(xzinterp[i,:,[0,1]].T)[-1]  # Have to transpose it b/c the slice makes it 2 x n, want nx2
            R = CF(xzinterp[i,:,[2,3]].T)[-1]
            PROFS[i, k*4+4] = R/Ro - 1

    # Vertical profiles
    # Interp based on undef Q and undef Y to get R/Ro - 1
    rng = (Q <= q_mid + 30) & (Q >= q_mid - 30)
    #xspace = n.hstack(( -16/25.4, linspace( A[rng,2].min(), A[rng,2].max(), 3 ), 20/25.4))
    xspace = linspace( A[rng,2].min(), A[rng,2].max(), 3 )
    PROFS[:, k*4+1:k*4+4]  = griddata( A[:,[2,3]], A[:,-2]/A[:,-3] - 1, (xspace[None,:], yspace_pro), method='linear')
    
    # Calculate delta from edges
    lower = A[ (A[:,3] >= rdlim[0]) & (A[:,3] <= rdlim[1]) & (n.abs(A[:,2]) <= 0.2), :]
    upper = A[ (A[:,3] >= rdlim[2]) & (A[:,3] <= rdlim[3]) & (n.abs(A[:,2]) <= 0.2), :]
    D[k,6] = n.mean(upper[:,6]) - n.mean(lower[:,6])


    
#########
# End of Loop
#########

# delta/L 
GL = D[0,6]
D[:,6]-=GL
D[:,6]*=(1/GL) 

# Correct stage 0
D[0,:] = 0

# Save data files!
#########
#[0] Stage, [1,2,3]eps_x,q,xq(point avg), [4]eps_x(1"ext), [5]eps_q(BFcirc@mid), [6]d/L
headerline = '[0] Stage, [1,2,3]eps_x,q,xq(point avg), [4]eps_x(1"ext), [5]eps_q(BFcirc@mid), [6]d/L, Lg={:.6f}'.format(GL)
n.savetxt('Results.dat', X=D, fmt='%.0f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f',header=headerline)
headerline = ('First column:  Undef y-coord\n' +
             'k+1 to k+4 column:  Stage K ur/Ro for vert profile at X={:.3f}, {:.3f}, {:.3f} inches and using BFcirc'.format(*xspace))
n.savetxt('Profiles.dat', X=PROFS, fmt='%.6f', delimiter=',', header=headerline)


