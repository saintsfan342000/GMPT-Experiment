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
import os, glob, shutil, sys

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

- Analysis of Localization

'''
try:
    proj = sys.argv[1]
except:
    proj = 'GMPT-8_FS15SS5'
    
BIN = True
makecontours = True
saveAram = True                      # Whether to save the missing removed array to a .npy binary.  Only does so if BIN = False

analyze_basic_results = 0 # Whether to overwrite basic results
analyze_whole_field = 0 # Whether to take avg epsX and epsQ over whole surface
analyze_localization = 0 # Whether to scan localization for max point
analyze_LEpProfs = 0 # Whether to make LEpProf thru max point
analyze_urProfs = 0   # Whether to make urProf thru max point and using BF circ

print(" Have you added this experiment to the summary yet?\n"*4)
print(" And are the headerlines of 'ST.dat' commented out?\n"*4)

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
    # If that path don't exist then try Martin Deep drive
    if not os.path.exists(arampath):
        arampath = 'F:/GMPT/{}/AllPts'.format(proj)
    if not os.path.exists(arampath):
        raise FileNotFoundError('\nThe arampath does not exist.\n{}\n\n'.format(arampath))
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
    #[0]Pressure(psi)	[1]LVDT(V)  [2]Force(lbf)   [3]Disp.(in)    [4]FunctionGen Voltage [5]Time(s)
    LV = read_csv('./zMisc/GMPT-{:.0f}_LV.dat'.format(expt),
                  header=None, skiprows=1, comment='#', index_col=None,
                  skipinitialspace=True, sep='\s*', engine='python').values
    STPF = n.empty( (len(ST[:,0]),8) )
    # STPF will be:
    # [0]Stage, [1]Time, [2]Force(kip), [3]Pressure(ksi), [4]NomAxSts(ksi), [5]NomHoopSts(ksi), [6]LVDT(volt), [7]MTSDisp(in)
    STPF[:,[0,1]] = ST
    STPF[0,1] = LV[0,-1] # In case LV writes its first just after 0 sec
    LVint = interp1d(LV[:,-1],LV[:,:-1],axis=0).__call__(STPF[:,1]) # LVint is LV except time
    LVint[:,[0,2]]*=.001 #to kip, ksi)
    if LV.shape[1] == 5:
        # Then we do not have a Function Generator voltage
        STPF[:,[3,6,2,7]] = LVint
    elif LV.shape[1] == 6:
        # Then we do have a function generator voltage
        STPF[:,[3,6,2,7]] = LVint[:,:-1]  # Exclude FnGen Voltage
        n.savetxt('./zMisc/FnGen.dat',X=n.vstack((ST[:,0],LVint[:,-1])).T, fmt='%.0f, %.6f',header='[0]Stage, [1]FnGen Voltage')
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

###########################################################################
# Make the contour in which we select the area we analyze for localization
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

    stdLEP = nanstd(LEp)
    meanLEP = nanmean(LEp)

    laststgcontour = p.figure(1,facecolor='w',figsize=(6,12))
    p.tricontourf(x,y,LEp,linspace(meanLEP-2*stdLEP,meanLEP+2*stdLEP,256),extend='both',cmap='viridis')
    p.axis([n.min(x)*1.1, 1.1*n.max(x), n.min(y), n.max(y)])
    p.axis('equal')
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
    p.colorbar()
    p.draw()
    p.savefig('./LastStage.png')
    p.close()
else:
    Xmin,Xmax,Ymin,Ymax = n.genfromtxt('./zMisc/box_limits.dat',delimiter=',')
    box_x = array([Xmin , Xmax , Xmax , Xmin , Xmin])
    box_y = array([Ymin , Ymin , Ymax , Ymax , Ymin])

# Calculate the facet size and spacing relative to thickness
if not os.path.exists('./zMisc/facetsize.dat'):
    Atemp = A[ (abs(A[:,2])<=0.4) & (abs(A[:,3])<=0.5) , :]
    min_disp = n.min( pdist(Atemp[:,[2,3,4]]) )
    SS_th = (min_disp/thickness)
    FS_th = FS / SS * (min_disp/thickness)
    n.savetxt('./zMisc/facetsize.dat', X=[SS_th, FS_th], fmt='%.6f', header='[0]Step/thickness, [1]Size/thickness')
else:
    SS_th, FS_th = n.genfromtxt('./zMisc/facetsize.dat', delimiter=',')

# Localization direction, determined by alpha
# This is the orientation of the localization zone that develops
if alpha > 1.0:
    locdir = 'Hoop'
else:
    locdir = 'Axial'

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

######################
##### Initialize #####
#### Empty Arrays ####
######################
if True:
    # Main data array
    # [0] Stage, [1,2,3]eps_x,q,xq(point avg), [4]eps_x(1"ext), [5]eps_q(BFcirc@mid), [6]d/L
    D = n.empty((last+1, 7))
    # Linspace for best-fit circle
    yspace_bfc = linspace(-.1, .1, 3)[:,None]
    # Vertical profiles
    # First col:  Undeformed y-coords
    # next 4 cols:  -45 deg, 0deg, +45 deg, pratt BFC
    numpts = 4 * 8 # number of yspace points reduced for speed
    ur_profs = n.empty(( numpts,3*(last+1) + 1))*n.nan
    yspace_pro = linspace(-2,2,numpts)[:,None]
    ur_profs[:,[0]] = yspace_pro  # First column assigned
    # Initialize the strain profiles...want 5 wall thicknesses on each side of max point
    index_diff = int(round(10/SS_th)) # The number of pts on each side of max
    # Num rows:  2*number points on each side + the point itself
    # First col: Undef r-q (if circumf.) or undef y (if axial profile). Subsequent col is each stage
    LEp_prof = n.empty(( 2*index_diff + 1, 1*(last+1) + 1 ))*n.nan
    export_max=n.zeros( (last+1,9) )   #MaxPt data
    export_mean=n.zeros( (last+1,8) )  #MeanPt data
    export_stdv=n.zeros( (last+1,8) )  #Std Deviation data
    export_MaxPt=n.zeros( (last+1,7) ) #The last stage's max point traced thru all stages
    whole_field = n.zeros( (last+1 ,9) )

######################
# Iterate Thru Stage #
######################
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
    F = A[:,8:12].reshape(len(A[:,0]),2,2)   # A "stack" of 2x2 deformation gradients
    FtF = n.einsum('...ji,...jk',F,F)
    U = mysqrtm( FtF )     #Explicit calculation of sqrtm
    eigU = eigvalsh(U)  #Each row is the vector of eigenvalues for that row's matrix
    LE = n.log(eigU) #Element-wise
    LE0,LE1 = LE[:,0], LE[:,1]
    LEp = ne.evaluate('( 2/3 * ( LE0**2 + LE1**2 + (-LE0-LE1)**2 ) )**0.5')
    NEq = U[:,0,0] - 1 #Hoop
    NEx = U[:,1,1] - 1 #Axial
    NExq = U[:,0,1]
    Ro = n.sqrt(A[:,2]**2 + A[:,4]**2)
    R =  n.sqrt(A[:,5]**2 + A[:,7]**2)
    # Append R, Ro, Q to A
    A = n.hstack(( A, Ro[:,None], R[:,None], Q[:,None] ))

    # Localization analysis
    if analyze_localization and k > 0:
        # All these field quantites within the box have the suffix "1", so as not to override those var names outside the box
        LEp1, NEx1, NEy1, gamma1, xcoord1, ycoord1, aramX1, aramY1, NEx_alt1, NEy_alt1, gamma_alt1 = ( [] for _ in range(11) )
        #[0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) *)
        rng = (A[:,2]>=Xmin) & (A[:,2]<=Xmax) & (A[:,3]>=Ymin) & (A[:,3]<=Ymax)
        Abox = A[ rng, :]
        colLE0,colLE1 = LE[rng,0], LE[rng,1]
        colLEp = ne.evaluate('( 2/3 * ( colLE0**2 + colLE1**2 + (-colLE0-colLE1)**2 ) )**0.5')
        colNEx = U[rng,0,0] - 1
        colNEy = U[rng,1,1] - 1
        colNExy = U[rng,0,1]
        colG = n.arctan(colNExy/(1+colNEx)) + n.arctan(colNExy/(1+colNEy))
        colNEx_alt = F[rng,0,0]-1
        colNEy_alt = F[rng,1,1]-1
        colG_alt=n.arctan(F[rng,0,1]/F[rng,1,1]);
        # What we have effectively done is created an additional column for each data point.
        # Now I just need to sort though those which are in passable columns
        if locdir == 'Axial':
            parallel_indices = Abox[:,0].copy() # Aramis x index runs parallel to axial loc zone
            cross_index_col = 1 # Aram_y indices cross the loc. zone
        elif locdir == 'Hoop':
            parallel_indices = Abox[:,1].copy() # Aramis y index runs parallel to hoop loc band
            cross_index_col = 0
        for j in n.unique( parallel_indices ):
            rng = (parallel_indices == j)
            cross_ind = Abox[ rng, cross_index_col]
            if len(cross_ind) == (cross_ind.max() - cross_ind.min() +1):
                locLEp = n.argmax(colLEp[rng])                #Location of...
                LEp1.append( colLEp[rng][locLEp] )             #Max LEp in the current column
                NEx1.append( colNEx[rng][locLEp] )
                NEy1.append( colNEy[rng][locLEp] )
                gamma1.append( colG[rng][locLEp] )
                xcoord1.append( Abox[rng,2][locLEp] )
                ycoord1.append( Abox[rng,3][locLEp] )
                aramX1.append( Abox[rng,0][locLEp] )
                aramY1.append( Abox[rng,1][locLEp] )
                NEx_alt1.append( colNEx_alt[rng][locLEp] )
                NEy_alt1.append( colNEy_alt[rng][locLEp] )
                gamma_alt1.append( colG_alt[rng][locLEp] )

        LEp1, NEx1, NEy1, gamma1, xcoord1, ycoord1, aramX1, aramY1 = map(array,
                [LEp1, NEx1, NEy1, gamma1, xcoord1, ycoord1, aramX1, aramY1])    #Convert lists to arrays
        NEx_alt1, NEy_alt1, gamma_alt1 = map(array,[NEx_alt1, NEy_alt1, gamma_alt1])
        '''
        # Don't filter for now
        ratio = NEy1 / NEx1
        ratioAvg = nanmean(ratio)
        ratioSDEV = nanstd(ratio)
        passed = (ratio >= ratioAvg - 1 * ratioSDEV) & (ratio <= ratioAvg + 1 * ratioSDEV)
        '''
        # Need to add something, though, that doesn't accept a point if it doesn't have all its neighbors
        passed = n.ones_like(LEp1, dtype=bool)
        LEp1=LEp1[passed]
        NEx1=NEx1[passed]
        NEy1=NEy1[passed]
        gamma1=gamma1[passed]
        NEx_alt1=NEx_alt1[passed]
        NEy_alt1=NEy_alt1[passed]
        gamma_alt1=gamma_alt1[passed]
        xcoord1=xcoord1[passed]
        ycoord1=ycoord1[passed]
        aramX1 = aramX1[passed]
        aramY1 = aramY1[passed]

        ## Max 10 stages and identify I,J of max point in last
        locmax = n.argmax( LEp1 )
        if k == last:
            aramXmaxlast, aramYmaxlast = aramX1[locmax], aramY1[locmax]   #Save for making strain profiles
            MaxTen = n.flipud(array( vstack( (LEp1,aramX1,aramY1,xcoord1,ycoord1) ) ).T[n.argsort(LEp1),:][-10:,:])

        export_max[k] =  [ NEx1[locmax], NEy1[locmax], abs(gamma1[locmax]),
                           NEx_alt1[locmax], NEy_alt1[locmax], abs(gamma_alt1[locmax]),
                           LEp1[locmax], aramX1[locmax], aramY1[locmax] ]
        export_mean[k] = [ nanmean(NEx1), nanmean(NEy1), abs(nanmean(gamma1)),
                           nanmean(NEx_alt1), nanmean(NEy_alt1), abs(nanmean(gamma_alt1)),
                           nanmean(LEp1), sum(passed)]
        export_stdv[k] = [ nanstd(NEx1), nanstd(NEy1), abs( nanstd(gamma1) ),
                           nanstd(NEx_alt1), nanstd(NEy_alt1), abs(nanstd(gamma_alt1)),
                           nanstd(LEp1), sum(passed)]
        ## Store Last stage max point data
        ## Find the row in A (or depth-layer in F, or index in LEp, etc.) where the max point is
        rowmax = n.nonzero( (Abox[:,0]==aramXmaxlast) & (Abox[:,1]==aramYmaxlast) )[0]
        # In case the last stage max point doesn't show up in all stages, we have a if statement
        if len(rowmax) != 0:
            export_MaxPt[k] = [ colNEx[rowmax], colNEy[rowmax], colG[rowmax],
                                colNEx_alt[rowmax], colNEy_alt[rowmax], colG_alt[rowmax], colLEp[rowmax] ]
        else:
            # If max point doesn't show up in this stage, then populate with nans
            export_MaxPt[k] = [ n.nan, n.nan, n.nan, n.nan, n.nan, n.nan, n.nan ]
    else:
        # k == 0 so don't bother with a localiz'n analysis
        export_max[k] =  [0,0,0,0,0,0,0,0,0]
        export_mean[k] = [0,0,0,0,0,0,0,0]
        export_stdv[k] = [0,0,0,0,0,0,0,0]
        export_MaxPt[k]= [0,0,0,0,0,0,0]

    ######################
    #### LEp profiles ####
    ######################
    if analyze_LEpProfs:
        if k==last and (analyze_localization == False):
            # Then we need the max point aramX and aramY from Max10.dat since we won't get it from loc analysis
            aramXmaxlast, aramYmaxlast = n.genfromtxt('Max10.dat', delimiter=',')[0,[1,2]]
        rowmax = n.nonzero( (A[:,0]==aramXmaxlast) & (A[:,1]==aramYmaxlast) )[0]
        # Max point data is in Abox[rowmax]
        if locdir == 'Axial' and k>0 and len(rowmax)!=0:
            # Interpolating with a y-coord identical to max point and 10-thicknesses on either side R-q
            RQ = A[:,12]*A[:,14]*n.pi/180   # Arc length
            RQ_max = A[rowmax,12]*A[rowmax,14]*n.pi/180
            RQspace = linspace( RQ_max-10*thickness, RQ_max+10*thickness,  len(LEp_prof[:,0]) )
            rng = (A[:,0]==A[rowmax,0]) & (RQ>=RQ_max-12*thickness) & (RQ<=RQ_max+12*thickness)
            LEp_prof[:,k+1] = interp1d(RQ[rng],LEp[rng]).__call__(RQspace)
            if k == last:
                LEp_prof[:,0] = RQspace - RQ_max
        elif locdir == 'Hoop' and k>0 and len(rowmax)!=0:
            # Interpolating with a x coord identical to max point and 10-thicknesses above and below
            y_max = A[rowmax,3]
            yspace = linspace(y_max-10*thickness, y_max+10*thickness, len(LEp_prof[:,0]))
            rng = (A[:,1]==A[rowmax,1]) & (A[:,3]>=y_max-12*thickness) & (A[:,3]<=y_max+12*thickness)
            LEp_prof[:,k+1] = interp1d(A[rng,3],LEp[rng]).__call__(yspace)
            if k == last:
                LEp_prof[:,0] = yspace - y_max
        elif len(rowmax)==0:
            LEp_prof[:,k+1] = n.nan
        else:
            LEp_prof[:,k+1] = 0

    ######################
    #### Basic Results ###
    ## Lots of griddata ##
    ######################
    if analyze_basic_results:
        # Pointwise average
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
        D[k,3] = NExq[rng][keeps].mean()       # Virtual axial extensometer
        # Interp at +/- 0.5" undeformed for axial
        rng = (Q <= q_mid+10) & (Q >= q_mid-10)
        xspace = linspace( A[rng,2].min(), A[rng,2].max(), 2*len(n.unique(A[rng,1])) )
        Atemp = A[(A[:,3]>=.5-thickness) & (A[:,3]<=0.5+thickness)]
        x_hi = griddata( Atemp[:,[2,3]], Atemp[:,6], (xspace[None,:],array([[.5]])))[0].mean()
        Atemp = A[(A[:,3]<=-0.5+thickness) & (A[:,3]>=-0.5-thickness)]
        x_lo = griddata( Atemp[:,[2,3]], Atemp[:,6], (xspace[None,:],array([[-.5]])))[0].mean()
        # Assign
        D[k,4] = (x_hi-x_lo)-1
        
        # BF circ at +0.1, 0, and -0.1
        # Do I still need this?  Can't I just pull the data our of ur_profs??
        rng = (Q <= q_mid + 30) & (Q >= q_mid - 30) & (abs(A[:,3])<.2)
        xspace = linspace( A[rng,2].min(), A[rng,2].max(), 2*len(n.unique(A[rng,1])) )
        rng =  (
                (Q <= q_mid + 32) & 
                (Q >= q_mid - 32) & 
                (
                 ((A[:,3]<=0.1+thickness) & (A[:,3]>=0.1-thickness)) | 
                 ((A[:,3]<=0.0+thickness) & (A[:,3]>=0.0-thickness)) | 
                 ((A[:,3]<=-0.1+thickness) & (A[:,3]>=-0.1-thickness))
                )
               )
        Atemp = A[rng]
        xzinterp = griddata( Atemp[:,[2,3]], Atemp[:,[2,4,5,7]], (xspace[None,:], yspace_bfc), method='linear')
        # This interpolatin is of shape ( len(yspace_bfc), len(xspace), len(Atemp[0]) )
        # i.e. (number of y pts, number of x pts, number of columns interpolated)
        Ro, R = 0.0, 0.0
        for i in range(yspace_bfc.shape[0]):
            Ro += CF(xzinterp[i,:,[0,1]].T)[-1]  # Have to transpose it b/c the slice makes it 2 x n, want nx2
            R += CF(xzinterp[i,:,[2,3]].T)[-1]
        # Assign
        D[k,5] = R/Ro - 1
        # Calculate delta from edges
        lower = A[ (A[:,3] >= rdlim[0]) & (A[:,3] <= rdlim[1]) & (n.abs(A[:,2]) <= 0.2), :]
        upper = A[ (A[:,3] >= rdlim[2]) & (A[:,3] <= rdlim[3]) & (n.abs(A[:,2]) <= 0.2), :]
        D[k,6] = n.mean(upper[:,6]) - n.mean(lower[:,6])

    ######################
    #### Ur/R profiles ###
    ######################
    if analyze_urProfs:
        if k==last and (analyze_localization == False):
            # Then we need the max point aramX and aramY from Max10.dat since we won't get it from loc analysis
            aramXmaxlast, aramYmaxlast = n.genfromtxt('Max10.dat', delimiter=',')[0,[1,2]]
        
        # Vertical profile thru max point
        rng = (A[:,-1]>=q_mid-45) & (A[:,-1]<=q_mid+45) 
        rng = (A[:,1] == aramYmaxlast)
        ur_profs[:, k*3+1] = interp1d(A[rng,3],NEq[rng], fill_value='extrapolate').__call__(yspace_pro.ravel())       
        Atemp = A[ A[:,1] == aramYmaxlast ]
        ur_profs[:, k*3+2] = interp1d(Atemp[:,3],Atemp[:,13]/Atemp[:,12]-1, fill_value='extrapolate').__call__(yspace_pro.ravel())       
        
        # BF Circle along length for Ur/R profile
        # Looping b/c there is great time-savings in using a minimal size Atemp within griddata
        # This also allows me to use a much larger qspace, providing better CFs
        for i,y in enumerate(yspace_pro.ravel()):
            Atemp = A [(A[:,3]>=y-thickness) & (A[:,3]<=y+thickness) ]
            q0, q1 = Atemp[:,-1].min()+10, Atemp[:,-1].max()-10
            qspace = linspace( q0, q1, 2*len(n.unique(Atemp[:,1])) )
            xzinterp = griddata( Atemp[:,[-1,3]], Atemp[:,[2,4,5,7]], (qspace[None,:], yspace_pro), method='linear')
            Ro = CF(xzinterp[i,:,[0,1]].T)[-1]  # Have to transpose it b/c the slice makes it 2 x n, want nx2
            R = CF(xzinterp[i,:,[2,3]].T)[-1]
            ur_profs[i, k*3+3] = R/Ro - 1
            
    ######################
    ##### whole_field ####
    ######################
    if analyze_whole_field:
        whole_field[k,0] = k
        qrng = (Q>=q_mid-45) & (Q<=q_mid+45)
        rng = qrng & (n.abs(A[:,3])<=0.5)
        whole_field[k,1] = n.nanmean(NEx[rng])
        whole_field[k,2] = n.nanmean(NEq[rng])
        rng = qrng & (n.abs(A[:,3])<=1.0)
        whole_field[k,3] = n.nanmean(NEx[rng])
        whole_field[k,4] = n.nanmean(NEq[rng])
        rng = qrng & (n.abs(A[:,3])<=1.5)
        whole_field[k,5] = n.nanmean(NEx[rng])
        whole_field[k,6] = n.nanmean(NEq[rng])
        rng = qrng & (n.abs(A[:,3])<=1.9)
        whole_field[k,7] = n.nanmean(NEx[rng])
        whole_field[k,8] = n.nanmean(NEq[rng])
        
#################
## End of Loop ##
#################

# delta/L
GL = D[0,6]
D[:,6]-=GL
D[:,6]*=(1/GL)

# Correct stage 0
D[0,:] = 0

# Save data files!
#########
if analyze_basic_results:
    # Results.dat
    headerline = '[0] Stage, [1,2,3]eps_x,q,xq(point avg), [4]eps_x(1"ext), [5]eps_q(BFcirc@mid), [6]d/L, Lg={:.6f}'.format(GL)
    n.savetxt('Results.dat', X=D, fmt='%.0f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f',header=headerline)
if analyze_localization:
    # loc_max.dat
    headerline='[0]NEx [1]NEy [2]Gamma [3]F11-1 [4]F22-1 [5]atan(F12/F22) [6]epeq [7]AramX [8]AramY'
    n.savetxt('loc_max.dat', X=export_max, fmt='%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.0f, %.0f',header=headerline)
    # loc_mean.dat
    headerline='[0]NEx [1]NEy [2]Gamma [3]F11-1 [4]F22-1 [5]atan(F12/F22) [6]epeq [7]Num pts'
    n.savetxt('loc_mean.dat', X=export_mean, fmt='%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.0f',header=headerline)
    # loc_std.dat
    headerline='[0]NEx [1]NEy [2]Gamma [3]F11-1 [4]F22-1 [5]atan(F12/F22) [6]epeq [7]Num pts'
    n.savetxt('loc_std.dat', X=export_stdv, fmt='%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.0f',header=headerline)
    # MaxPt.dat
    headerline='Last Stage Last Point, traced thru whol exp.\n[0]NEx [1]NEy [2]Gamma [3]F11-1 [4]F22-1 [5]atan(F12/F22) [6]epeq'
    n.savetxt('MaxPt.dat', X=export_MaxPt, fmt='%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f',header=headerline)
    # Save the Max10
    headerline = '[0]Epeq [1]AramXIndex [2]AramYIndex [3]UndefXCoord [4]UndefYCoord'
    n.savetxt('Max10.dat',X=MaxTen,fmt='%.6f, %.0f, %.0f, %.6f, %.6f',header=headerline)
if analyze_urProfs:
    # ur_profiles.dat
    headerline = ('First column:  Undef y-coord\n' +
                 'k+1 to k+3 column:  Stage K eps_theta, ur/Ro for vert profile thru max point, and using BFcirc')
    n.savetxt('ur_profiles.dat', X=ur_profs, fmt='%.6f', delimiter=',', header=headerline)
if analyze_LEpProfs:
    # Save the LEp_profiles
    headerline = ('Col [0]: Undeformed RQ or y coord.\n' +
                  'Col [k+1]: Stage k LEp along profile')
    n.savetxt('LEp_profiles.dat', X=LEp_prof, fmt='%.6f', delimiter=',', header=headerline)
if analyze_whole_field:
    # Save whole_field
    headerline = ('[0]Stage, [1,2]Epsx,q, +/-.5", [3,4]1", [5,6]1.5", [7,8] 1.9"' + 
                  'Central 90 deg of points')
    whole_field[0] = 0
    n.savetxt('WholeFieldAverage.dat', X=whole_field, fmt='%.0f'+' , %.6f'*8, header=headerline)

os.system('python ../AA_PyScripts/B_Plots.py {} {} {} {}'.format(int(expt), int(FS), int(SS), savepath))
