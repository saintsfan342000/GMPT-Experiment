def CircleFitByPratt(XY):
    '''
    %--------------------------------------------------------------------------
    %  
    %     Circle fit by Pratt
    %      V. Pratt, "Direct least-squares fitting of algebraic surfaces",
    %      Computer Graphics, Vol. 21, pages 145-152 (1987)
    %
    %     Input:  XY(n,2) is the array of coordinates of n points x(i)=XY(i,1), y(i)=XY(i,2)
    %
    %     Output: Par = [a b R] is the fitting circle:
    %                           center (a,b) and radius R
    %
    %     Note: this fit does not use built-in matrix functions (except "mean"),
    %           so it can be easily programmed in any programming language
    %
    %--------------------------------------------------------------------------
    '''

    from numpy import array, shape, mean, sqrt, hstack

    n = XY.shape[0]      # number of data points

    centroid = mean(XY, axis=0)   # the centroid of the data set

    #     computing moments (note: all moments will be normed, i.e. divided by n)

    Mxx, Myy, Mxy, Mxz, Myz, Mzz = 0, 0, 0, 0, 0, 0

    for i in range(n):
        Xi = XY[i,0] - centroid[0]  #  centering data
        Yi = XY[i,1] - centroid[1]  #  centering data
        Zi = Xi*Xi + Yi*Yi
        Mxy = Mxy + Xi*Yi
        Mxx = Mxx + Xi*Xi
        Myy = Myy + Yi*Yi
        Mxz = Mxz + Xi*Zi
        Myz = Myz + Yi*Zi
        Mzz = Mzz + Zi*Zi
       
    Mxx = Mxx/n
    Myy = Myy/n
    Mxy = Mxy/n
    Mxz = Mxz/n
    Myz = Myz/n
    Mzz = Mzz/n

    #    computing the coefficients of the characteristic polynomial

    Mz = Mxx + Myy
    Cov_xy = Mxx*Myy - Mxy*Mxy
    Mxz2 = Mxz*Mxz
    Myz2 = Myz*Myz

    A2 = 4*Cov_xy - 3*Mz*Mz - Mzz
    A1 = Mzz*Mz + 4*Cov_xy*Mz - Mxz2 - Myz2 - Mz*Mz*Mz
    A0 = Mxz2*Myy + Myz2*Mxx - Mzz*Cov_xy - 2*Mxz*Myz*Mxy + Mz*Mz*Cov_xy
    A22 = A2 + A2

    epsilon=1e-12
    ynew=1e+20
    IterMax=20
    xnew = 0

    #    Newton's method starting at x=0
    k=1
    for i in range(IterMax):
        yold = ynew
        ynew = A0 + xnew*A1 + xnew*xnew*A2 + 4*xnew*xnew*xnew*xnew
        if (abs(ynew)>abs(yold)):
            print('Newton-Pratt goes wrong direction: |ynew| > |yold|');
            xnew = 0
            break
        Dy = A1 + xnew*(A22 + 16*xnew*xnew)
        xold = xnew
        xnew = xold - ynew/Dy
        if (abs((xnew-xold)/xnew) < epsilon):
             break
        if (i >= IterMax):
            print('Newton-Pratt will not converge')
            xnew = 0
        if (xnew<0.):
            print('Newton-Pratt negative root:  x={}'.format(xnew))
            xnew = 0

    #    computing the circle parameters

    DET = xnew*xnew - xnew*Mz + Cov_xy
    Center = array([Mxz*(Myy-xnew)-Myz*Mxy , Myz*(Mxx-xnew)-Mxz*Mxy])/DET/2

    #Par = array([Center+centroid , sqrt(Center*Center'+Mz+2*xnew)])
    Par = hstack( (Center+centroid, sqrt(sum(Center*Center)+Mz+2*xnew)) )
    #x,y,rad
    return Par
    
def PrattPlotter(XY):
    import matplotlib.pyplot as p
    import numpy as n
    xc,yc,R = CircleFitByPratt(XY)
    fig = p.figure()
    p.plot(XY[:,0],XY[:,1],'.',linestyle='none')
    x = n.linspace( n.min(XY[:,0]),n.max(XY[:,0]),1000)
    ypos = n.sqrt( R**2 - (x-xc)**2 ) + yc
    #yneg = -n.sqrt( R**2 - (x-xc)**2 ) + yc
    p.plot(x,ypos,'r')
    p.text(.98,.02,'R = {:.3f}'.format(R),transform=fig.gca().transAxes,ha='right',va='bottom')
    #p.plot(x,yneg,'r')
    return fig.gca()
    
