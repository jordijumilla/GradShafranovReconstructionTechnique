import numpy as np
from ReconstructionUtils import *

def w(y, y_max, alpha):
    return 1-alpha*abs(y/y_max)

def exp_filter(A, y, y_max, alpha):
    pes = w(y, y_max, alpha)
    A_filt = [A[0]]
    for i in range(1, len(A)-1):
        aux = pes*A[i] + 0.5*(1 - pes)*(A[i-1] + A[i+1])
        A_filt = np.append(A_filt, aux)
    A_filt = np.append(A_filt, (A[-1]))
    return A_filt

def solveGS(x, dx, y_points, Brec, Ax0, filterOn, alpha, PtA_polyfit_d, A_l, A_u, alpha_l, beta_l, alpha_u, beta_u):
    
    def dPtdA(*args):
        res = []
        for A in args:
            if A < A_l:
                res = np.append(res, alpha_l*np.exp(alpha_l*A + beta_l))
            elif A > A_u:
                res = np.append(res, alpha_u*np.exp(alpha_u*A + beta_u))
            else: 
                res = np.append(res, np.polyval(PtA_polyfit_d, A))
        return res
    '''
    def dPtdA(*args):
        res = []
        for A in args:
            res = np.append(res, np.polyval(PtA_polyfit_d, A))
        return res
    '''
    
    x_points = len(x)
    x_max = np.max(x)/2

    dy = dx/10

    y_max = (y_points - 1)/(x_points - 1)*x_max/10
    y_aux = np.linspace(-y_max, y_max, y_points, endpoint = True)

    #################################################################
    # Definition of A grid
    # TESTED
    gridA = np.zeros((y_points, x_points))

    gridA[y_points // 2, :] = Ax0

    # Definition of Bx grid
    gridBx = np.zeros((y_points, x_points))
    gridBx[y_points // 2, :] = Brec[:, 0]

    # Definition of By grid
    gridBy = np.zeros((y_points, x_points))
    gridBy[y_points // 2, :] = Brec[:, 1]

    ############################
    # Definition of second serivative of A wrt y grid
    gridd2Adx2 = np.zeros((y_points, x_points))    
    gridd2Adx2[y_points // 2, :] = - noise_diff(Brec[:, 1], dx)

    gridd2Ady2 = np.zeros((y_points, x_points))
    gridd2Ady2[y_points // 2, :] = - gridd2Adx2[y_points // 2, :] - mu_0*dPtdA(*Ax0)

    ###################################
    # PDE solve
    for sign in [-1, 1]:
        if sign == -1:
            fin = 0
        else:
            fin = y_points - 1
        for i in range(y_points // 2, fin, sign):
            # Update A
            gridA[i + sign, :] = gridA[i, :] - sign*gridBx[i, :]*dy + 0.5*gridd2Ady2[i, :]*dy**2
            if filterOn == 1:
                gridA[i + sign, :] = exp_filter(gridA[i + sign, :], dy*(i - (y_points // 2)), y_max, alpha)

            # Update Bx
            gridBx[i + sign, :] = gridBx[i, :] - sign*gridd2Ady2[i, :]*dy
            gridBy[i + sign, :] = - noise_diff(gridA[i + sign, :], dx)
            
            if filterOn == 1:
                gridBx[i + sign, :] = exp_filter(gridBx[i + sign, :], dy*(i - (y_points // 2)), y_max, alpha)
                gridBy[i + sign, :] = exp_filter(gridBy[i + sign, :], dy*(i - (y_points // 2)), y_max, alpha)

            # Update second derivative wrt to x & y
            gridd2Adx2[i + sign, :] = second_derivative(gridA[i + sign, :], dx)
            if filterOn == 1:
                gridd2Adx2[i + sign, :] = exp_filter(gridd2Adx2[i + sign, :], dy*(i - (y_points // 2)), y_max, alpha)
            
            gridd2Ady2[i + sign, :] = - gridd2Adx2[i + sign, :] - mu_0*dPtdA(*gridA[i + sign, :])
            if filterOn == 1:
                gridd2Ady2[i + sign, :] = exp_filter(gridd2Ady2[i + sign, :], dy*(i - (y_points // 2)), y_max, alpha)
    
    return [x_max, y_max, gridA, gridBx, gridBy, gridd2Adx2, gridd2Ady2]

def getCenter(gridBz, dx, dy):
    indexMaxBz = np.argmax(gridBz)
    x_points = len(gridBz[0, :])
    y_points = len(gridBz[:, 0])

    i_center = indexMaxBz//x_points
    j_center = indexMaxBz%x_points
    x_center = j_center*dx
    y_center = (y_points//2 - i_center)*dy
    y_0 = (i_center - y_points/2)*dy
    return [i_center, j_center, x_center, y_center, y_0]

def plotGSSolution(x_plot, y_plot, gridA, gridBx, gridBy, gridd2Ady2, gridBz, J_z, path):
    x_points = len(x_plot)
    y_points = len(y_plot)
    delta_x = x_plot[1] - x_plot[0]
    delta_y = y_plot[1] - y_plot[0]
    
    X, Y = np.meshgrid(x_plot/AU, y_plot/AU)
    
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 25))

    mycmap1 = plt.get_cmap('gnuplot2')
    ax1.set_aspect('equal')
    ax1.set_title('Reconstructed $A(x, y) (T·m)$')
    cf1 = ax1.contourf(X, Y, gridA, levels = 20, cmap=mycmap1)
    fig.colorbar(cf1, ax = ax1)

    mycmap2 = plt.get_cmap('gnuplot2')
    ax2.set_aspect('equal')
    ax2.set_title('Reconstructed $B_x(x, y)$ (nT)')
    cf2 = ax2.contourf(X, Y, gridBx*1e9, levels = 15, cmap=mycmap2)
    fig.colorbar(cf2, ax = ax2)

    mycmap3 = plt.get_cmap('gnuplot2')
    ax3.set_aspect('equal')
    ax3.set_title('Reconstructed $B_y(x, y) (nT)$')
    cf3 = ax3.contourf(X, Y, gridBy*1e9, levels = 15, cmap=mycmap3)
    fig.colorbar(cf3, ax = ax3)

    mycmap4 = plt.get_cmap('gnuplot2')
    ax4.set_aspect('equal')
    ax4.set_title('Reconstructed $d^2A/dy^2(x, y)$ (nT/m^2)')
    cf4 = ax4.contourf(X, Y, -gridd2Ady2/mu_0, levels = 15, cmap=mycmap4)
    fig.colorbar(cf4, ax = ax4)
    
    [i_center, j_center, x_center, y_center, y_0] = getCenter(gridBz, delta_x, delta_y)

    mycmap5 = plt.get_cmap('gnuplot2')
    ax5.set_aspect('equal')
    ax5.set_title('Reconstructed $B_z(x, y)$ (nT)')
    cf5 = ax5.contourf(X, Y, gridBz*1e9, levels = 20, cmap=mycmap5)
    fig.colorbar(cf5, ax = ax5)
    ax5.plot(j_center*delta_x/AU, (i_center-y_points//2)*delta_y/AU, marker="o", markersize=5, markeredgecolor="black", color = "white")

    mycmap6 = plt.get_cmap('gnuplot2')
    ax6.set_aspect('equal')
    ax6.set_title('Density current $j_z(x, y)$ (pA)')
    cf6 = ax6.contourf(X, Y, J_z*1e12, levels = 25, cmap=mycmap6)
    fig.colorbar(cf6, ax = ax6)
    plt.savefig(path + '/Reconstruction solution.png')
    plt.show()
    
def getJxJy(x_points, dx, y_points, dy, gridBx, gridBy, gridBz): 
    # Current density
    griddB_z_dx = np.zeros((y_points, x_points))
    for i in range(y_points):
        griddB_z_dx[i, :] = noise_diff(gridBz[i, :], dx)

    griddB_z_dy = np.zeros((y_points, x_points))
    for j in range(x_points):
        griddB_z_dy[:, j] = - noise_diff(gridBz[:, j], dy)

    gridJx = griddB_z_dy/mu_0
    gridJy = -griddB_z_dx/mu_0
    #gridJz = (griddB_y_dx - griddB_x_dy)/mu_0
    return [gridJx, gridJy]

def plotCrossSection(x, y, gridMag, factor, arrowsOn, x_arrow, y_arrow, center, numLvls, title):
    X, Y = np.meshgrid(x/AU, y/AU)

    fig, ax = plt.subplots(figsize=(10, 6), dpi = 200, tight_layout=True) #18, 10

    mycmap1 = plt.get_cmap('gnuplot2')
    ax.set_aspect('equal')
    cf1 = ax.contourf(X, Y, gridMag*factor, levels = 2*numLvls, cmap=mycmap1)
    cf2 = ax.contour(X, Y, gridMag*factor, levels = numLvls, colors='black', linewidths = 0.5)
    
    # Arrows
    if arrowsOn:
        maxBxy = np.max(np.sqrt(x_arrow**2 + y_arrow**2))
        minLen = min(np.max(x), np.max(y))/(2*AU)

        for i in range(len(x)):
            ax.arrow(x[i]/AU, 0, 0.15*minLen*x_arrow[i]/maxBxy, 0.15*minLen*y_arrow[i]/maxBxy, width = 0.0008, lw = 0.5, facecolor = 'w', edgecolor='k')  
    ax.plot(center[0]/AU, center[1]/AU, marker="o", markersize=8, markeredgecolor="black", color = "white")
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_title(title)

    fig.colorbar(cf1)
    
def getDiameter(i_correct, j_correct, dx, dy):
    D = 0
    i_1 = 0
    i_2 = 0
    j_1 = 0
    j_2 = 0

    for k in range(len(i_correct)):
        for p in range(len(i_correct)):
            d = np.sqrt((dx*(j_correct[k] - j_correct[p]))**2 + (dy*(i_correct[k] - i_correct[p]))**2)
            if d > D:
                D = d
                i_1 = i_correct[k]
                i_2 = i_correct[p]
                j_1 = j_correct[k]
                j_2 = j_correct[p]
    return [D, i_1, j_1, i_2, j_2]

def getNumTurns(x0, y0, dl, numIter, x_plot, y_plot, center, gridBx, gridBy, gridBz, plotOn = True):
    # Number of turns
    from scipy import interpolate
    interpBx = interpolate.interp2d(x_plot, y_plot, gridBx, kind='cubic')
    interpBy = interpolate.interp2d(x_plot, y_plot, gridBy, kind='cubic')
    interpBz = interpolate.interp2d(x_plot, y_plot, gridBz, kind='cubic')

    x_center = center[0]
    y_center = center[1]
    
    x_pos = np.zeros(numIter)
    y_pos = np.zeros(numIter)
    z_pos = np.zeros(numIter )
    phi_pos = np.zeros(numIter)

    x_pos[0] = x0
    y_pos[0] = y0
    z_pos[0] = 0
    phi_pos[0] = math.atan2(y_pos[0] - y_center, x_pos[0] - x_center)
    nTurns = 0

    for k in range(1, numIter):
        Bx = interpBx(x_pos[k - 1], y_pos[k - 1])*1e9
        By = interpBy(x_pos[k - 1], y_pos[k - 1])*1e9
        Bz = interpBz(x_pos[k - 1], y_pos[k - 1])*1e9
        modB = math.sqrt(Bx**2 + By**2 + Bz**2)

        x_pos[k] = x_pos[k - 1] + (Bx/modB)*dl
        y_pos[k] = y_pos[k - 1] + (By/modB)*dl
        z_pos[k] = z_pos[k - 1] + (Bz/modB)*dl

        phi_pos[k] = math.atan2(y_pos[k] - y_center, x_pos[k] - x_center)

        if phi_pos[k]*phi_pos[k-1] < -2*math.pi + 0.01:
            if phi_pos[k] < 0:
                nTurns = nTurns + 1
            else:
                nTurns = nTurns - 1

    totalAngle = phi_pos[-1] - phi_pos[0] + 2*math.pi*nTurns
    totalTurns = totalAngle/(2*math.pi)

    twist = totalAngle/z_pos[-1]
    
    if plotOn == True:
        fig, ax = plt.subplots(figsize=(10, 6), dpi = 200, tight_layout=True)
        plt.plot(z_pos/AU, x_pos/AU, '-r', label = 'x')
        plt.plot(z_pos/AU, y_pos/AU, '-g', label = 'y')
        ax.set_title('x-y coordinates of magnetic field line')
        ax.set_xlabel('z (AU)')
        ax.set_ylabel('Distance (AU)')
        ax.legend(facecolor = 'white')
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6), dpi = 200, tight_layout=True)
        plt.plot(np.linspace(0, numIter - 1, numIter, endpoint = True), np.rad2deg(phi_pos), '-m')
        plt.axhline(y = 180, linestyle = '--', alpha = 0.3)
        plt.axhline(y = -180, linestyle = '--', alpha = 0.3)

        ax.set_title('$\\varphi$ angle in the x-y plane')
        ax.set_xlabel('z (AU)')
        ax.set_ylabel('$\\varphi$ (deg)')
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6), dpi = 200, tight_layout=True)
        plt.plot(x_pos/AU, y_pos/AU, '-k')
        ax.set_title('Magnetic field line projection in the x-y plane')
        ax.set_xlabel('x (AU)')
        ax.set_xlabel('y (AU)')
        ax.set_aspect('equal')
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 8), dpi = 200, tight_layout=True)
        ax = plt.axes(projection='3d')
        plt.plot(x_pos[0]/AU, y_pos[0]/AU, z_pos[0]/AU, 'og', label = 'Starting point')
        plt.plot(x_pos[-1]/AU, y_pos[-1]/AU, z_pos[-1]/AU, 'ob', label = 'Ending point')
        ax.plot3D(x_pos/AU, y_pos/AU, z_pos/AU, 'k')
        ax.set_xlabel('$x (AU)$')
        ax.set_ylabel('$y (AU)$')
        ax.set_zlabel('$z (AU)$')
        ax.set_title('Magnetic field line')
        ax.legend(facecolor = 'white')
        plt.show()
    return twist