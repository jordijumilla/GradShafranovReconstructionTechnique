import numpy as np
import math
from scipy import integrate
from scipy.interpolate import UnivariateSpline
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from ReconstructionUtils import *

def has_inflection(a):
    if np.all(np.diff(a) > 0) or np.all(np.diff(a) < 0):
        return False
    return True
    
def get_2_branches(A, P_t):
    index_A_max = np.argmax(np.abs(A))
   
    # First half of trejectory
    A_first = A[0 : index_A_max + 1]
    P_first = P_t[0 : index_A_max + 1]

    sorting_index_first = np.argsort(A_first)
    A_first = np.array(A_first)[sorting_index_first]
    P_first = np.array(P_first)[sorting_index_first]

    # Second half of trejectory
    A_second = A[index_A_max : len(A)]
    P_second = P_t[index_A_max : len(A)]

    sorting_index_second = np.argsort(A_second)
    A_second = np.array(A_second)[sorting_index_second]
    P_second = np.array(P_second)[sorting_index_second]
    
    #A_l = np.max([np.min(A_first), np.min(A_second)])
    '''
    A_first_aux = A_first[A_first >= A_l]
    P_first = P_first[A_first >= A_l]
    A_second_aux = A_second[A_second >= A_l]
    P_second = P_second[A_second >= A_l]
    '''
    return [A_first, P_first, A_second, P_second]

def residueMap(B_rtn, p, seconds, V_HT, m_0, phi_range, theta_range, plot):
    n_x = len(seconds)
    
    num_phi = round((phi_range[1] - phi_range[0])/(phi_range[2])) + 1
    num_theta = round((theta_range[1] - theta_range[0])/(theta_range[2])) + 1
    
    phi_values = np.linspace(phi_range[0], phi_range[1], num = num_phi, endpoint = True)
    theta_values = np.linspace(theta_range[0], theta_range[1], num = num_theta, endpoint = True)
    
    res = np.zeros((3, 0))
    
    for phi in phi_values:
        for theta in theta_values:
            e_z = make_vector_deg(phi, theta) # In the intermediate base
            [e_x, e_y] = complete_base(e_z, V_HT)
            A_basis = make_matrix_change(e_x, e_y, e_z)

            B_res = np.dot(B_rtn, A_basis)
            x = np.dot(-V_HT, e_x)*seconds

            Ax0 = integrate.cumtrapz(-B_res[:, 1], x, initial = 0)
            
            if not has_inflection(Ax0):
                res = np.append(res, [[phi], [theta], [0]], axis = 1)
            else:
                P_mag = (pow(B_res[:,2], 2))/(2*mu_0)
                P_t =  (p + P_mag).to_numpy() 

                [A_first, P_first, A_second, P_second] = get_2_branches(Ax0, P_t)
                A_m = np.max(Ax0)
                A_l = max(np.min(A_first), np.min(A_second))
                A_res = np.linspace(A_l, A_m, num = 25, endpoint = True)

                len_1 = len(A_first)
                len_2 = len(A_second)

                if len_1 <= n_x/4 or len_2 <= n_x/4:
                    res = np.append(res, [[phi], [theta], [0]], axis = 1)
                else:                            
                    # Straight lines interpolation
                    PA_spl_1 = UnivariateSpline(A_first, P_first, s = 0, k = 1)
                    PA_spl_2 = UnivariateSpline(A_second, P_second, s = 0, k = 1)

                    # Stencil of A where the residue is calculated
                    A_res = np.linspace(A_l, A_m, num = m_0, endpoint = True)

                    P_min = np.min([np.min(PA_spl_1(A_res)), np.min(PA_spl_2(A_res))])
                    P_max = np.max([np.max(PA_spl_1(A_res)), np.max(PA_spl_2(A_res))])

                    residue = np.linalg.norm(PA_spl_1(A_res) - PA_spl_2(A_res))/((P_max - P_min)*m_0)

                    new_column = [[phi], [theta], [1/residue]]
                    res = np.append(res, new_column, axis = 1)
                    if plot == 1:
                        print('phi =', phi, ', theta =', theta, ', residue =', residue)
                        # Plots
                        figure(figsize=(8, 6), dpi = 80)
                        plt.plot(Ax0, P_t*1e9, 'x', color = 'k')
                        plt.plot(A_first, P_first*1e9, 'x', color = 'r')
                        plt.plot(A_res, PA_spl_1(A_res)*1e9, '-', color = 'r')

                        plt.plot(A_second, P_second*1e9, 'x', color = 'g')
                        plt.plot(A_res, PA_spl_2(A_res)*1e9, '-', color = 'g')

                        plt.axhline(y = P_min, linestyle = '--', alpha = 0.3)
                        plt.axhline(y = P_max, linestyle = '--', alpha = 0.3)
                        plt.legend(['1st half of trajectory measures', '1st half of trajectory spline', '2nd half of trajectory measures', '2nd half of trajectory spline'])
                        plt.xlabel('$A(x,0) (T·m)$')
                        plt.ylabel('$P_t(x,0)$ (nPa)')
                        plt.title('Transverse pressure as a function of magnetic potential')

                        plt.show()
    return res

def getMinResidue(phi_res, theta_res, invresidue):
    index_min_residue = np.argmax(invresidue)
    phi_min = phi_res[index_min_residue]
    theta_min = theta_res[index_min_residue]
    residue_min = 1/invresidue[index_min_residue]
        
    return [phi_min, theta_min, residue_min]

# Plot the residue
def plotResidueMap(phi_res, theta_res, invresidue, e_z_MVAB, e_z_MVUB):
    [phi_min, theta_min, residue_min] = getMinResidue(phi_res, theta_res, invresidue)
    
    max_theta = np.max(theta_res)
    max_phi = np.deg2rad(np.max(phi_res))
    number_points = len(invresidue)

    points = np.zeros((number_points, 2))
    points[:, 0] = theta_res
    points[:, 1] = np.deg2rad(phi_res)

    values = invresidue
    
    # Intermediate variance
    [phi_MVAB, theta_MVAB] = spherical_angles_deg(e_z_MVAB)
    [phi_MVUB, theta_MVUB] = spherical_angles_deg(e_z_MVUB)
    
    # We create a grid of values, interpolated from our random sample above
    phi = np.linspace(0, max_phi, 720)
    theta = np.linspace(0, max_theta, 360)
    grid_theta, grid_phi = np.meshgrid(theta, phi)
    data = griddata(points, values, (grid_theta, grid_phi), method='cubic', fill_value = 0)

    #Create a polar projection
    # First component is phi (rad)
    # Second component is theta (deg)
    #fig = plt.figure(figsize=(8, 6), dpi=200)
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

    #ax = plt.subplot(projection="polar")
    ax.plot(np.deg2rad(phi_MVAB), theta_MVAB, 'xk', markersize = 8)
    ax.plot(np.deg2rad(phi_MVUB), theta_MVUB, 'xr', markersize = 4)
    
    ax.plot(np.deg2rad(phi_min), theta_min, 'ow', markersize = 4)
    #polar_plot = ax.pcolormesh(phi, theta, data.T)
    polar_plot = ax.contourf(phi, theta, data.T, levels = 100)
    
    ax.grid(linewidth=15)

    fig.colorbar(polar_plot)
    plt.show()
    
def intermedVar(A):
    M = np.cov(A.transpose()*1e9)
    eigenvalues, W = np.linalg.eig(M)

    # Finding the intermediate eigenvalue & corresponding eigenvector
    index_min = np.argmin(eigenvalues)
    index_max = np.argmax(eigenvalues)
    index_intermediate = 3 - index_min - index_max

    inter = W[:, index_intermediate]
    return inter
    
    
def plotFrame(e_x, e_y, e_z, vHT, path):
    plt.figure(figsize=(8,8), dpi = 200, tight_layout=True)
    ax = plt.axes(projection = '3d')
    ax.view_init(30, 30)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title('Reconstruction frame')

    vHT_normalized = vHT/np.linalg.norm(vHT)
    point = ax.plot(0, 0, 0, 'yp', markersize = 8)
    aux1 = ax.quiver(0, 0, 0, e_x[0], e_x[1], e_x[2], color = 'r')
    aux2 = ax.quiver(0, 0, 0, e_y[0], e_y[1], e_y[2], color = 'g')
    aux3 = ax.quiver(0, 0, 0, e_z[0], e_z[1], e_z[2], color = 'b')
    aux4 = ax.quiver(0, 0, 0, -vHT_normalized[0], -vHT_normalized[1], -vHT_normalized[2], color = 'black')
    ax.legend([aux1, aux2, aux3, aux4], ['$e_x$', '$e_y$', '$e_z$', '$-v_{HT}$'])
    plt.savefig(path + '/Reconstrction frame.png')