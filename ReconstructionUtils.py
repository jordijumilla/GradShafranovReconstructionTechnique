import numpy as np
import math
from scipy import integrate
from scipy.interpolate import UnivariateSpline
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import warnings, traceback, sys

# Physical constants
mu_0 = np.float64(4.0 * np.pi * 1e-7)   # Permeability (H/m) = (N/A^2) = (kg*m/s^2)/A^2
AU = 149597871000 # meters

# Returns phi, theta in degrees
def spherical_angles_deg(v):
    assert len(v) == 3, "Input vector should be 3-dimensional"
    phi = math.atan2(v[1], v[0])
    if phi < 0:
        phi = phi + 2*math.pi
    theta = math.acos(v[2])
    return [np.rad2deg(phi), np.rad2deg(theta)]

def pseudospherical_angles_deg(v, e_x, e_y, e_z):
    if len(v) == 3 and len(e_x) == 3 and len(e_y) == 3 and len(e_z) == 3:
        theta = math.acos(np.dot(v, e_z)/np.linalg.norm(v))
        proj_xy = v - (np.dot(v, e_z))*e_z
        
        if np.linalg.norm(proj_xy) == 0:
            phi = 0
        else:
            comp_x = np.dot(proj_xy, e_x)
            comp_y = np.dot(proj_xy, e_y)
            phi = math.atan2(comp_y, comp_x)
            if phi < 0:
                phi = phi + 2*math.pi
        return [np.rad2deg(phi), np.rad2deg(theta)]
    
    else:
        print("Input vectors should be 3-dimensional")

def spherical_angles_deg(v):
    return pseudospherical_angles_deg(v, np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]))
        
def complete_base(e_z, V_HT):
    e_x = -V_HT + np.dot(V_HT, e_z)*e_z
    e_x = e_x/np.linalg.norm(e_x)
    e_y = np.cross(e_z, e_x)
    return [e_x, e_y]

def make_vector_deg(phi, theta):
    phi_r = np.deg2rad(phi)
    theta_r = np.deg2rad(theta)
    return np.array([math.cos(phi_r)*math.sin(theta_r), math.sin(phi_r)*math.sin(theta_r), math.cos(theta_r)])

def get_angle_deg(v1, v2):
    inner = np.dot(v1, v2)
    norms = np.linalg.norm(v1)*np.linalg.norm(v2)
    return np.rad2deg(math.acos(inner/norms))

def make_matrix_change(u_x, u_y, u_z):
    A = np.zeros((3,3))
    A[:, 0] = u_x
    A[:, 1] = u_y
    A[:, 2] = u_z
    return A

# A has the vectors of the basis in columns in the B1 base
# Multiply by A changes from B2 into B1, so we have to multiply by its inverse matrix
# Both v and A are in B1 base
def change_of_basis(v, A):
    A_inv = np.linalg.inv(A)
    return A_inv*v

def second_derivative(f, dx):
    n = len(f)
    
    d2 = [(2*f[0] - 5*f[1] + 4*f[2] - f[3])]
    for i in range(1, n - 1):
        second_deriv = (f[i - 1] - 2*f[i] + f[i + 1])
        d2 = np.append(d2, second_deriv)
    second_deriv = (f[i - 1] - 2*f[i] + f[i + 1])
    last = n - 1
    d2 = np.append(d2, 2*f[last] - 5*f[last - 1] + 4*f[last - 2] - f[last - 3])
    d2 = d2/(dx**2)
    return d2

def noise_diff(f, dx):
    n = len(f)
    
    d = [(f[1] - f[0])/dx, (f[2] - f[1])/dx]
    
    for i in range(2, len(f) - 2, 1):
        d_i = (2*(f[i+1] - f[i-1]) + (f[i+2] - f[i-2]))/(8*dx)
        d = np.append(d, d_i)
    d = np.append(d, [(f[n-2] - f[n-3])/dx, (f[n-1] - f[n-2])/dx])
    return d
        
def exponential_tail(f, f_prime, x):
    alpha = f_prime / f
    beta = math.log(f) - alpha*x
    return [alpha, beta]
    
def mean_filter(v):
    v_filt = [[0]]
    for i in range(1, len(v) - 1):
        mean = (v[i - 1] + v[i] + v[i + 1])/3
        v_filt = np.append(v_filt, mean)
    v_filt = np.append(v_filt, v[len(v) - 1])
    return v_filt