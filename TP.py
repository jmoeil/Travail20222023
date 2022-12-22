import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import matplotlib.animation as animation


'''
Unless stated otherwise, all the code below is my own content.
'''

def d1_mat(nx,dx):
    '''
    Constructs the second-order centered first-order derivative.
    
    Parameters
    ----------
    nx : integer
        Number of grid points
    dx : float
        Grid spacing

    Returns
    ----------
    d1_mat : np.ndarray
        Matrix to compute the second-order centered first-order derivative
    '''
    diagonals = [-1/2,1/2]
    offsets = [-1,1]

    return diags(diagonals, offsets, shape=(nx, nx)).A / dx

def d1_mat_periodic(nx,dx):
    '''
    Constructs the second-order centered first-order derivative with question 3 periodic conditions.
    
    Parameters
    ----------
    nx : integer
        Number of grid points
    dx : float
        Grid spacing

    Returns
    ----------
    d3_mat : np.ndarray
        Matrix to compute the second-order centered first-order derivative with question 3 periodic conditions.
    '''
    d1 = d1_mat(nx,dx) * dx
    d1[0,-2] = -1/2
    d1[-1,1] = 1/2

    return d1 / dx

def d3_mat(nx,dx):
    '''
    Constructs the second-order centered third-order derivative.
    
    Parameters
    ----------
    nx : integer
        Number of grid points
    dx : float
        Grid spacing

    Returns
    ----------
    d3_mat : np.ndarray
        Matrix to compute the second-order centered third-order derivative
    '''
    diagonals = [-1/2,1,-1,1/2]
    offsets = [-2,-1,1,2]

    return diags(diagonals, offsets, shape=(nx, nx)).A / dx**3

def d3_mat_periodic(nx,dx):
    '''
    Constructs the second-order centered third-order derivative with question 4 periodic conditions.
    
    Parameters
    ----------
    nx : integer
        Number of grid points
    dx : float
        Grid spacing

    Returns
    ----------
    d3_mat : np.ndarray
        Matrix to compute the second-order centered third-order derivative with question 4 periodic conditions.
    '''
    d3 = d3_mat(nx,dx) * dx**3
    d3[0,-3] = -1/2
    d3[0,-2] = 1
    d3[1,-2] = -1/2
    d3[-2,1] = 1/2
    d3[-1,1] = -1
    d3[-1,2] = 1/2

    return d3 / dx**3

def kdv_rhs(u,d1m2nd,d3m2nd):
    '''
    Constructs the solution at each stencil at constant time.
    
    Parameters
    ----------
    u : np.ndarray
        Vector containing the solution at a certain grid point $j$.

    Returns
    ----------
    u : np.ndarray
        Vector containing the solution at the grid point $j+1$.
    '''
    return -6*u*(d1m2nd @ u) - d3m2nd @ u

def rk4(u,f,dt,*args):
    '''
    Finds the solution at the next time at constant stencil.
    
    Parameters
    ----------
    u : np.ndarray
        Vector containing the solution at constant spatial grid and at a time spatial grid j.
    f : python function
        Performs the time-independant computations
    dt : float
        Time step

    Returns
    ----------
    np.ndarray : Vector containing the solution at constant spatial grid and at a time spatial grid j+1.
    '''
    k1 = f(u,*args)
    k2 = f(u+dt*k1/2,*args)
    k3 = f(u+dt*k2/2,*args)
    k4 = f(u+k3*dt,*args)
    return u + dt/6 *(k1+2*(k2+k3)+k4)
    
def soliton(x,t,c,a):
    '''
    Constructs the exact solution array.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid points
    t : np.ndarray
        Temporal grid points
    c : Integer
        Multiplicative constant
    a : Integer
        Multiplicative constant

    Returns
    ----------
    soliton : np.ndarray
        Initial value considered
    '''
    return c/2 * np.cosh(np.sqrt(c)/2*(x-c*t-a))**(-2)

def solution_soliton(x,dx,nx,dt,nt,a,c,e,c2,a2):
    '''
    Constructs the solution array to the problem at-hand.
    
    Parameters
    ----------
    x : np.ndarray
        Spatial grid points
    dx : float
        Spatial grid spacing
    nx : integer
        Number of spatial grid points
    dt : float
        Temporal grid spacing
    nt : integer
        Number of temporal grid points
    a : Integer
        Multiplicative constant
    c : Integer
        Multiplicative constant
    e : Boolean argument
        Value set to either 0 or 1. Set the problem to have either 1 or 2 solitons.
    c2 : Integer
        Multiplicative constant
    a2 : Integer
        Multiplicative constant
    Returns
    ----------
    u : np.ndarray
        Solution array
    '''
    d1m2nd = d1_mat(nx,dx)
    d3m2nd = d3_mat(nx,dx)

    u = np.empty((nt+1,nx))
    u[0] = soliton(x,0,c,a) + e*soliton(x,0,c2,a2)

    for i in range(nt):
        u[i+1] = rk4(u[i],kdv_rhs,dt,d1m2nd,d3m2nd)
    return u

def q3_solution_periodic(x,dx,nx,dt,nt):
    '''
    Constructs the solution array to the problem at-hand with periodic conditions.
    
    Parameters
    ----------
    x : np.ndarray
        Spatial grid points
    dx : float
        Spatial grid spacing
    nx : integer
        Number of spatial grid points
    dt : float
        Temporal grid spacing
    nt : integer
        Number of temporal grid points
    Returns
    ----------
    u : np.ndarray
        Solution array at time-grid point j+1.
    '''
    d1m2nd = d1_mat_periodic(nx,dx)
    d3m2nd = d3_mat_periodic(nx,dx)

    u = np.empty((nt+1,nx))
    u[0] = 10/3 * np.cos(np.pi*x/20)
    for i in range(nt):
        u[i+1] = rk4(u[i],kdv_rhs,dt,d1m2nd,d3m2nd)
    return u

def rhs_q4_rk4(u,d1m2nd):
    '''
    Computes the RK4-part of the RHS in Problem 4.

    Parameters
    ----------
    u : np.ndarray
        Solution array
    d1m2nd : np.ndarray
        Second-order finite difference approximation for the first-derivative
    Returns
    ----------
    d1m2nd@u**2 : np.ndarray
        RK4-part of the RHS in Problem 4.
    '''
    return (d1m2nd@u**2)

def q4_rk4(u,f,dt,*args):
    '''
    Finds the solution at the next time at constant stencil.
    
    Parameters
    ----------
    u : np.ndarray
        Vector containing the solution at constant spatial grid and at a time spatial grid j.
    f : python function
        Performs the time-independant computations
    dt : float
        Time step

    Returns
    ----------
    np.ndarray : Vector containing the solution at constant spatial grid and at a time spatial grid j+1.
    '''
    k1 = f(u,*args)
    k2 = f(u+dt*k1/2,*args)
    k3 = f(u+dt*k2/2,*args)
    k4 = f(u+k3*dt,*args)
    return (k1+2*(k2+k3)+k4)/6

def q4_solution_periodic(x,dx,nx,dt,nt):
    '''
    Constructs the solution array to Problem 4 at-hand with periodic conditions.
    
    Parameters
    ----------
    x : np.ndarray
        Spatial grid points
    dx : float
        Spatial grid spacing
    nx : integer
        Number of spatial grid points
    dt : float
        Temporal grid spacing
    nt : integer
        Number of temporal grid points
    Returns
    ----------
    q4 : np.ndarray
        Solution array at time-grid point j+1.
    '''
    A = d3_mat_periodic(nx,dx)
    I = np.eye(nx,nx)
    k = np.linalg.inv(I+dt*A)
    B = d1_mat_periodic(nx,dx)
    q4 = np.empty((nt+1,nx))
    q4[0] = 10/3 * np.cos(np.pi*x/20)
    for i in range(nt):
        q4[i+1] = k@(q4[i]-3*dt*q4_rk4(q4[i],rhs_q4_rk4,dt,B))
    return q4