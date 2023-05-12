
""" General Functions utilised by the steady state and time dependent solvers """

import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix
from matrixeqs import System
from tqdm import tqdm

pi = np.pi

def get_commutator(a,b):
    return np.dot(a,b) - np.dot(b,a)

def get_anti_commutator(a,b):
    return np.dot(a,b) + np.dot(b,a)

""" Operator creation functions """

def create_creation_operator(n):
    """ Create an annihilation operator for the n-th mode """
    return np.matrix(np.diag(np.sqrt(np.arange(1, n)), -1), dtype=complex)

def create_annihilation_operator(n):
    """ Create a creation operator for the n-th mode """
    return np.matrix(np.diag(np.sqrt(np.arange(1, n)), 1), dtype=complex)

def create_position_operator(n):
    return (create_annihilation_operator(n) + create_creation_operator(n))/np.sqrt(2)

def create_momentum_operator(n):
    return (1j)*(create_creation_operator(n) - create_annihilation_operator(n))/np.sqrt(2)

def get_function_of_operator(f, op):
    """ Return the function of the operator using Sylvester's formula """
    
    # Get eigenvalues of operator
    eigs = np.linalg.eig(op)[0]

    # Get Frobenius covariants of operators
    covs = []

    for eig in eigs:
        cov = np.ones((len(eigs), len(eigs)), dtype=complex)
        remaining = [i for i in eigs if i != eig]
        for i in remaining:
            cov *= (op - np.identity(len(eigs))*i)/(eig - i)
        covs.append(cov)

    result = np.zeros((len(eigs), len(eigs)), dtype=complex)
    for i in range(0, len(eigs)):
        result += f(eigs[i])*covs[i]
    
    return result

def create_cos_phi(phi, phi_o, phi_x, alpha): 
    """
    Create a cos(phi) operator for the n-th mode   
    """
    return get_function_of_operator(lambda x: np.cos(x), phi)
    
def create_sin_phi(phi, phi_o, phi_x, alpha):
    """
    Create a sin(phi) operator for the n-th mode     
    """
    cos_const = np.cos((2*pi)*(phi_x/phi_o))
    sin_const = np.sin((2*pi)*(phi_x/phi_o))
    cos_phi = get_function_of_operator(lambda x: np.cos(x), alpha*phi)
    sin_phi = get_function_of_operator(lambda x: np.sin(x), alpha*phi)
    return cos_const*cos_phi + sin_const*sin_phi

def validate_steady_state(handler, Lrho, n):
    """ Validates whether the steady state is correct by checking if Lrho approaches 0 """

    times = np.arange(4, 500, 50)
    lrho_sum = []
    for t in tqdm(times):
        t_i = 0
        t_f = t
        h = 1e-2
        nsteps = int((t_f-t_i)/h)
        sol = np.zeros((nsteps+1, n,n), dtype=complex)
        sol[0] = make_initial_density_matrix(n)
        
        sol = solver(sol, handler, h)
        lrho_sum.append(np.sum(np.dot(Lrho, sol[-1].reshape(n*n, 1))))

    plt.plot(times, lrho_sum)
    plt.title("Steady State Validation, sum(Lrho) vs Length of Simulation")
    plt.ylabel("sum(Lrho) = dp/dt")
    plt.xlabel("Length of Simulation")
    plt.show()

def create_phi(n):
    return create_annihilation_operator(n) + create_creation_operator(n)

def exponential_series(x, n):
    """ Returns the exponential series of x to the n-th term for matrices"""
    return np.sum([np.linalg.matrix_power(x, i)/np.math.factorial(i) for i in range(0,n)], axis=0)

def cosphi_taylor(phi, n=20):

    return (exponential_series(1j*phi, n) + exponential_series(-1j*phi, n))/2

def sinphi_taylor(phi, n=20):
        return (exponential_series(1j*phi, n) - exponential_series(-1j*phi, n))/(2j)
        
""" Miscallaneous functions """

def make_initial_density_matrix(n):
    return np.ones((n,n), dtype=complex)/n

def check_reached_steady_state(rho):
    """ Checks if the system has reached a steady state """
    # Two ways can do firstly change in last two steps is negligible
    # Or I can check if the product of rho and L is zero
    pass

def get_purity_simple(rho):
    """ Returns the purity of the density matrix """
    return np.trace(np.dot(rho, rho))

def get_trace(rho):
    return [np.trace(i) for i in rho]

def get_purity(rho):    
    return [np.trace(np.dot(i,i)) for i in rho]

def measure_pureness_state(rho):
    """ Sum of off diagonal elements of matrix """
    return (np.sum(rho) - np.sum(np.diag(rho)))

""" RK4 solver """

def RK4step(x, h, f):
    """ Runge Kutta step, x is the current state, h is the step size, f is the function to be integrated """
    k1 = f(x)
    k2 = f(x+h*k1/2)
    k3 = f(x+h*k2/2)
    k4 = f(x+h*k3)

    return x+(h/6)*(k1+2*k2+2*k3+k4)

def solver(sol_arr, f, h, plot_intervals=False):

    for i in tqdm(range(1, sol_arr.shape[0])):
        sol_arr[i] = RK4step(sol_arr[i-1], h, f)
        if plot_intervals:
            if i%20 == 0:
                plot_steady_state_td_3d(sol_arr)

    return sol_arr

def steady_state_solver(L):
    """ Solving AX = 0 for the steady state of the system, minimum solution"""
    return Matrix(L).nullspace()[0]


def null_space(A, rcond=None):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q
""" Plotting functions """

def plot_diagonal_density_matrix_elements(rho, ti=0, title="", show=True, trace_purity=True):

    fig, ax = plt.subplots(figsize=(12, 9))
    
    for i in range(0, rho.shape[1]):
        plt.plot(np.real(rho[ti:,i,i]), label = 'Re{}{}'.format(i+1, i+1))
    
    plt.xlabel('$\gamma t$')
    plt.ylabel("Probability")
    plt.legend(loc="lower right", numpoints=1,frameon=True)
    plt.title("Diagonal elements Density Matrix Dynamics {}".format(title))
    plt.show()
    return

def plot_offdiagonal_density_matrix_elements(rho, ti=0, title="", show=True, trace_purity=True):
    """ Plots the off diagonal density matrix elements """  
    
    fig, ax = plt.subplots(figsize=(12, 9))

    for i in range(0, rho.shape[1]):
        for j in range(i+1, rho.shape[1]):
            plt.plot(np.real(rho[ti:,i,j]), label = 'Re{}{}'.format(i+1, j+1))
            plt.plot(np.imag(rho[ti:,i,j]), label = 'Im{}{}'.format(i+1, j+1))

    plt.xlabel('$\gamma t$')
    plt.ylabel("Probability")
    plt.title("Dynamics of Off Diagonal Elements of the Density Matrix {}".format(title))
    plt.show()
    return

def plot_density_matrix_elements(rho, ti=0, title="", show=True, trace_purity=True):
    """ Plot the density matrix elements """

    fig, ax = plt.subplots(figsize=(12, 9))

    plt.plot(np.real(rho[ti:,0,0]), label = r'$\rho_{11}$')
    plt.plot(np.real(rho[ti:,0,1]), label = r'$\mathrm{Re}[\rho_{12}]$')
    plt.plot(np.imag(rho[ti:,0,1]), label = r'$\mathrm{Im}[\rho_{12}]$')

    if trace_purity:
        plt.plot(get_trace(rho[ti:,:,:]), label = r'$\mathrm{Tr}[\rho]$')
        plt.plot(get_purity(rho[ti:,:,:]), label = r'$\mathrm{Tr}[\rho^2]$')
    
    plt.xlabel('$\gamma t$')
    plt.legend(loc="lower right", numpoints=1,frameon=True)
    plt.title("Density Matrix Dynamics {}".format(title))
    plt.show()
    return


def plot_trace_purity(rho, title="", pureness=False):
    """ Plot the trace and purity of the density matrix """
    
    plt.plot(get_trace(rho), label = r'$\mathrm{Tr}[\rho]$')
    plt.plot(get_purity(rho), label = r'$\mathrm{Tr}[\rho^2]$')
    #plt.plot([1 - i for i in get_purity(rho)], label = r'$\mathrm{Decoherence}$')
    
    plt.legend()
    plt.title("Trace, Purity of Density Matrix {}".format(title))
    plt.show()
    return


def plot_steady_state_td_2d(rho, title=""):
    """ Image plot for Steady State Density Matrix obtained through time evolution """

    steady_state = rho[-1]
    plt.imshow(np.array(steady_state).astype(np.float64))
    plt.title("Steady State Density Matrix {}".format(title))
    plt.colorbar()
    plt.show()
    return

def plot_steady_state_td_3d(t, title=""):
    """ Surface plot for Steady State Density Matrix obtained through time evolution """
    
    steady_state = t[-1]
    n = steady_state.shape[0]
    
    X, Y = np.arange(0,n), np.arange(0, n)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, steady_state, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    plt.title("Steady State Density Matrix {}".format(title))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    return

""" Simulation functions  - to be converted to main command centre """

def run_normal_simulation(n, handler, t_i=0, t_f=200, h=1e-2):
    
    nsteps = int((t_f-t_i)/h)
    t = np.zeros((nsteps+1, n,n), dtype=complex)
    t[0] = make_initial_density_matrix(n)
    t = solver(t, handler, h)

    # Plotting
    plot_density_matrix_elements(t)
    #plot_trace_purity(t)
    #plot_diagonal_density_matrix_elements(t)
    #plot_offdiagonal_density_matrix_elements(t)
    plot_steady_state_td_2d(t)
    #plot_steady_state_td_3d(t)
    return t

def run_simulation(n, H, L, gamma, t_i=0, t_f=300, h=1e-2, title=""):
    # Setting simulation parameters

    system = System(H, L, gamma)
    nsteps = int((t_f-t_i)/h)
    t = np.zeros((nsteps+1, n,n), dtype=complex)
    t[0] = make_initial_density_matrix(n)
    
    t = solver(t, system.LinEm, h)

    plot_density_matrix_elements(t, title=title)
    plot_trace_purity(t, title=title)
    #plot_diagonal_density_matrix_elements(t, title=title)
    #plot_offdiagonal_density_matrix_elements(t, title=title)
    plot_steady_state_td_2d(t, title=title)
    plot_steady_state_td_3d(t, title=title)

    return t
