
""" General Functions utilised by the steady state and time dependent solvers """

import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

pi = np.pi

def get_commutator(a,b):
    return np.dot(a,b) - np.dot(b,a)

def get_anti_commutator(a,b):
    return np.dot(a,b) + np.dot(b,a)

""" Operator creation functions """
def create_annihilation_operator(n):
    """ Create an annihilation operator for the n-th mode """
    return np.matrix(np.diag(np.sqrt(np.arange(1, n)), -1), dtype=complex)

def create_creation_operator(n):
    """ Create a creation operator for the n-th mode """
    return np.matrix(np.diag(np.sqrt(np.arange(1, n)), 1), dtype=complex)

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
    loll this is wrong
    """
    cos_const = np.cos((2*pi)*(phi_x/phi_o))
    sin_const = np.sin((2*pi)*(phi_x/phi_o))
    cos_phi = get_function_of_operator(lambda x: np.cos(x), alpha*(phi + phi_x))
    sin_phi = get_function_of_operator(lambda x: np.sin(x), alpha*(phi + phi_x))
    return cos_phi - sin_phi 

def create_sin_phi(phi, phi_o, phi_x, alpha):
    """
    Create a sin(phi) operator for the n-th mode     
    """
    const = np.cos((2*pi)*(phi_x/phi_o))
    cos_phi = get_function_of_operator(lambda x: np.cos(x), alpha*phi + const)
    sin_phi = get_function_of_operator(lambda x: np.sin(x), alpha*phi + const)
    return cos_phi + sin_phi



def make_initial_density_matrix(n):
    return np.ones((n,n), dtype=complex)/n

""" RK4 solver """

def RK4step(x, h, f):
    """ Runge Kutta step, x is the current state, h is the step size, f is the function to be integrated """
    k1 = f(x)
    k2 = f(x+h*k1/2)
    k3 = f(x+h*k2/2)
    k4 = f(x+h*k3)

    return x+(h/6)*(k1+2*k2+2*k3+k4)

def solver(sol_arr, f, h):

    for i in tqdm(range(1, sol_arr.shape[0])):
        sol_arr[i] = RK4step(sol_arr[i-1], h, f)

    return sol_arr

def steady_state_solver(L):
    """ Solving AX = 0 for the steady state of the system, minimum solution"""
    return Matrix(L).nullspace()[0]

""" Plotting functions """

def plot_density_matrix_elements(rho, ti=0, title="", show=True, trace_purity=True):
    """ Plot the density matrix elements """

    fig, ax = plt.subplots(figsize=(12, 9))

    # Plotting density matrix elements - choose one off diagonal and one diagonal
    plt.plot(np.real(rho[ti:,1,1]), label = r'$\rho_{22}$')
    plt.plot(np.real(rho[ti:,0,1]), label = r'$\mathrm{Re}[\rho_{12}]$')
    plt.plot(np.imag(rho[ti:,0,1]), label = r'$\mathrm{Im}[\rho_{12}]$')
    purity = get_purity(rho[ti:,:,:])

    if trace_purity:
        plt.plot([np.trace(i) for i in rho[ti:,:,:]], label = r'$\mathrm{Tr}[\rho]$')
        plt.plot(purity, label = r'$\mathrm{Tr}[\rho^2]$')
    
    plt.xlabel('$\gamma t$')
    #plt.ylim(-0.5, 1.1)
    plt.legend(loc="lower right", numpoints=1,frameon=True)
    plt.title("Density Matrix Dynamics {}".format(title))

    plt.show()


def plot_steady_state_td(rho, title=""):
    steady_state = rho[-1]
    plt.imshow(np.array(steady_state).astype(np.float64))
    plt.title("Steady State Density Matrix {}".format(title))
    plt.colorbar()
    plt.show()

def get_trace(rho):
    trace = [np.trace(i) for i in rho]
    return trace

def get_purity(rho):    
    purity = [np.trace(np.dot(i,i)) for i in rho]
    return purity

def plot_trace_purity(rho, title="", show=True):
    """ Plot the trace and purity of the density matrix """

    # Calculating trace and purity
    trace = get_trace(rho)
    purity = get_purity(rho)
    pureness = [measure_pureness_state(i) for i in rho]
    
    # Plotting
    plt.plot(trace, label = r'$\mathrm{Tr}[\rho]$')
    plt.plot(purity, label = r'$\mathrm{Tr}[\rho^2]$')
    plt.plot(pureness, label = r'Sum of offdiags')
    
    plt.ylim(0,2)
    plt.legend()
    plt.title("Trace and Purity of Density Matrix {}".format(title))
    
    if show:
        plt.show()

def wigner_plot_steady_state(t, n):
    steady_state = t[-1]
    X = np.arange(0,n)
    Y = np.arange(0, n)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, steady_state, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    plt.title("Steady State Wigner Function of Density Matrix")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def measure_pureness_state(rho):
    """ Sum of off diagonal elements of matrix """
    return (np.sum(rho) - np.sum(np.diag(rho)))