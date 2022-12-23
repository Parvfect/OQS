
""" General Functions utilised by the steady state and time dependent solvers """

import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix

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
    """
    cos_const = np.cos((2*pi)*(phi_x/phi_o))
    sin_const = np.sin((2*pi)*(phi_x/phi_o))
    cos_phi = get_function_of_operator(lambda x: np.cos(x), alpha*phi)
    sin_phi = get_function_of_operator(lambda x: np.sin(x), alpha*phi)
    return cos_const*cos_phi - sin_const*sin_phi




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

    for i in range(1, sol_arr.shape[0]):
        sol_arr[i] = RK4step(sol_arr[i-1], h, f)

    return sol_arr

def steady_state_solver(L):
    """ Solving AX = 0 for the steady state of the system, minimum solution"""
    return Matrix(L).nullspace()[0]

""" Plotting functions """

def plot_density_matrix_elements(rho, title=""):
    """ Plot the density matrix elements """

    fig, ax = plt.subplots(figsize=(12, 9))

    # Plotting density matrix elements - choose one off diagonal and one diagonal
    plt.plot(np.real(rho[:,1,1]), label = r'$\rho_{22}$')
    plt.plot(np.real(rho[:,1,1]+rho[:,0,0]), label = r'$\mathrm{Tr}[\rho]$')
    plt.plot(np.real(rho[:,0,1]), label = r'$\mathrm{Re}[\rho_{12}]$')
    plt.plot(np.imag(rho[:,0,1]), label = r'$\mathrm{Im}[\rho_{12}]$')

    plt.xlabel('$\gamma t$')
    #plt.ylim(-0.5, 1.1)
    plt.legend(loc="lower right", numpoints=1,frameon=True)
    plt.title("Density Matrix Dynamics {}".format(title))

    plt.show()

def plot_trace_purity(rho, title=""):
    """ Plot the trace and purity of the density matrix """

    # Calculating trace and purity
    trace = [np.trace(i) for i in rho]
    purity = [np.trace(np.dot(i,i)) for i in rho]
    
    # Plotting
    plt.plot(trace, label = r'$\mathrm{Tr}[\rho]$')
    plt.plot(purity, label = r'$\mathrm{Tr}[\rho^2]$')
    plt.legend()
    plt.title("Trace and Purity of Density Matrix {}".format(title))
    plt.show()

