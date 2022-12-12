
""" Steady state solver for the SQUID model """

import numpy as np
import matplotlib.pyplot as plt

""" Operator creation functions """
def create_annihilation_operator(n):
    """ Create an annihilation operator for the n-th mode """
    return np.diag(np.sqrt(np.arange(1, n)), -1)

def create_creation_operator(n):
    """ Create a creation operator for the n-th mode """
    return np.diag(np.sqrt(np.arange(1, n)), 1)

def get_function_of_operator(f, op):
    """ Return the function of the operator using Sylvester's formula """
    
    # Get eigenvalues of operator
    eigs = np.linalg.eig(op)[0]

    # Get Frobenius covariants of operators
    covs = []

    for eig in eigs:
        cov = np.ones((len(eigs), len(eigs)))
        remaining = [i for i in eigs if i != eig]
        for i in remaining:
            cov *= (op - np.identity(len(eigs))*i)/(eig - i)
        covs.append(cov)

    result = np.zeros((len(eigs), len(eigs)))
    for i in range(0, len(eigs)):
        result += f(eigs[i])*covs[i]
    
    return result

def create_cos_phi(n, phi, phi_o, phi_x):
    """ Create a cos(phi) operator for the n-th mode """
    cos_phi_1 = (2*3.14/phi_o)*phi - 2*3.14*(phi_x/phi_o)*np.identity(n)
    return get_function_of_operator(lambda x: np.cos(x), phi)

""" Constants """
n = 2 # Hilbert Space Dimension
C = 5e-15 # Capacitance 
wo = 8.16e11 # Oscillator frequency
phi_o = 0.4 # Flux quantum
phi_x = 0.2 # Flux quantum
gamma = 0.005 * wo # Relaxation Rate
hbar = 6.6e-34 # hbar
l = 3e-10 # Inductance
cutoff = 20 # Cutoff frequency
mu = wo # Frequency for cos phi term, not sure if that value is valid
m = 1 # Mass

""" Operators """
adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator
phi = np.sqrt(hbar/(2*m*wo)) * (adag + a) # Flux operator (analogous to position operator)
P = np.sqrt(hbar*m*wo/(2)) * (1j)* (adag - a) # Momentum operator 


cos_phi = create_cos_phi(n, phi, phi_o, phi_x) # cos(phi) operator
print(cos_phi)
H = 0.5 * ((phi * phi)/l + (P * P)/C + hbar*mu*(cos_phi * cos_phi)) # Hamiltonian
L = np.sqrt((C*wo*gamma)/(hbar)) * phi + np.sqrt((gamma)/(C*wo*hbar)) * (1j - (wo/2*cutoff)) * P # Lindblad operator
Ldag = np.conj(L.T) # Conj transpose of Lindblad


""" Steady state """

def first_order_equation():
    """ First order equation for steady state """
    hamiltonian_part = -(1j/hbar) * (np.kron(H, np.identity(n)) - np.kron(np.identity(n), H))
    lindblad_part_1 = 2*( np.kron(np.identity(n), (np.dot(Ldag, L))) - np.kron(Ldag.T, L) ) 
    lindblad_part_2 = - np.kron((np.dot(Ldag, L)), np.identity(n)) 
    return hamiltonian_part + lindblad_part_1 + lindblad_part_2

def steady_state_solver():
    """ Steady state solver """
    
    superop = first_order_equation()
    zeros = np.zeros((n*n, 1))
    p = np.linalg.solve(superop, zeros)
    p = p.reshape((n, n))
    print(p)
    return p

#if __name__ == "__main__":
    





